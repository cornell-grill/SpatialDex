import argparse
import json
import os
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer, SiglipVisionTransformer

from spatialdex.data.wuji_dataset import WujiDataset
from spatialdex.model import SpatialDex, load_weights
from spatialdex.model.action.dit import DiT

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TRAINING_ONLY = ("geo_prediction.", "backbone.language_model.lm_head.")


def load_config(task, config_dir):
    path = Path(config_dir) / f"{task}.json"
    config = json.loads(path.read_text())
    config["statistics_path"] = str(path.parent / config["statistics_path"])
    vlm = config["vlm"]["pretrained_model_name_or_path"]
    if not Path(vlm).is_absolute():
        config["vlm"]["pretrained_model_name_or_path"] = str(REPO_ROOT / vlm)
    return config


def lr_multiplier(step, warmup_steps, decay_start, decay_end, min_ratio):
    if step <= 0:
        return 0.0
    if step < warmup_steps:
        return step / warmup_steps
    if step <= decay_start:
        return 1.0
    if step >= decay_end:
        return min_ratio
    return 1 - (1 - min_ratio) * (step - decay_start) / (decay_end - decay_start)


def parameter_groups(model, training):
    groups = {(b, d): [] for b in (True, False) for d in (True, False)}
    for name, param in model.named_parameters():
        if param.requires_grad:
            no_decay = param.ndim <= 1 or name.endswith(".bias") or "cognition_token" in name
            groups[("backbone" in name, not no_decay)].append(param)
    out = []
    for (backbone, decay), params in groups.items():
        lr = training["learning_rate"] if backbone else training["action_model_learning_rate"]
        out.append(dict(params=params, lr=lr, weight_decay=training["weight_decay"] if decay else 0.0))
    return out


def worker_init(worker_id):
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(model, path):
    policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, policy):
        state = model.state_dict()
    if dist.get_rank() == 0:
        state = {k: v.detach().float().contiguous() for k, v in state.items() if not k.startswith(TRAINING_ONLY)}
        save_file(state, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pretrained", default=str(REPO_ROOT / "weights" / "pretrain.safetensors"))
    parser.add_argument("--data_root", default=str(REPO_ROOT / "data" / "wuji_tianji"))
    parser.add_argument("--config_dir", default=str(REPO_ROOT / "configs" / "wuji_tianji"))
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    config = load_config(args.task, args.config_dir)
    training = config["training"]
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(config, indent=2))

    model = SpatialDex(config, training_heads=True)
    load_weights(model, args.pretrained, strict=False)
    model.use_bf16 = True
    model.backbone.config.use_cache = False
    model.vision_tower.requires_grad_(False)
    model.train()

    dataset = WujiDataset(Path(args.data_root) / args.task / "train", config, config["statistics_path"])
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=SEED, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=training["batch_size"],
        sampler=sampler,
        collate_fn=dataset.collate,
        num_workers=args.num_workers,
        worker_init_fn=worker_init,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    model.sonata_encoder.float().to(device)
    ignored = list(model.sonata_encoder.parameters()) + [p for p in model.parameters() if p.ndim == 0]
    for param in ignored:
        param.data = param.data.to(device)
        if param.requires_grad:
            dist.broadcast(param.data, src=0)
    synced = [p for p in ignored if p.requires_grad]
    model = FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy({DiT, SiglipEncoderLayer, SiglipVisionTransformer, Gemma2DecoderLayer}),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        device_id=device,
        limit_all_gathers=True,
        use_orig_params=True,
        ignored_states=ignored,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda m: isinstance(m, Gemma2DecoderLayer),
    )
    optimizer = torch.optim.AdamW(parameter_groups(model, training), betas=(0.9, 0.999))
    schedule = partial(
        lr_multiplier,
        warmup_steps=training["warmup_steps"],
        decay_start=training["decay_start"],
        decay_end=training["decay_end"],
        min_ratio=training["min_lr_ratio"],
    )
    scheduler = LambdaLR(optimizer, schedule)

    random.seed(SEED + rank)
    np.random.seed(SEED + rank)
    torch.manual_seed(SEED + rank)
    step, epoch, start = 0, 0, time.time()
    while step < training["max_steps"]:
        sampler.set_epoch(epoch)
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            losses = model(batch, global_step=step)
            losses["loss"].backward()
            for param in synced:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world)
            grad_norm = model.clip_grad_norm_(training["gradient_clip_val"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if rank == 0 and (step % 10 == 0 or step == 1):
                values = " ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
                rate = (time.time() - start) / step
                print(
                    f"step {step} {values} grad_norm={grad_norm.item():.3f} lr={scheduler.get_last_lr()[0]:.2e} {rate:.2f}s/it",
                    flush=True,
                )
            if step % training["save_steps"] == 0 or step == training["max_steps"]:
                save_checkpoint(model, output / f"step_{step:06d}.safetensors")
            if step >= training["max_steps"]:
                break
        epoch += 1
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
