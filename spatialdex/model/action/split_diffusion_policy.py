import torch
import torch.distributed as dist
from torch import nn

from spatialdex.model.action.diffusion import DDIMSampler, TrainingDiffusion
from spatialdex.model.action.dit import DiT

HORIZON = 16
ACTION_DIM = 192
WRIST_INDICES = list(range(0, 6)) + list(range(51, 57))
FINGER_INDICES = list(range(6, 51)) + list(range(57, 102))
LOSS_COMPONENTS = (
    ("left_trans", 0, 3, 1.6),
    ("left_rot", 3, 6, 1.6),
    ("left_hand", 6, 51, None),
    ("right_trans", 51, 54, 1.6),
    ("right_rot", 54, 57, 1.6),
    ("right_hand", 57, 102, None),
)


def group_loss_components(indices, hand_loss_weight):
    components = []
    for _, start, end, weight in LOSS_COMPONENTS:
        local = [i for i, full in enumerate(indices) if start <= full < end]
        if local:
            components.append((local, hand_loss_weight if weight is None else weight))
    return components


class DiffusionHead(nn.Module):
    def __init__(self, action_dim, token_size, state_dim, cross_attention, causal, loss_components=()):
        super().__init__()
        self.action_dim = action_dim
        self.net = DiT(
            action_dim=action_dim,
            token_size=token_size,
            state_dim=state_dim,
            horizon=HORIZON,
            cross_attention=cross_attention,
            causal=causal,
        )
        self.loss_components = loss_components
        self.diffusion = TrainingDiffusion()
        self.samplers = {}

    def sampler(self, num_ddim_steps):
        if num_ddim_steps not in self.samplers:
            self.samplers[num_ddim_steps] = DDIMSampler(num_ddim_steps)
        return self.samplers[num_ddim_steps]

    def sample(
        self,
        features,
        cfg_scale,
        state,
        state_mask,
        num_ddim_steps,
        action_mask,
        condition_mask=None,
        wrist=None,
        wrist_mask=None,
    ):
        batch = features.shape[0]
        device = features.device
        if condition_mask is not None:
            condition_mask = condition_mask.to(device=device, dtype=torch.bool)
        noise = torch.randn(batch, HORIZON, self.action_dim, device=device)
        action_mask = action_mask.to(device)
        if cfg_scale <= 1.0:
            return self.sampler(num_ddim_steps).sample(
                self.net, noise, z=features, x_mask=action_mask, state=state, state_mask=state_mask
            )
        kwargs = dict(z=features, x_mask=action_mask, cfg_scale=cfg_scale, state=state, state_mask=state_mask)
        if condition_mask is not None:
            kwargs["condition_mask"] = torch.cat([condition_mask, condition_mask], dim=0)
        if wrist is not None:
            wrist = wrist.to(device=device)
            wrist_mask = wrist_mask.to(device=device, dtype=torch.bool)
            kwargs["wrist"] = torch.cat((wrist, wrist), dim=0)
            kwargs["wrist_mask"] = torch.cat((wrist_mask, torch.zeros_like(wrist_mask)), dim=0)
        samples = self.sampler(num_ddim_steps).sample(self.net.forward_with_cfg, torch.cat([noise, noise], 0), **kwargs)
        return samples.chunk(2, dim=0)[0]

    def loss(self, x, z, x_mask, state, state_mask):
        noise = torch.randn_like(x)
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device=x.device)
        x_t = self.diffusion.q_sample(x, timestep, noise) * x_mask
        noise_pred = self.net(torch.cat([x_t, x_mask], dim=2), timestep, z, state, state_mask)
        square = (noise_pred - noise) ** 2 * x_mask
        numerator, count = 0, 0
        for local, weight in self.loss_components:
            index = torch.as_tensor(local, dtype=torch.long, device=x.device)
            mask = x_mask.index_select(-1, index)
            value = square.index_select(-1, index).sum() / (mask.sum() + 1e-08) * weight
            n = mask.any(dim=-1).sum()
            numerator = numerator + value * n
            count = count + n
        return numerator / (count + 1e-08), numerator, count


class SplitDiffusionPolicy(nn.Module):
    def __init__(
        self,
        token_size,
        state_dim,
        split_config,
        in_channels=ACTION_DIM,
        cross_attention=True,
        self_attention_mode="bidirectional",
        hand_loss_weight=0.4,
    ):
        super().__init__()
        self.register_buffer("global_indices", torch.tensor(WRIST_INDICES, dtype=torch.long), persistent=False)
        self.register_buffer("finger_indices", torch.tensor(FINGER_INDICES, dtype=torch.long), persistent=False)
        finger_cfg = split_config.get("finger", {})
        causal = self_attention_mode == "causal"
        self.global_head = DiffusionHead(
            len(WRIST_INDICES),
            token_size,
            state_dim,
            cross_attention,
            causal,
            group_loss_components(WRIST_INDICES, hand_loss_weight),
        )
        self.finger_head = DiffusionHead(
            len(FINGER_INDICES),
            token_size,
            state_dim,
            cross_attention,
            causal,
            group_loss_components(FINGER_INDICES, hand_loss_weight),
        )
        condition_dim = HORIZON * len(WRIST_INDICES) * 2
        hidden_dim = int(finger_cfg.get("global_condition_hidden_size", token_size))
        self.global_condition_projector = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_size),
        )
        schedule = split_config.get("scheduled_sampling", {})
        self.schedule = (
            int(schedule.get("start_step", 0)),
            int(schedule.get("end_step", 10000)),
            float(schedule.get("start_prob", 0.0)),
            float(schedule.get("end_prob", 1.0)),
        )
        self.sampling_ddim_steps = int(schedule.get("num_ddim_steps", 4))
        self.sampling_cfg_scale = float(schedule.get("cfg_scale", 1.0))

    def condition_on_wrist(self, features, wrist_actions, wrist_mask):
        wrist_actions = wrist_actions[:, -HORIZON:, :]
        wrist_mask = (
            wrist_mask[:, -HORIZON:, :].to(device=wrist_actions.device, dtype=torch.bool).to(wrist_actions.dtype)
        )
        condition = torch.cat([wrist_actions * wrist_mask, wrist_mask], dim=-1).reshape(wrist_actions.shape[0], -1)
        projector_dtype = next(self.global_condition_projector.parameters()).dtype
        token = self.global_condition_projector(condition.to(device=features.device, dtype=projector_dtype))
        return torch.cat([features, token.to(dtype=features.dtype).unsqueeze(1)], dim=1)

    @staticmethod
    def finger_condition_mask(features, condition_mask):
        if condition_mask is None:
            return None
        condition_mask = condition_mask.to(device=features.device, dtype=torch.bool)
        ones = torch.ones(features.shape[0], 1, dtype=torch.bool, device=features.device)
        return torch.cat([condition_mask, ones], dim=1)

    def sample(
        self,
        features,
        cfg_scale,
        state,
        state_mask,
        num_ddim_steps,
        action_mask,
        condition_mask=None,
        wrist=None,
        wrist_mask=None,
    ):
        global_mask = action_mask.index_select(-1, self.global_indices.to(action_mask.device))
        finger_mask = action_mask.index_select(-1, self.finger_indices.to(action_mask.device))
        wrist_actions = self.global_head.sample(
            features, cfg_scale, state, state_mask, num_ddim_steps, global_mask, condition_mask, wrist, wrist_mask
        )
        finger_features = self.condition_on_wrist(features, wrist_actions, global_mask)
        finger_actions = self.finger_head.sample(
            finger_features,
            cfg_scale,
            state,
            state_mask,
            num_ddim_steps,
            finger_mask,
            self.finger_condition_mask(features, condition_mask),
            wrist,
            wrist_mask,
        )
        out = torch.zeros(
            (wrist_actions.shape[0], wrist_actions.shape[1], ACTION_DIM),
            dtype=wrist_actions.dtype,
            device=wrist_actions.device,
        )
        out.index_copy_(-1, self.global_indices.to(out.device), wrist_actions)
        out.index_copy_(-1, self.finger_indices.to(out.device), finger_actions)
        return out

    def sampling_probability(self, step):
        start_step, end_step, start_prob, end_prob = self.schedule
        if step <= start_step:
            return start_prob
        if step >= end_step:
            return end_prob
        ratio = float(step - start_step) / float(end_step - start_step)
        return max(0.0, min(1.0, start_prob + ratio * (end_prob - start_prob)))

    def sample_wrist_condition(self, z, wrist_mask, wrist_target, state, state_mask):
        sampled = self.global_head.sample(
            z, self.sampling_cfg_scale, state, state_mask, self.sampling_ddim_steps, wrist_mask
        )
        return sampled.to(device=wrist_target.device, dtype=wrist_target.dtype)

    def loss(self, x, z, x_mask, state, state_mask, global_step):
        global_x = x.index_select(-1, self.global_indices.to(x.device))
        finger_x = x.index_select(-1, self.finger_indices.to(x.device))
        global_mask = x_mask.index_select(-1, self.global_indices.to(x.device))
        finger_mask = x_mask.index_select(-1, self.finger_indices.to(x.device))
        probability = self.sampling_probability(global_step)
        global_loss, global_num, global_count = self.global_head.loss(global_x, z, global_mask, state, state_mask)
        if probability <= 0.0:
            condition = global_x[:, -HORIZON:]
        elif probability < 1.0:
            selected = torch.rand(global_x.shape[0], device=global_x.device) < probability
            condition = global_x[:, -HORIZON:].clone()
            local_any = bool(selected.any().item())
            any_selected = local_any
            if dist.is_available() and dist.is_initialized():
                flag = torch.tensor(int(local_any), dtype=torch.int32, device=global_x.device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                any_selected = bool(flag.item())
            if any_selected:
                index = torch.nonzero(selected, as_tuple=False).flatten()
                sample_index = index if local_any else torch.zeros(1, dtype=torch.long, device=global_x.device)
                with torch.no_grad():
                    sampled = self.sample_wrist_condition(
                        z.index_select(0, sample_index),
                        global_mask.index_select(0, sample_index),
                        global_x.index_select(0, sample_index),
                        state.index_select(0, sample_index),
                        state_mask.index_select(0, sample_index),
                    )
                if local_any:
                    condition.index_copy_(0, index, sampled)
        else:
            with torch.no_grad():
                condition = self.sample_wrist_condition(z, global_mask, global_x, state, state_mask)
        finger_z = self.condition_on_wrist(z, condition, global_mask[:, -HORIZON:])
        finger_loss, finger_num, finger_count = self.finger_head.loss(
            finger_x, finger_z, finger_mask, state, state_mask
        )
        total = (global_num + finger_num) / (global_count + finger_count + 1e-08)
        return dict(loss=total, wrist_loss=global_loss.detach(), finger_loss=finger_loss.detach())
