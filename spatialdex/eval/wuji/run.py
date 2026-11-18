import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spatialdex.eval.wuji.policy import WujiPolicy
from spatialdex.eval.wuji.sim import Scene

TASKS = ("relocate_ball", "pass_toy", "sweep_trash")
REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTEPS = 8
CHUNK = 8


def load_config(task, config_dir):
    path = Path(config_dir) / f"{task}.json"
    config = json.loads(path.read_text())
    config["statistics_path"] = str(path.parent / config["statistics_path"])
    vlm = config["vlm"]["pretrained_model_name_or_path"]
    if not Path(vlm).is_absolute():
        config["vlm"]["pretrained_model_name_or_path"] = str(REPO_ROOT / vlm)
    return config


def open_video(path, fps=30):
    import imageio

    return imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1)


def run_episode(policy, settings, scene_path, args, video_path=None):
    scene = Scene(scene_path.with_suffix(".xml"), scene_path, settings["recorder_ik"])
    scene.open_active_hands()
    sim_dt = SUBSTEPS * float(scene.model.opt.timestep)
    frame_dt = settings["stride"] * (scene.recorder_frame_dt if settings["recorder_clock"] else 1 / 50)
    writer = open_video(video_path) if video_path is not None else None
    since_predict = float("inf")
    chunk = None
    for _ in range(args.steps):
        rgb = scene.render()
        K, camera_to_world = scene.camera_pose()
        if since_predict / frame_dt >= CHUNK:
            chunk = policy.predict(
                rgb,
                scene.render_depth(),
                K,
                camera_to_world,
                scene.hand_state("left"),
                scene.hand_state("right"),
                scene.instruction,
                scene.active,
            )
            since_predict = 0.0
        if writer is not None:
            writer.append_data(rgb)
        index = since_predict / frame_dt
        last = chunk[0].shape[0] - 1
        lo = int(np.floor(index))
        frac = float(index - lo)
        lo, hi = max(0, min(last, lo)), max(0, min(last, lo + 1))
        for arm, side in enumerate(("left", "right")):
            if scene.active[arm]:
                scene.apply(side, (1.0 - frac) * chunk[arm][lo] + frac * chunk[arm][hi])
        scene.step(SUBSTEPS)
        since_predict += sim_dt
    success = scene.success()
    scene.close()
    if writer is not None:
        writer.close()
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default=str(REPO_ROOT / "data" / "wuji_tianji"))
    parser.add_argument("--config_dir", default=str(REPO_ROOT / "configs" / "wuji_tianji"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, nargs="*")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_video", action="store_true")
    args = parser.parse_args()

    config = load_config(args.task, args.config_dir)
    statistics = Path(args.checkpoint).with_suffix(".json")
    if statistics.exists():
        config["statistics_path"] = str(statistics)
    policy = WujiPolicy(config, args.checkpoint)
    scenes = sorted((Path(args.data_root) / args.task / "eval").glob("episode_*.hdf5"))
    if args.episodes:
        scenes = [s for s in scenes if int(s.stem.split("_")[1]) in set(args.episodes)]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for scene_path in scenes:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        video = None if args.no_video else out / f"{scene_path.stem}.mp4"
        results[scene_path.stem] = run_episode(policy, config["wuji_tianji"], scene_path, args, video)
        rate = sum(results.values()) / len(results)
        print(f"{scene_path.stem}: {'success' if results[scene_path.stem] else 'failure'} ({rate:.1%} so far)")
        (out / "results.json").write_text(json.dumps(dict(success_rate=rate, episodes=results), indent=2))


if __name__ == "__main__":
    main()
