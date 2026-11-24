import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
import torch

from spatialdex.eval.wuji.policy import WujiPolicy
from spatialdex.eval.wuji.sim import Scene, layout_bodies, load_scene_model

TASKS = ("relocate_ball", "pass_toy", "sweep_trash")
REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTEPS = 8
CHUNK = 8
NOMINAL_SCENES = 50
NOMINAL_JITTER = 0.02  # metres


def load_config(task, config_dir):
    path = Path(config_dir) / f"{task}.json"
    config = json.loads(path.read_text())
    config["statistics_path"] = str(path.parent / config["statistics_path"])
    vlm = config["vlm"]["pretrained_model_name_or_path"]
    if not Path(vlm).is_absolute():
        config["vlm"]["pretrained_model_name_or_path"] = str(REPO_ROOT / vlm)
    return config


def nominal_scenes(scenes):
    """The most typical recorded starting scene of each hand role: among the scenes whose object lies the
    usual way up, the one whose object and target positions are closest to all the others."""
    models, roles = {}, {}
    for path in scenes:
        xml = path.with_suffix(".xml").read_bytes()
        if xml not in models:
            models[xml] = load_scene_model(path.with_suffix(".xml"))
        data = mujoco.MjData(models[xml])
        with h5py.File(path, "r") as f:
            data.qpos[:] = f["initial_qpos"][:]
            if "initial_mocap_pos" in f:
                data.mocap_pos[:] = f["initial_mocap_pos"][:].reshape(data.mocap_pos.shape)
            role = str(f.attrs["instruction"])
        mujoco.mj_forward(models[xml], data)
        obj, target = layout_bodies(models[xml])
        _, x, y, _ = data.xquat[obj]
        upright = bool(1.0 - 2.0 * (x * x + y * y) > 0.94)
        roles.setdefault(role, []).append((path, upright, np.concatenate([data.xpos[obj][:2], data.xpos[target][:2]])))
    picks = []
    for _, group in sorted(roles.items()):
        usual = sum(upright for _, upright, _ in group) * 2 >= len(group)
        pool = [(path, layout) for path, upright, layout in group if upright == usual]
        layouts = np.stack([layout for _, layout in pool])
        picks.append(pool[int(np.linalg.norm(layouts[:, None] - layouts[None], axis=-1).sum(1).argmin())][0])
    return picks


def open_video(path, fps=30):
    import imageio

    return imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1)


def run_episode(policy, settings, scene_path, args, video_path=None, rng=None):
    scene = Scene(scene_path.with_suffix(".xml"), scene_path, settings["recorder_ik"])
    if rng is not None:
        scene.jitter(rng, NOMINAL_JITTER)
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
    parser.add_argument("--layout", choices=("nominal", "recorded"), help="defaults to the task config")
    parser.add_argument("--ddim_steps", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.task, args.config_dir)
    statistics = Path(args.checkpoint).with_suffix(".json")
    if statistics.exists():
        config["statistics_path"] = str(statistics)
    policy = WujiPolicy(config, args.checkpoint, ddim_steps=args.ddim_steps)
    scenes = sorted((Path(args.data_root) / args.task / "eval").glob("episode_*.hdf5"))
    scenes = [s for s in scenes if s.with_suffix(".xml").exists()]
    layout = "recorded" if args.episodes else args.layout or config["wuji_tianji"].get("layout", "recorded")
    if args.episodes:
        scenes = [s for s in scenes if int(s.stem.split("_")[1]) in set(args.episodes)]
    if layout == "nominal":
        # A typical starting scene per hand role, with the object and the bin shifted a little each time.
        starts = nominal_scenes(scenes)
        trials = [
            (f"scene_{i:03d}", starts[i % len(starts)], np.random.default_rng(i)) for i in range(NOMINAL_SCENES)
        ]
    else:
        starts = scenes
        trials = [(s.stem, s, None) for s in scenes]
    print(f"layout: {layout}, {len(trials)} scenes from {len(starts)} recorded starting scenes")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(layout=layout, ddim_steps=args.ddim_steps)
    if layout == "nominal":
        summary["starting_scenes"] = [s.stem for s in starts]
    results = {}
    for name, scene_path, rng in trials:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        video = None if args.no_video else out / f"{name}.mp4"
        results[name] = run_episode(policy, config["wuji_tianji"], scene_path, args, video, rng)
        rate = sum(results.values()) / len(results)
        print(f"{name}: {'success' if results[name] else 'failure'} ({rate:.1%} so far)")
        (out / "results.json").write_text(json.dumps(dict(success_rate=rate, **summary, episodes=results), indent=2))


if __name__ == "__main__":
    main()
