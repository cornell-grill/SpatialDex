import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from spatialdex.eval.dexjoco import sim
from spatialdex.eval.dexjoco.policy import SpatialDexPolicy, load_task_config

TASKS = ("microwave", "assembly", "fold_glasses", "water_plant", "pick_bucket", "hammer_nail")
REPO_ROOT = Path(__file__).resolve().parents[3]


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2))


def run_episode(env, raw_env, policy, task, seed, args, out_dir, ep, rgb_cache, video_stride):
    dual_arm = policy.dual_arm
    sim.reseed_env(raw_env, seed)
    raw_env.hz = 1e9
    cam_id = sim.resolve_camera_id(raw_env, task["camera"])
    obs, _ = env.reset()
    sim.reset_task_state(raw_env)
    if rgb_cache is not None:
        rgb_cache.reset()
    sim.set_camera_fovy(raw_env, cam_id, task["fovy"])
    if rgb_cache is None and task["camera"] in obs:
        obs = dict(obs)
        obs[task["camera"]] = sim.render_rgb(raw_env, cam_id)
    initial_rgb = rgb_cache.get() if rgb_cache is not None else sim.get_rgb(raw_env, obs, task["camera"], cam_id)
    H, W = initial_rgb.shape[:2]
    K = sim.intrinsics_from_fovy(H, W, float(raw_env.model.cam_fovy[cam_id]))
    sim_dt = float(raw_env.control_dt)
    train_dt = 1.0 / policy.train_hz
    replan_interval = max(1, int(round(args.replan_ratio * 16)))
    chunk = None
    num_replans = 0
    since_predict = 0.0
    last_info = {}
    failure = None
    writer = None
    video_path = out_dir / f"episode_{ep:02d}_seed{seed}.mp4"
    if not args.no_video:
        writer = sim.open_video_writer(video_path, round(1.0 / sim_dt / video_stride))
        writer.append_data(initial_rgb)
    steps = 0
    try:
        for t in range(args.max_steps):
            if chunk is None or since_predict / train_dt >= replan_interval:
                head = rgb_cache.get() if rgb_cache is not None else sim.get_rgb(raw_env, obs, task["camera"], cam_id)
                views = sim.policy_views(raw_env, obs, head, dual_arm=dual_arm)
                depth = sim.render_depth(raw_env, cam_id)
                cam_pose = sim.camera_to_robot_base(raw_env, cam_id)
                command = sim.read_command44(raw_env, dual_arm)
                chunk = np.asarray(
                    policy.predict_chunk(
                        views,
                        depth,
                        K,
                        obs["state"],
                        task["instruction"],
                        camera_to_robot_base=cam_pose,
                        executed_command44=command,
                    ),
                    dtype=np.float32,
                ).copy()
                if num_replans == 0:
                    chunk[0] = chunk[1]
                num_replans += 1
                since_predict = 0.0
            f_idx = since_predict / train_dt
            lo = int(np.floor(f_idx))
            frac = float(f_idx - lo)
            n = chunk.shape[0]
            hi = max(0, min(n - 1, lo + 1))
            lo = max(0, min(n - 1, lo))
            if dual_arm:
                env_action = sim.bimanual_env_action(sim.interpolate44(chunk[lo], chunk[hi], frac))
            else:
                env_action = sim.compact22_to_pose23(sim.interpolate22(chunk[lo], chunk[hi], frac))
            obs, _, terminated, truncated, info = env.step(env_action)
            steps = t + 1
            if rgb_cache is not None:
                rgb_cache.reset()
            last_info = dict(info)
            since_predict += sim_dt
            if writer is not None and steps % video_stride == 0:
                writer.append_data(
                    rgb_cache.get() if rgb_cache is not None else sim.get_rgb(raw_env, obs, task["camera"], cam_id)
                )
            stop, failure = sim.step_decision(policy.env_name, raw_env, last_info, terminated, truncated)
            if stop:
                break
    finally:
        if writer is not None:
            writer.close()
    success = bool(last_info.get("succeed", False))
    if not success and failure is None:
        failure = "max_steps"
    return dict(
        episode=ep,
        seed=seed,
        success=success,
        failure_reason=None if success else failure,
        simulation_steps=steps,
        num_replans=num_replans,
    )


def main():
    p = argparse.ArgumentParser(description="Closed-loop DexJoCo evaluation of a SpatialDex checkpoint.")
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--config-dir", type=Path, default=REPO_ROOT / "configs" / "dexjoco")
    p.add_argument("--dexjoco-root", type=Path, default=None)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--replan-ratio", type=float, default=1.0)
    p.add_argument("--ddim-steps", type=int, default=10)
    p.add_argument("--cfg-scale", type=float, default=5.0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.dexjoco_root is not None:
        sys.path.insert(0, str(args.dexjoco_root / "dexjoco"))
    config = load_task_config(args.task, args.config_dir)
    task = config["dexjoco"]
    checkpoint = args.checkpoint or REPO_ROOT / "weights" / "dexjoco" / f"{args.task}.safetensors"
    policy = SpatialDexPolicy(config, checkpoint, ddim_steps=args.ddim_steps, cfg_scale=args.cfg_scale)
    args.output.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed + i for i in range(args.episodes)]
    render_fps = task.get("render_fps")
    env = sim.make_env(policy.env_name, seeds[0], image_obs=render_fps is None)
    raw_env = env.unwrapped
    rgb_cache = None
    video_stride = 1
    if render_fps is not None:
        rgb_cache = sim.RgbCache(raw_env, sim.resolve_camera_id(raw_env, task["camera"]))
        video_stride = sim.render_stride(float(raw_env.control_dt), render_fps)

    results = []
    for ep, seed in enumerate(seeds):
        result = run_episode(env, raw_env, policy, task, seed, args, args.output, ep, rgb_cache, video_stride)
        results.append(result)
        write_json(args.output / f"episode_{ep:02d}_seed{seed}.json", result)
        done = sum(r["success"] for r in results)
        print(
            f"[{ep + 1}/{len(seeds)}] seed={seed} {'SUCCESS' if result['success'] else 'FAIL'} ({done}/{ep + 1})",
            flush=True,
        )
    successes = sum(r["success"] for r in results)
    summary = dict(
        task=args.task,
        episodes=len(results),
        successes=successes,
        success_rate=successes / len(results),
        results=results,
    )
    write_json(args.output / "results.json", summary)
    print(json.dumps({k: summary[k] for k in ("task", "episodes", "successes", "success_rate")}), flush=True)


if __name__ == "__main__":
    main()
