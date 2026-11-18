import random

import imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

CAMERA_ATTRIBUTES = {
    "ego": "_front_camera_id",
    "front": "_front_camera_id",
    "wrist": "_wrist_camera_id",
    "wrist_left": "_wrist_left_camera_id",
    "wrist_right": "_wrist_right_camera_id",
}
ENV_STEP_CAPS = {
    "bimanual_microwave_cook": 1100,
    "bimanual_assembly": 1500,
    "fold_glasses": 1200,
    "water_plant": 1000,
    "pick_bucket": 1200,
    "hammer_nail": 1000,
}


def make_env(env_name, seed, *, image_obs=True):
    from dexjoco.tasks import CONFIG_MAPPING

    cfg = CONFIG_MAPPING[env_name]()
    kwargs = dict(image_obs=image_obs) if env_name.startswith("bimanual_") else {}
    return cfg.get_environment(
        policy_mode=True, render_mode="rgb_array", randomize=False, seed=seed, randomize_dynamics=False, **kwargs
    )


def reseed_env(raw_env, seed):
    random.seed(seed)
    np.random.seed(seed)
    raw_env._random = np.random.RandomState(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_task_state(raw_env):
    raw_env._success_started = False
    raw_env._success_counter = 0


def water_trigger_failed(raw_env):
    if not raw_env._trigger_pulled:
        return False
    site = raw_env._data.site_xpos[raw_env._model.site("ref_point").id]
    plant = raw_env._data.body("plant").xpos
    dx, dy, dz = site - plant
    return bool(dx * dx + dy * dy > 0.2**2 or not -0.2 <= dz <= 0.2)


def step_decision(env_name, raw_env, info, terminated, truncated):
    if truncated:
        return True, "truncated"
    if info.get("succeed", False):
        return True, None
    if env_name == "water_plant" and terminated and water_trigger_failed(raw_env):
        return True, "trigger_outside_plant"
    if not terminated:
        return False, None
    if raw_env.env_step >= ENV_STEP_CAPS[env_name]:
        return False, None
    return True, "task_terminated"


def render(raw_env, render_mode, camera_id):
    renderer = getattr(raw_env, "_viewer", None) or raw_env._mj_viewer
    viewer = renderer._get_viewer(render_mode="rgb_array")
    viewer.make_context_current()
    return viewer.render(render_mode=render_mode, camera_id=camera_id)


def linearize_depth(zbuf, znear, zfar):
    z = np.asarray(zbuf, dtype=np.float32)
    denom = zfar - z * (zfar - znear)
    depth = np.full_like(z, np.inf, dtype=np.float32)
    mask = denom > 0
    depth[mask] = (znear * zfar / denom)[mask]
    return depth


def render_depth(raw_env, camera_id):
    extent = float(raw_env._model.stat.extent)
    znear = float(raw_env._model.vis.map.znear) * extent
    zfar = float(raw_env._model.vis.map.zfar) * extent
    return linearize_depth(render(raw_env, "depth_array", camera_id), znear, zfar)


def render_rgb(raw_env, camera_id):
    return render(raw_env, "rgb_array", camera_id)


class RgbCache:
    def __init__(self, raw_env, camera_id):
        self.raw_env = raw_env
        self.camera_id = camera_id
        self.frame = None

    def reset(self):
        self.frame = None

    def get(self):
        if self.frame is None:
            self.frame = render_rgb(self.raw_env, self.camera_id)
        return self.frame


def render_stride(control_dt, render_fps):
    return int(round(1.0 / control_dt / render_fps))


def resolve_camera_id(raw_env, name):
    return int(getattr(raw_env, CAMERA_ATTRIBUTES[name]))


def get_rgb(raw_env, obs, name, cam_id):
    return np.asarray(obs[name]) if name in obs else render_rgb(raw_env, cam_id)


def policy_views(raw_env, obs, head, *, dual_arm):
    def wrist(name):
        return np.ascontiguousarray(get_rgb(raw_env, obs, name, resolve_camera_id(raw_env, name)))

    if dual_arm:
        return [head, wrist("wrist_left"), wrist("wrist_right")]
    return [head, np.zeros_like(head), wrist("wrist")]


def open_video_writer(path, fps):
    return imageio.get_writer(path, format="FFMPEG", mode="I", fps=int(fps), codec="libx264", macro_block_size=16)


def set_camera_fovy(raw_env, cam_id, fovy):
    raw_env.model.cam_fovy[cam_id] = float(fovy)


def intrinsics_from_fovy(height, width, fovy_deg):
    fy = height / 2.0 / np.tan(np.deg2rad(fovy_deg) / 2.0)
    return np.array([[fy, 0.0, (width - 1) / 2.0], [0.0, fy, (height - 1) / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def camera_to_robot_base(raw_env, camera_id):
    rotation = np.asarray(raw_env.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(raw_env.data.cam_xpos[camera_id], dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def read_command44(raw_env, dual_arm):
    ids = np.asarray(raw_env._allegro_ctrl_ids)
    fingers = np.asarray(raw_env._data.ctrl[ids], dtype=np.float64)
    poses = raw_env.get_end_effector_pose_matrix()
    if not dual_arm:
        poses = [poses]
    result = np.zeros(44, dtype=np.float32)
    for i, j in enumerate((1, 0) if dual_arm else (1,)):
        pose = np.asarray(poses[i], dtype=np.float64)
        result[j * 22 : j * 22 + 3] = pose[:3, 3]
        result[j * 22 + 3 : j * 22 + 6] = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
        result[j * 22 + 6 : (j + 1) * 22] = fingers[i * 16 : (i + 1) * 16]
    return result


def compact22_to_pose23(compact22):
    compact22 = np.asarray(compact22, dtype=np.float32)
    quat_wxyz = Rotation.from_rotvec(compact22[3:6].astype(np.float64)).as_quat(scalar_first=True)
    return np.concatenate([compact22[:3], quat_wxyz.astype(np.float32), compact22[6:22]], axis=0).astype(np.float32)


def bimanual_env_action(action44):
    left23 = compact22_to_pose23(action44[:22])
    right23 = compact22_to_pose23(action44[22:44])
    return np.concatenate([right23[:7], left23[:7], right23[7:23], left23[7:23]], axis=0).astype(np.float32)


def interpolate22(action0, action1, fraction):
    action0 = np.asarray(action0, dtype=np.float32)
    action1 = np.asarray(action1, dtype=np.float32)
    fraction = float(fraction)
    if fraction == 0.0 or np.array_equal(action0, action1):
        return action0.copy()
    if fraction == 1.0:
        return action1.copy()
    out = ((1.0 - fraction) * action0 + fraction * action1).astype(np.float32)
    rotations = Rotation.from_rotvec(np.stack([action0[3:6], action1[3:6]]).astype(np.float64))
    out[3:6] = Slerp([0.0, 1.0], rotations)([fraction]).as_rotvec()[0].astype(np.float32)
    return out


def interpolate44(action0, action1, fraction):
    return np.concatenate(
        [interpolate22(action0[:22], action1[:22], fraction), interpolate22(action0[22:44], action1[22:44], fraction)]
    ).astype(np.float32)
