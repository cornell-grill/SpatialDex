import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from spatialdex.data.camera_anchor import decode
from spatialdex.data.mano_mapping import (
    hand_supervised_dim_indices,
    pack_allegro_pose_into_mano_layout,
    unpack_mano_layout_to_allegro_pose,
)
from spatialdex.model import load_model

CV_FROM_GL = np.diag([1.0, -1.0, -1.0, 1.0])
HORIZON = 16


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_task_config(task, config_dir):
    path = Path(config_dir) / f"{task}.json"
    config = json.loads(path.read_text())
    config["statistics_path"] = str(path.parent / config["statistics_path"])
    vlm_path = Path(config["vlm"]["pretrained_model_name_or_path"])
    if not vlm_path.is_absolute():
        config["vlm"]["pretrained_model_name_or_path"] = str(REPO_ROOT / vlm_path)
    return config


class Normalizer:
    def __init__(self, path):
        stats = json.loads(Path(path).read_text())
        both = lambda name, key: np.concatenate([np.array(stats[f"{name}_{side}"][key]) for side in ("left", "right")])
        self.state_mean, self.state_std = both("state", "mean"), both("state", "std")
        self.action_mean, self.action_std = both("action", "mean"), both("action", "std")


def pose23_to_compact22(pose23):
    pose23 = np.asarray(pose23, dtype=np.float32)
    quat_wxyz = pose23[3:7].astype(np.float64)
    quat_xyzw = np.concatenate([quat_wxyz[1:4], quat_wxyz[:1]])
    rotvec = Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
    return np.concatenate([pose23[:3], rotvec, pose23[7:23]], axis=0).astype(np.float32)


def single_arm_state_to_compact44(state):
    out = np.zeros(44, dtype=np.float32)
    out[22:44] = pose23_to_compact22(np.asarray(state, dtype=np.float32)[:23])
    return out


def bimanual_state_to_compact44(state):
    state = np.asarray(state, dtype=np.float32)
    right23 = np.concatenate([state[0:7], state[14:30]])
    left23 = np.concatenate([state[7:14], state[30:46]])
    return np.concatenate([pose23_to_compact22(left23), pose23_to_compact22(right23)]).astype(np.float32)


def resize_pixel_intrinsics(intrinsics, source_size, target_size):
    resized = np.array(intrinsics)
    resized[0, :] *= target_size[1] / source_size[1]
    resized[1, :] *= target_size[0] / source_size[0]
    return resized


def resize_views(views, image_height):
    return [cv2.resize(view, (image_height, image_height), interpolation=cv2.INTER_AREA) for view in views]


def crop_head(rgb, depth, intrinsics, box):
    x0, y0, x1, y1 = box
    matrix = np.asarray(intrinsics).copy()
    matrix[0, 2] -= x0
    matrix[1, 2] -= y0
    return rgb[y0:y1, x0:x1], depth[y0:y1, x0:x1], matrix


def canonical_crop_rgbd(views, depth, K):
    head_shape = views[0].shape[:2]
    if head_shape != (512, 512):
        views = resize_views(views, 512)
        depth = cv2.resize(np.asarray(depth), (512, 512), interpolation=cv2.INTER_NEAREST)
        K = resize_pixel_intrinsics(K, head_shape, (512, 512))
    elif any(view.shape[:2] != (512, 512) for view in views):
        views = resize_views(views, 512)
    return views, depth, K


def resize_depth_to_vit(depth, K, vit_size, *, training_height=None, max_valid_m=None, device="cuda"):
    if training_height is not None:
        source_shape = depth.shape
        target_width = int(round(source_shape[1] * training_height / source_shape[0]))
        depth = np.nan_to_num(np.where(depth > max_valid_m, 0.0, depth), nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )
        depth = cv2.resize(depth, (target_width, training_height), interpolation=cv2.INTER_NEAREST)
        K = resize_pixel_intrinsics(K, source_shape, depth.shape).astype(np.float32)
    depth_t = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)
    depth_t = torch.nn.functional.interpolate(depth_t, size=(vit_size, vit_size), mode="nearest").to(device)
    K_scaled = resize_pixel_intrinsics(K, depth.shape, (vit_size, vit_size))
    return depth_t, torch.from_numpy(K_scaled.astype(np.float32)).unsqueeze(0).to(device)


class SpatialDexPolicy:
    def __init__(self, config, checkpoint, *, ddim_steps=10, cfg_scale=5.0, device="cuda"):
        self.config = config
        task = config["dexjoco"]
        self.env_name = task["env_name"]
        self.dual_arm = self.env_name.startswith("bimanual_")
        self.image_height = int(task["image_height"])
        self.head_crop_box = task.get("head_crop_box")
        self.depth_grid = (int(task["depth_image_height"]), float(task["depth_max_valid_m"]))
        self.stride = int(task["stride"])
        self.train_hz = float(task["train_hz"])
        self.device = device
        self.normalizer = Normalizer(config["statistics_path"])
        self.active = np.array([self.dual_arm, True])
        self.mounts = np.repeat(np.eye(4)[None], 2, axis=0)
        self.mask = np.zeros(192, bool)
        for j, side in enumerate(("left", "right")):
            if self.active[j]:
                self.mounts[j] = np.array(task["ee_to_mano"][side])
                self.mask[j * 51 : j * 51 + 6] = True
                self.mask[j * 51 + 6 + hand_supervised_dim_indices(side)] = True
        self.model = load_model(config, checkpoint).to(device).eval()
        self.model_dtype = next(self.model.act_model.parameters()).dtype
        self.ddim_steps = int(ddim_steps)
        self.cfg_scale = float(cfg_scale)
        self.last_prediction = None

    def encode(self, compact44, camera_to_world_gl):
        compact = np.asarray(compact44, dtype=np.float64)
        camera = camera_to_world_gl @ CV_FROM_GL
        world2cam = np.linalg.inv(camera)
        packed = pack_allegro_pose_into_mano_layout(compact).reshape(2, 51)
        wrists = np.repeat(np.eye(4)[None], 2, axis=0)
        metric = np.zeros((2, 51), dtype=np.float32)
        for j in range(2):
            if not self.active[j]:
                continue
            ee = np.eye(4)
            ee[:3, :3] = Rotation.from_rotvec(compact[j * 22 + 3 : j * 22 + 6]).as_matrix()
            ee[:3, 3] = compact[j * 22 : j * 22 + 3]
            wrists[j] = world2cam @ ee @ self.mounts[j]
            metric[j, :3] = wrists[j, :3, 3]
            metric[j, 3:6] = Rotation.from_matrix(wrists[j, :3, :3]).as_euler("xyz")
            metric[j, 6:] = packed[j, 6:]
        state = torch.zeros(212, dtype=torch.float32)
        state_mask = torch.zeros(212, dtype=torch.bool)
        state_mask[:192] = torch.from_numpy(self.mask)
        for j in range(2):
            mean = torch.as_tensor(self.normalizer.state_mean[j * 61 : j * 61 + 51], dtype=torch.float32)
            std = torch.as_tensor(self.normalizer.state_std[j * 61 : j * 61 + 51], dtype=torch.float32)
            state[j * 51 : j * 51 + 51] = (
                (torch.from_numpy(metric[j]) - mean) / (std + 1e-7) * state_mask[j * 51 : j * 51 + 51]
            )
        for j in np.flatnonzero(self.active):
            rotvec = Rotation.from_matrix(wrists[j, :3, :3]).as_rotvec()
            mean = np.asarray(self.normalizer.state_mean)[j * 61 + 3 : j * 61 + 6]
            std = np.asarray(self.normalizer.state_std)[j * 61 + 3 : j * 61 + 6]
            state[j * 51 + 3 : j * 51 + 6] = torch.as_tensor((rotvec - mean) / (std + 1e-7), dtype=torch.float32)
        context = dict(camera=camera, wrists=wrists, slots=packed[:, 6:].copy())
        action_mask = torch.from_numpy(np.broadcast_to(self.mask, (HORIZON, 192)).copy())
        return state, state_mask, action_mask, context

    def decode(self, prediction, context):
        norm = np.asarray(prediction, dtype=np.float64)
        metric = norm[0, :, :102] * (np.asarray(self.normalizer.action_std) + 1e-7) + self.normalizer.action_mean
        anchor = np.zeros((2, 61), dtype=np.float64)
        for j in np.flatnonzero(self.active):
            anchor[j, :3] = context["wrists"][j, :3, 3]
            anchor[j, 3:6] = Rotation.from_matrix(context["wrists"][j, :3, :3]).as_rotvec()
            anchor[j, 6:51] = context["slots"][j]
        future, slots = decode(metric, anchor)
        absolute = np.zeros((HORIZON, 102), dtype=np.float64)
        for j in np.flatnonzero(self.active):
            flange = context["camera"] @ future[:, j] @ np.linalg.inv(self.mounts[j])
            absolute[:, j * 51 : j * 51 + 3] = flange[:, :3, 3]
            absolute[:, j * 51 + 3 : j * 51 + 6] = Rotation.from_matrix(flange[:, :3, :3]).as_rotvec()
            ids = hand_supervised_dim_indices(("left", "right")[j])
            absolute[:, j * 51 + 6 + ids] = slots[:, j, ids]
        return unpack_mano_layout_to_allegro_pose(absolute).astype(np.float32)

    def predict_chunk(self, rgb, depth, K, state, instruction, *, camera_to_robot_base, executed_command44):
        views = [np.asarray(view) for view in rgb]
        H, W = views[0].shape[:2]
        if self.head_crop_box is not None:
            views, depth, K = canonical_crop_rgbd(views, depth, K)
            views[0], depth, K = crop_head(views[0], depth, K, self.head_crop_box)
            H, W = views[0].shape[:2]
            images = views
        else:
            images = resize_views(views, self.image_height)
        fov = np.array([2 * np.arctan2(W / 2.0, K[0, 0]), 2 * np.arctan2(H / 2.0, K[1, 1])], dtype=np.float32)
        if self.dual_arm:
            state44 = bimanual_state_to_compact44(state)
        else:
            state44 = single_arm_state_to_compact44(state)
        current_state, current_state_mask, action_mask, context = self.encode(state44, camera_to_robot_base)
        vit_size = int(self.model.vision_tower.config.image_size)
        if self.head_crop_box is not None:
            depth = np.nan_to_num(
                np.where(depth > self.depth_grid[1], 0.0, depth), nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            depth_t, intr_t = resize_depth_to_vit(depth, K, vit_size, device=self.device)
        else:
            depth_t, intr_t = resize_depth_to_vit(
                depth,
                K,
                vit_size,
                training_height=self.depth_grid[0],
                max_valid_m=self.depth_grid[1],
                device=self.device,
            )
        norm_action = self.model.predict_action(
            images=images,
            instruction=instruction,
            current_state=current_state.unsqueeze(0).to(dtype=self.model_dtype),
            current_state_mask=current_state_mask.unsqueeze(0),
            action_mask=action_mask.unsqueeze(0),
            fov=torch.from_numpy(fov).unsqueeze(0).to(dtype=self.model_dtype),
            depth=depth_t,
            intrinsics=intr_t,
            num_ddim_steps=self.ddim_steps,
            cfg_scale=self.cfg_scale,
        )
        self.last_prediction = norm_action
        future44 = self.decode(norm_action, context)
        origin = np.asarray(executed_command44, dtype=np.float32)
        action44 = np.concatenate([origin[None], np.asarray(future44, dtype=np.float32)], axis=0)
        return action44 if self.dual_arm else action44[:, 22:44]
