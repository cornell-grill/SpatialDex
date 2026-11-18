import bisect
import glob
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PaliGemmaProcessor

from spatialdex.data.wuji import camera_basis, hand_dims, pack_wuji_pose, transform_pose
from spatialdex.data.wuji_text import episode_instruction

HORIZON = 16
IMAGE_HEIGHT = 224
ACTION_DIM = 192
STATE_DIM = 212
CAMERA = "camera_head_cam"


def rotvec(quat_wxyz):
    xyzw = np.concatenate([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], axis=-1)
    return Rotation.from_quat(xyzw).as_rotvec().astype(np.float32)


def anchor_delta(action, anchor):
    target = action.astype(np.float64)
    state = anchor.astype(np.float64)
    result = target.copy()
    for start in (0, 26):
        rot = target[:, start + 3 : start + 6]
        ref = np.broadcast_to(state[start + 3 : start + 6], rot.shape)
        rot = Rotation.from_rotvec(np.ascontiguousarray(rot)).as_matrix()
        ref = Rotation.from_rotvec(np.ascontiguousarray(ref)).as_matrix()
        result[:, start : start + 3] = target[:, start : start + 3] - state[start : start + 3]
        result[:, start + 3 : start + 6] = Rotation.from_matrix(np.matmul(rot, np.swapaxes(ref, -1, -2))).as_rotvec()
    return result.astype(action.dtype)


def warp_depth(depth, K, camera_to_world, sub=2):
    src = np.ascontiguousarray(depth[::sub, ::sub])
    K_src = np.asarray(K, dtype=np.float64).copy()
    K_src[:2, :] /= sub
    K_dst = np.asarray(K, dtype=np.float64)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    height, width = depth.shape
    v, u = np.meshgrid(
        np.arange(src.shape[0], dtype=np.float32), np.arange(src.shape[1], dtype=np.float32), indexing="ij"
    )
    valid = src > 1e-06
    z = src[valid]
    x = (u[valid] - K_src[0, 2]) * z / K_src[0, 0]
    y = (v[valid] - K_src[1, 2]) * z / K_src[1, 1]
    points = np.stack([x, y, z], axis=1).astype(np.float64)
    T = np.linalg.inv(pose) @ pose
    points = points @ T[:3, :3].T + T[:3, 3]
    points = points[points[:, 2] > 1e-06]
    Z = points[:, 2]
    px = points[:, 0] * K_dst[0, 0] / Z + K_dst[0, 2]
    py = points[:, 1] * K_dst[1, 1] / Z + K_dst[1, 2]
    keep = np.isfinite(px) & np.isfinite(py) & np.isfinite(Z)
    keep &= (px >= -0.5) & (px < width - 0.5) & (py >= -0.5) & (py < height - 0.5)
    pi, qi = np.rint(px[keep]).astype(np.int64), np.rint(py[keep]).astype(np.int64)
    out = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(out, qi * width + pi, Z[keep].astype(np.float32))
    out[~np.isfinite(out)] = 0.0
    return out.reshape(height, width)


def clean_depth(depth, max_m):
    depth = np.where(depth > max_m, 0.0, depth)
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def scale_intrinsics(K, source, target):
    out = K.copy()
    out[0, :] *= target[1] / source[1]
    out[1, :] *= target[0] / source[0]
    return out


def resize_nearest(depth, size):
    return F.interpolate(torch.tensor(depth)[None, None], size=size, mode="nearest")[0]


class WujiDataset(Dataset):
    def __init__(self, data_dir, config, statistics_path):
        self.stride = int(config["wuji_tianji"]["stride"])
        self.depth_max = float(config["wuji_tianji"]["depth_max_valid_m"])
        self.processor = PaliGemmaProcessor.from_pretrained(config["vlm"]["pretrained_model_name_or_path"])
        self.pad_id = self.processor.tokenizer.pad_token_id
        stats = json.loads(Path(statistics_path).read_text())
        both = lambda name, key: np.concatenate([np.array(stats[f"{name}_{side}"][key]) for side in ("left", "right")])
        self.state_mean, self.state_std = both("state", "mean"), both("state", "std")
        self.action_mean, self.action_std = both("action", "mean"), both("action", "std")
        labels = json.loads((Path(data_dir) / "episode_labels.json").read_text())
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
        self.episodes, lengths = [], []
        for path in self.files:
            with h5py.File(path, "r") as f:
                self.episodes.append(episode_instruction(f, labels[Path(path).stem]))
                lengths.append(max(0, f["left_arm_qpos"].shape[0] - (HORIZON - 1) * self.stride - 1))
        self.offsets = np.cumsum([0] + lengths)

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, idx):
        ep = bisect.bisect_right(self.offsets, idx) - 1
        t0 = idx - int(self.offsets[ep])
        steps = np.arange(HORIZON) * self.stride + t0
        instruction, active = self.episodes[ep]
        with h5py.File(self.files[ep], "r") as f:
            ctrl = f["ctrl"][steps]
            action = np.concatenate(
                [
                    f["mocap_pos_left"][steps],
                    rotvec(f["mocap_quat_left"][steps]),
                    ctrl[:, 7:27],
                    f["mocap_pos_right"][steps],
                    rotvec(f["mocap_quat_right"][steps]),
                    ctrl[:, 34:54],
                ],
                axis=1,
            ).astype(np.float32)
            state = np.concatenate(
                [
                    f["left_ee_pos"][t0],
                    rotvec(f["left_ee_quat_wxyz"][t0]),
                    f["left_hand_qpos"][t0],
                    f["right_ee_pos"][t0],
                    rotvec(f["right_ee_quat_wxyz"][t0]),
                    f["right_hand_qpos"][t0],
                ]
            ).astype(np.float32)
            rgb = f["rgb_head_cam"][t0]
            depth = clean_depth(f["depth_head_cam"][t0], self.depth_max)
            future = clean_depth(f["depth_head_cam"][steps[-1]], self.depth_max)
            K = f[f"{CAMERA}/intrinsic_matrix"][()].astype(np.float32)
            extrinsic = f[f"{CAMERA}/extrinsic_matrix"][()]
        basis = camera_basis(extrinsic)
        action = transform_pose(action, basis, True, (True, True))
        state = transform_pose(state, basis, True, (True, True))
        action = anchor_delta(action, state)
        if (future > 1e-06).any():
            future = warp_depth(future, K, extrinsic.astype(np.float32))

        height, width = rgb.shape[:2]
        size = (IMAGE_HEIGHT, int(round(width * IMAGE_HEIGHT / height)))
        rgb = cv2.resize(rgb, size[::-1], interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, size[::-1], interpolation=cv2.INTER_NEAREST)
        future = cv2.resize(future, size[::-1], interpolation=cv2.INTER_NEAREST)
        K = scale_intrinsics(K, (height, width), size).astype(np.float32)
        fov = np.array(
            [2 * np.arctan(size[1] / (2 * K[0][0])), 2 * np.arctan(size[0] / (2 * K[1][1]))], dtype=np.float32
        )

        inputs = self.processor(text="<image>" + instruction, images=[Image.fromarray(rgb)], return_tensors="pt")
        pixel_values = inputs["pixel_values"][0]
        view = tuple(pixel_values.shape[-2:])

        action = pack_wuji_pose(((action - self.action_mean) / (self.action_std + 1e-07)).astype(np.float32))
        state = pack_wuji_pose(((state - self.state_mean) / (self.state_std + 1e-07)).astype(np.float32))
        side = torch.tensor(active).repeat_interleave(51)
        action_labels = torch.zeros(HORIZON, ACTION_DIM)
        action_labels[:, :102] = torch.tensor(action) * side.float()
        action_masks = torch.zeros(HORIZON, ACTION_DIM, dtype=torch.bool)
        for arm, name in enumerate(("left", "right")):
            action_masks[:, arm * 51 : arm * 51 + 6] = active[arm]
            action_masks[:, arm * 51 + 6 + hand_dims(name)] = active[arm]
        current_state = torch.zeros(STATE_DIM)
        current_state[:102] = torch.tensor(state) * side.float()
        current_state_mask = torch.zeros(STATE_DIM, dtype=torch.bool)
        current_state_mask[:102] = side
        return dict(
            pixel_values=pixel_values,
            input_ids=inputs["input_ids"][0],
            intrinsics=torch.tensor(scale_intrinsics(K, size, view)),
            depth=resize_nearest(depth, view),
            fov=torch.tensor(fov),
            current_state=current_state,
            current_state_mask=current_state_mask,
            action_labels=action_labels,
            action_masks=action_masks,
            future_depth=resize_nearest(future, view),
        )

    def collate(self, samples):
        batch = {key: torch.stack([s[key] for s in samples]) for key in samples[0] if key != "input_ids"}
        input_ids = pad_sequence([s["input_ids"] for s in samples], batch_first=True, padding_value=self.pad_id)
        batch.update(input_ids=input_ids, attention_mask=input_ids.ne(self.pad_id))
        return batch
