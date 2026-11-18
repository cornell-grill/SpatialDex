import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from spatialdex.data.wuji import (
    anchor_delta_to_absolute,
    camera_basis,
    hand_dims,
    pack_wuji_pose,
    transform_pose,
    unpack_wuji_pose,
)
from spatialdex.model import load_model

HORIZON = 16
STATE_DIM = 212


def resize_short_side(image, target=224):
    width, height = image.size
    return image.resize((int(width * target / height), target), Image.Resampling.LANCZOS)


def resize_pixel_intrinsics(K, source_size, target_size):
    resized = np.asarray(K, dtype=np.float64).copy()
    resized[..., 0, :] *= target_size[1] / source_size[1]
    resized[..., 1, :] *= target_size[0] / source_size[0]
    return resized


class Normalizer:
    def __init__(self, path):
        stats = json.loads(Path(path).read_text())
        self.state_mean = np.concatenate([np.array(stats[k]["mean"]) for k in ("state_left", "state_right")])
        self.state_std = np.concatenate([np.array(stats[k]["std"]) for k in ("state_left", "state_right")])
        self.action_mean = np.concatenate([np.array(stats[k]["mean"]) for k in ("action_left", "action_right")])
        self.action_std = np.concatenate([np.array(stats[k]["std"]) for k in ("action_left", "action_right")])

    def normalize_state(self, state):
        return (state - self.state_mean) / (self.state_std + 1e-07)

    def unnormalize_action(self, action):
        return action * (self.action_std + 1e-07) + self.action_mean


def pad_state(state102, active):
    state = torch.tensor(state102, dtype=torch.float32)
    mask = torch.tensor(np.asarray(active, dtype=bool)).repeat_interleave(51)
    masked = state * mask.to(state.dtype)
    padded = torch.zeros(STATE_DIM, dtype=torch.float32)
    padded_mask = torch.zeros(STATE_DIM, dtype=torch.bool)
    padded[:102] = masked
    padded_mask[:102] = mask
    return padded, padded_mask


class WujiPolicy:
    def __init__(self, config, checkpoint, *, ddim_steps=5, cfg_scale=5.0, device="cuda"):
        self.config = config
        self.normalizer = Normalizer(config["statistics_path"])
        self.model = load_model(config, checkpoint).to(device).eval()
        self.ddim_steps = ddim_steps
        self.cfg_scale = cfg_scale
        self.device = device

    def predict(self, rgb, depth, K, camera_to_world, state_left, state_right, instruction, active):
        image = resize_short_side(Image.fromarray(rgb))
        height, width = rgb.shape[:2]
        fov = np.array([2 * np.arctan2(width / 2.0, K[0, 0]), 2 * np.arctan2(height / 2.0, K[1, 1])], dtype=np.float32)
        basis = camera_basis(camera_to_world)
        state = np.concatenate([state_left, state_right]).astype(np.float32)
        state = transform_pose(state[None], basis, True, active)[0].astype(np.float32)
        norm_state = pack_wuji_pose(self.normalizer.normalize_state(state.copy()).astype(np.float32))
        current_state, current_state_mask = pad_state(norm_state, active)
        action_mask = torch.zeros((HORIZON, 192), dtype=torch.bool)
        for arm, side in enumerate(("left", "right")):
            if active[arm]:
                action_mask[:, arm * 51 : arm * 51 + 6] = True
                action_mask[:, arm * 51 + 6 + hand_dims(side)] = True
        vit_size = self.model.vision_tower.config.image_size
        depth_t = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)
        depth_t = torch.nn.functional.interpolate(depth_t, size=(vit_size, vit_size), mode="nearest").to(self.device)
        K_t = torch.from_numpy(resize_pixel_intrinsics(K, depth.shape[-2:], (vit_size, vit_size))).float().unsqueeze(0)
        norm_action = self.model.predict_action(
            images=np.array(image),
            instruction=instruction,
            current_state=current_state.unsqueeze(0),
            current_state_mask=current_state_mask.unsqueeze(0),
            action_mask=action_mask.unsqueeze(0),
            fov=torch.from_numpy(fov).unsqueeze(0),
            depth=depth_t,
            intrinsics=K_t.to(self.device),
            num_ddim_steps=self.ddim_steps,
            cfg_scale=self.cfg_scale,
        )
        norm = np.asarray(norm_action[:, :, :102])
        action = self.normalizer.unnormalize_action(np.stack([unpack_wuji_pose(chunk) for chunk in norm]))
        action = np.asarray(action)
        action = np.stack([anchor_delta_to_absolute(chunk, state, active) for chunk in action])
        action = np.stack([transform_pose(chunk, basis, False, active) for chunk in action]).astype(np.float32)
        return action[0, :, 0:26], action[0, :, 26:52]
