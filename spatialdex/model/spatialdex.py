import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from spatialdex.model.action.split_diffusion_policy import SplitDiffusionPolicy
from spatialdex.model.geometry_encoder import SIGLIP_MEAN, SIGLIP_STD
from spatialdex.model.geometry_head import FutureGeometryHead
from spatialdex.model.sonata_encoder import SonataPointTransformer
from spatialdex.model.vlm import build_vlm
from spatialdex.model.wrist_encoder import (
    FrozenDinoV2PatchEncoder,
    WristActionTokenPooler,
    prepare_dino_pixels_from_siglip,
)
from spatialdex.utils.tensor import move_masked_to_left, move_masked_to_left_ids

COGNITION_TOKEN_ID = 10


class MLPProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, x):
        return self.projector(x)


class GroupedChannelAttention(nn.Module):
    def __init__(self, dim, num_groups, attn_dim=16, diagonal_bias=0.0):
        super().__init__()
        self.num_groups = num_groups
        self.group_dim = dim // num_groups
        self.scale = attn_dim ** (-0.5)
        self.diagonal_bias = float(diagonal_bias)
        self.q_proj = nn.Linear(2, attn_dim)
        self.k_proj = nn.Linear(2, attn_dim)

    @staticmethod
    def _channel_stats(x, valid_mask):
        mask = valid_mask.to(dtype=x.dtype).unsqueeze(-1).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        mean = (x * mask).sum(dim=1) / denom
        var = ((x - mean.unsqueeze(1)).pow(2) * mask).sum(dim=1) / denom
        std = torch.sqrt(var.clamp_min(1e-06))
        return torch.stack([mean, std], dim=-1)

    def forward(self, image_feature, geo_feature, valid_mask):
        bsz, num_tokens, channels = geo_feature.shape
        image_grouped = image_feature.reshape(bsz, num_tokens, self.num_groups, self.group_dim)
        geo_grouped = geo_feature.reshape(bsz, num_tokens, self.num_groups, self.group_dim)
        q = self.q_proj(self._channel_stats(image_grouped, valid_mask).float())
        k = self.k_proj(self._channel_stats(geo_grouped, valid_mask).float())
        scores = torch.einsum("bgda,bgea->bgde", q, k) * self.scale
        if self.diagonal_bias:
            eye = torch.eye(self.group_dim, device=scores.device, dtype=scores.dtype)
            scores = scores + eye.view(1, 1, self.group_dim, self.group_dim) * self.diagonal_bias
        attn = scores.softmax(dim=-1).to(dtype=geo_feature.dtype)
        aligned = torch.einsum("bgde,bnge->bngd", attn, geo_grouped)
        return aligned.reshape(bsz, num_tokens, channels)


class ChannelAlignedGeoFusion(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=2.0,
        channel_attention_groups=32,
        channel_attention_dim=16,
        diagonal_bias=0.0,
        learned_gate=False,
        image_dropout=0.0,
        geo_dropout=0.0,
    ):
        super().__init__()
        self.image_dropout = image_dropout
        self.geo_dropout = geo_dropout
        self.image_norm = nn.LayerNorm(dim)
        self.geo_norm = nn.LayerNorm(dim)
        self.geo_to_image = nn.Linear(dim, dim)
        self.channel_attention = GroupedChannelAttention(
            dim, channel_attention_groups, attn_dim=channel_attention_dim, diagonal_bias=diagonal_bias
        )
        hidden_dim = int(dim * mlp_ratio)
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim * 4), nn.Linear(dim * 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim)
        )
        if learned_gate:
            self.alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("alpha", None)

    def forward(self, image_feature, geo_feature, valid_mask):
        if self.training and (self.image_dropout > 0 or self.geo_dropout > 0):
            batch = image_feature.shape[0]
            image_keep = torch.ones(batch, 1, 1, device=image_feature.device, dtype=image_feature.dtype)
            geo_keep = torch.ones_like(image_keep)
            if self.image_dropout > 0:
                image_keep = (torch.rand(batch, 1, 1, device=image_feature.device) >= self.image_dropout).to(
                    image_feature.dtype
                )
            if self.geo_dropout > 0:
                geo_keep = (torch.rand(batch, 1, 1, device=image_feature.device) >= self.geo_dropout).to(
                    image_feature.dtype
                )
            geo_keep = torch.where((image_keep == 0) & (geo_keep == 0), torch.ones_like(geo_keep), geo_keep)
            image_feature = image_feature * image_keep
            geo_feature = geo_feature * geo_keep
            valid_mask = valid_mask & (geo_keep.squeeze(-1) > 0)
        image_aligned = self.image_norm(image_feature)
        geo_aligned = self.geo_norm(self.geo_to_image(geo_feature))
        geo_aligned = self.channel_attention(image_aligned, geo_aligned, valid_mask)
        fusion_input = torch.cat(
            [image_aligned, geo_aligned, image_aligned * geo_aligned, (image_aligned - geo_aligned).abs()], dim=-1
        )
        delta = self.fuse(fusion_input)
        if self.alpha is None:
            return image_feature + delta, geo_feature, valid_mask
        return image_feature + torch.tanh(self.alpha).to(delta.dtype) * delta, geo_feature, valid_mask


class GeoActionTokenProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, geo_feature, valid_mask):
        mask = valid_mask.to(dtype=geo_feature.dtype).unsqueeze(-1)
        count = mask.sum(dim=1)
        pooled = (geo_feature * mask).sum(dim=1) / count.clamp_min(1.0)
        has_valid = (count.squeeze(-1) > 0).to(dtype=geo_feature.dtype).unsqueeze(-1)
        return self.projector(pooled).unsqueeze(1) * has_valid.unsqueeze(1)


def backproject_depth(K, depth):
    batch, height, width = depth.shape
    fx, fy = K[:, 0, 0].view(batch, 1, 1), K[:, 1, 1].view(batch, 1, 1)
    cx, cy = K[:, 0, 2].view(batch, 1, 1), K[:, 1, 2].view(batch, 1, 1)
    v, u = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    u = u.unsqueeze(0).expand(batch, -1, -1)
    v = v.unsqueeze(0).expand(batch, -1, -1)
    return torch.stack(((u - cx) * depth / fx, (v - cy) * depth / fy, depth), dim=-1)


def future_geometry_loss(pred, target, beta=0.05, xyz_weight=0.25):
    valid = target[..., 2] > 1e-06
    pred_valid, target_valid = pred.float()[valid], target[valid]
    depth_loss = F.smooth_l1_loss(pred_valid[:, 2], target_valid[:, 2], beta=beta, reduction="mean")
    xyz_loss = F.smooth_l1_loss(pred_valid, target_valid, beta=beta, reduction="mean")
    return depth_loss + xyz_weight * xyz_loss


class SpatialDex(nn.Module):
    def __init__(self, configs, training_heads=False):
        super().__init__()
        self.configs = configs
        self.use_bf16 = False
        self.patch_reso = int(configs["patch_reso"])
        act_cfg = configs["action_model"]
        geo_cfg = configs.get("geo_fusion", {})
        wrist_cfg = configs.get("wrist_camera_conditioning")
        self.full_memory = act_cfg.get("condition_mode", "legacy_concat") == "cross_attention"
        self.geo_token_loss_weight = float(configs.get("geo_action_loss_weight", 0.0))
        self.repeated_diffusion_steps = int(configs.get("repeated_diffusion_steps", 8))

        self.processor, self.backbone = build_vlm(configs["vlm"])
        self.tokenizer = self.processor.tokenizer
        vision_config = self.vision_tower.config
        patch_size, image_size = vision_config.patch_size, vision_config.image_size
        step = patch_size / self.patch_reso
        y, x = torch.meshgrid(torch.arange(0, image_size, step), torch.arange(0, image_size, step), indexing="ij")
        y, x = y + step / 2, x + step / 2
        self.register_buffer("uv_h", torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1), persistent=False)

        vision_hidden = vision_config.hidden_size
        self.sonata_encoder = SonataPointTransformer(out_dim=vision_hidden)
        self.geo_feature_attention = nn.MultiheadAttention(
            embed_dim=vision_hidden, num_heads=8, batch_first=True, dropout=0.1
        )
        self.geo_query_token = nn.Parameter(torch.zeros(1, 1, vision_hidden))
        self.background_geo_token = nn.Parameter(torch.zeros(1, 1, vision_hidden))
        self.multi_modal_projector = self.model.multi_modal_projector
        self.geo_channel_fusion = ChannelAlignedGeoFusion(
            dim=vision_hidden,
            mlp_ratio=float(geo_cfg.get("mlp_ratio", 2.0)),
            channel_attention_groups=int(geo_cfg.get("channel_attention_groups", 32)),
            channel_attention_dim=int(geo_cfg.get("channel_attention_dim", 16)),
            diagonal_bias=float(geo_cfg.get("channel_attention_diagonal_bias", 0.0)),
            learned_gate=bool(geo_cfg.get("use_learned_gate", True)),
            image_dropout=float(geo_cfg.get("image_dropout_prob", 0.0)),
            geo_dropout=float(geo_cfg.get("geo_dropout_prob", 0.0)),
        )
        token_size = int(act_cfg.get("token_size", self.hidden_size))
        self.geo_action_projector = None
        if configs.get("enable_direct_geo_action_token", False):
            self.geo_action_projector = GeoActionTokenProjector(vision_hidden, token_size)
        self.geo_prediction = None
        if training_heads and configs.get("enable_future_geo_pred_supervision", False):
            self.geo_prediction = FutureGeometryHead(vision_config.projection_dim, patch_size, image_size)

        self.wrist_dino_encoder = None
        if wrist_cfg is not None:
            self.wrist_dino_encoder = FrozenDinoV2PatchEncoder(image_size=int(wrist_cfg["dino"].get("image_size", 224)))
            self.wrist_action_pooler = WristActionTokenPooler(
                vision_dim=self.wrist_dino_encoder.feature_dim,
                context_dim=self.hidden_size,
                action_dim=token_size,
                tokens_per_view=int(wrist_cfg.get("tokens_per_view", 256)),
                pool_dim=int(wrist_cfg.get("pool_dim", 512)),
                num_heads=int(wrist_cfg.get("num_heads", 8)),
                state_dim=int(configs["state_encoder"]["state_dim"]),
            )
        self.act_model = SplitDiffusionPolicy(
            token_size=token_size,
            in_channels=int(act_cfg.get("action_dim", 192)),
            state_dim=int(configs["state_encoder"]["state_dim"]),
            split_config=configs.get("split_action_heads", {}),
            cross_attention=self.full_memory,
            self_attention_mode=act_cfg.get("self_attention_mode", "causal"),
            hand_loss_weight=float(configs.get("hand_loss_weight", 0.4)),
        )
        self.fov_encoder = MLPProjector(2, self.hidden_size)
        self.cognition_token = nn.Parameter(torch.zeros(self.hidden_size))

    @property
    def model(self):
        return self.backbone

    @property
    def hidden_size(self):
        return self.model.config.text_config.hidden_size

    @property
    def text_tower(self):
        return self.model.language_model.model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    def backproject_patch(self, K, depth):
        b, c, h, w = depth.shape
        patch_size, reso = self.vision_tower.config.patch_size, self.patch_reso
        hp, wp = h // patch_size, w // patch_size
        patch_depth = F.interpolate(depth, size=(hp * reso, wp * reso), mode="area").reshape(b, c, -1)
        p_cam = torch.linalg.inv(K.float()) @ self.uv_h.float() * patch_depth
        return p_cam.reshape(b, 3, hp, reso, wp, reso).permute(0, 2, 4, 3, 5, 1).reshape(b, hp * wp, -1)

    def encode_geometry(self, pixel_values, intrinsics, depth, image_feature):
        reso = self.patch_reso
        xyz = self.backproject_patch(intrinsics, depth)
        batch = xyz.shape[0]
        xyz = xyz.reshape(batch, -1, 3)
        valid_mask = xyz.norm(dim=-1) > 1e-06
        b, c, h, w = pixel_values.shape
        patch_size = self.vision_tower.config.patch_size
        hp, wp = h // patch_size, w // patch_size
        patch_color = F.interpolate(pixel_values, size=(hp * reso, wp * reso), mode="area")
        color = patch_color.reshape(b, c, hp, reso, wp, reso).permute(0, 2, 4, 3, 5, 1).reshape(b, -1, c)
        point_valid = valid_mask.clone()
        sonata_dtype = next(self.sonata_encoder.parameters()).dtype
        with torch.autocast("cuda", enabled=False):
            point_features = self.sonata_encoder(
                coord=xyz.to(dtype=sonata_dtype), color=color.to(dtype=sonata_dtype), mask=valid_mask
            )
        point_features = point_features.to(image_feature.dtype)
        batch, total_points, dim = point_features.shape
        points_per_patch = reso**2
        num_patches = total_points // points_per_patch
        point_features = point_features.reshape(batch * num_patches, points_per_patch, dim)
        point_valid = point_valid.reshape(batch * num_patches, points_per_patch)
        patch_valid = point_valid.any(dim=1)
        query = self.geo_query_token.to(point_features.dtype).expand(batch * num_patches, -1, -1)
        pooled = torch.zeros(batch * num_patches, 1, dim, dtype=point_features.dtype, device=point_features.device)
        if patch_valid.any():
            index = patch_valid.nonzero(as_tuple=False).squeeze(-1)
            features = point_features[index]
            out, _ = self.geo_feature_attention(
                query[index], features, features, key_padding_mask=point_valid[index].logical_not()
            )
            pooled[index] = out.to(dtype=pooled.dtype)
        geo_feature = pooled.reshape(batch, num_patches, dim)
        patch_valid = patch_valid.reshape(batch, num_patches)
        background = self.background_geo_token.to(dtype=geo_feature.dtype)
        geo_feature = torch.where(patch_valid.unsqueeze(-1), geo_feature, background.expand_as(geo_feature))
        fused, geo_used, valid_used = self.geo_channel_fusion(image_feature, geo_feature, patch_valid)
        has_geometry = patch_valid.any(dim=1, keepdim=True)
        fused = torch.where(has_geometry.unsqueeze(-1), fused, image_feature)
        geo_token = None
        if self.geo_action_projector is not None:
            geo_used = torch.where(has_geometry.unsqueeze(-1), geo_used, torch.zeros_like(geo_used))
            geo_token = self.geo_action_projector(geo_used, valid_used & has_geometry)
        return fused, geo_token

    def image_features(self, pixel_values, intrinsics, depth):
        siglip_pixels = TF.normalize(pixel_values, mean=SIGLIP_MEAN, std=SIGLIP_STD)
        image_feature = self.vision_tower(siglip_pixels).last_hidden_state
        fused, geo_token = self.encode_geometry(pixel_values, intrinsics, depth, image_feature)
        features = self.multi_modal_projector(fused)
        return features / self.text_tower.config.hidden_size**0.5, geo_token

    def vlm_inputs(self, pixel_values, input_ids, intrinsics, depth, attention_mask, fov):
        batch = input_ids.shape[0]
        attention_mask = attention_mask.to(dtype=torch.bool)
        word_embeds = self.model.get_input_embeddings()(input_ids)
        dtype = word_embeds.dtype
        fov_embeds = self.fov_encoder(fov).to(dtype)
        cog_embeds = self.cognition_token.unsqueeze(0).unsqueeze(0).expand(batch, -1, -1)
        ones = torch.ones((batch, 1), dtype=torch.bool, device=attention_mask.device)
        inputs_embeds = torch.cat([word_embeds, fov_embeds.unsqueeze(1), cog_embeds], dim=1).to(dtype)
        inputs_masks = torch.cat([attention_mask, ones, ones], dim=1)
        extra_ids = torch.full((batch, 2), COGNITION_TOKEN_ID, dtype=input_ids.dtype, device=input_ids.device)
        inputs_embeds, _ = move_masked_to_left(inputs_embeds, inputs_masks)
        input_ids, inputs_masks = move_masked_to_left_ids(torch.cat([input_ids, extra_ids], dim=1), inputs_masks)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            image_features, geo_token = self.image_features(pixel_values, intrinsics, depth)
        image_tokens = input_ids == self.model.config.image_token_index
        image_mask = image_tokens.unsqueeze(-1).expand_as(inputs_embeds)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask.to(inputs_embeds.device), image_features)
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        causal_mask = self.model._update_causal_mask(inputs_masks, None, None, cache_position, inputs_embeds, False)
        inputs = dict(
            attention_mask=causal_mask,
            position_ids=cache_position.unsqueeze(0) + 1,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
        )
        return inputs, inputs_masks, image_tokens, geo_token

    def vlm_features(self, pixel_values, input_ids, intrinsics, depth, attention_mask, fov):
        inputs, inputs_masks, image_tokens, geo_token = self.vlm_inputs(
            pixel_values, input_ids, intrinsics, depth, attention_mask, fov
        )
        runtime_dtype = torch.bfloat16 if self.use_bf16 else inputs["inputs_embeds"].dtype
        inputs["inputs_embeds"] = inputs["inputs_embeds"].to(runtime_dtype)
        if torch.is_floating_point(inputs["attention_mask"]):
            inputs["attention_mask"] = inputs["attention_mask"].to(runtime_dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            outputs = self.model.language_model(
                past_key_values=None, use_cache=False, output_hidden_states=True, num_logits_to_keep=0, **inputs
            )
        return outputs.hidden_states[-1], inputs_masks, image_tokens, geo_token

    def wrist_tokens(
        self, wrist_pixel_values, wrist_view_mask, context, context_mask, current_state, current_state_mask
    ):
        batch, views = wrist_pixel_values.shape[:2]
        flat = wrist_pixel_values.reshape(batch * views, *wrist_pixel_values.shape[2:])
        dino_pixels = prepare_dino_pixels_from_siglip(flat)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16 and dino_pixels.is_cuda):
            patches = self.wrist_dino_encoder(dino_pixels)
        patches = patches.reshape(batch, views, patches.shape[1], patches.shape[2])
        tokens, mask = self.wrist_action_pooler(
            patches,
            wrist_view_mask,
            ego_context=context,
            ego_context_mask=context_mask,
            current_state=current_state,
            current_state_mask=current_state_mask,
        )
        return tokens, mask

    @staticmethod
    def cognition_feature(hidden, attention_mask):
        cumulative = attention_mask.cumsum(dim=1)
        last = (cumulative == cumulative.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
        index = last.unsqueeze(-1).expand(-1, hidden.size(-1))
        return hidden.gather(1, index.unsqueeze(1))

    def action_condition(self, hidden, attention_mask, geo_token):
        batch = hidden.shape[0]
        cognition = self.cognition_feature(hidden, attention_mask)
        if self.full_memory:
            features = torch.cat([cognition, hidden], dim=1)
            ones = torch.ones(batch, 1, dtype=torch.bool, device=attention_mask.device)
            condition_mask = torch.cat([ones, attention_mask.to(dtype=torch.bool)], dim=1)
        else:
            features, condition_mask = cognition, None
            if geo_token is not None:
                features = torch.cat([features, geo_token.to(device=features.device, dtype=features.dtype)], dim=1)
        features = features.to(next(self.act_model.parameters()).dtype)
        if condition_mask is not None:
            condition_mask = condition_mask.to(device=features.device)
        return features, condition_mask

    @staticmethod
    def repeat(x, repeats):
        if x is None:
            return None
        return x.unsqueeze(0).repeat(repeats, *([1] * x.dim())).view(x.shape[0] * repeats, *x.shape[1:])

    def sample_actions(
        self,
        features,
        condition_mask,
        action_mask,
        current_state,
        current_state_mask,
        wrist,
        wrist_mask,
        cfg_scale,
        num_ddim_steps,
        repeats,
    ):
        dtype = features.dtype
        if wrist is not None:
            wrist = wrist.to(device=features.device, dtype=dtype)
            wrist_mask = wrist_mask.to(device=features.device, dtype=torch.bool)
        return self.act_model.sample(
            self.repeat(features, repeats),
            cfg_scale,
            self.repeat(current_state.unsqueeze(1), repeats),
            self.repeat(current_state_mask.unsqueeze(1), repeats),
            num_ddim_steps,
            self.repeat(action_mask, repeats),
            condition_mask=self.repeat(condition_mask, repeats),
            wrist=self.repeat(wrist, repeats),
            wrist_mask=self.repeat(wrist_mask, repeats),
        )

    @torch.no_grad()
    def predict_action(
        self,
        images,
        instruction,
        current_state,
        current_state_mask,
        action_mask,
        fov,
        depth,
        intrinsics,
        num_ddim_steps=10,
        cfg_scale=5.0,
        sample_times=1,
    ):
        batch = current_state.shape[0]
        if not isinstance(images, (list, tuple)):
            images = [images]
        images = [Image.fromarray(im) if isinstance(im, np.ndarray) else im for im in images]
        device = "cuda"
        wrist_pixel_values = None
        if self.wrist_dino_encoder is not None:
            wrist_pixel_values = self.processor.image_processor(images=list(images[1:]), return_tensors="pt").to(device)
            wrist_pixel_values = wrist_pixel_values["pixel_values"].unsqueeze(0)
        inputs = self.processor(text="<image>" + instruction, images=images[0], return_tensors="pt").to(device)
        pixel_values, input_ids = inputs["pixel_values"], inputs["input_ids"]
        device = input_ids.device
        action_mask = action_mask.to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        current_state_mask = current_state_mask.to(device)
        current_state = current_state.to(device)
        fov = fov.to(device)
        depth = torch.as_tensor(depth, device=device, dtype=torch.float32)
        intrinsics = torch.as_tensor(intrinsics, device=device, dtype=torch.float32)
        hidden, masks, _, geo_token = self.vlm_features(pixel_values, input_ids, intrinsics, depth, attention_mask, fov)
        wrist = wrist_mask = None
        if wrist_pixel_values is not None:
            wrist_view_mask = torch.ones((batch, 2), dtype=torch.bool, device=wrist_pixel_values.device)
            wrist, wrist_mask = self.wrist_tokens(
                wrist_pixel_values, wrist_view_mask, hidden, masks, current_state, current_state_mask
            )
        features, condition_mask = self.action_condition(hidden, masks, geo_token)
        samples = self.sample_actions(
            features,
            condition_mask,
            action_mask,
            current_state,
            current_state_mask,
            wrist,
            wrist_mask,
            cfg_scale,
            num_ddim_steps,
            sample_times,
        )
        return samples.cpu().numpy() * action_mask.cpu().numpy()

    def forward(self, batch, global_step=0):
        hidden, masks, image_tokens, geo_token = self.vlm_features(
            batch["pixel_values"],
            batch["input_ids"],
            batch["intrinsics"],
            batch["depth"],
            batch["attention_mask"],
            batch["fov"],
        )
        n = self.repeated_diffusion_steps
        actions = self.repeat(batch["action_labels"], n)
        action_masks = self.repeat(batch["action_masks"], n)
        state = self.repeat(batch["current_state"].unsqueeze(1), n)
        state_mask = self.repeat(batch["current_state_mask"].unsqueeze(1), n)
        features, _ = self.action_condition(hidden, masks, geo_token)
        losses = self.act_model.loss(actions, self.repeat(features, n), action_masks, state, state_mask, global_step)
        total = losses["loss"]
        if self.geo_token_loss_weight > 0 and geo_token is not None:
            dtype = next(self.act_model.parameters()).dtype
            aux = self.act_model.loss(
                actions,
                self.repeat(geo_token.to(dtype), n),
                action_masks,
                torch.zeros_like(state),
                torch.zeros_like(state_mask, dtype=torch.bool),
                global_step,
            )
            losses["geo_token_loss"] = aux["loss"].detach()
            total = total + aux["loss"] * self.geo_token_loss_weight
        if self.geo_prediction is not None:
            batch_size, _, hidden_dim = hidden.shape
            tokens = image_tokens.sum(dim=1)
            patches = (self.vision_tower.config.image_size // self.vision_tower.config.patch_size) ** 2
            frames = int(tokens[0].item()) // patches
            image_hidden = hidden[image_tokens].reshape(batch_size, frames, patches, hidden_dim)
            pred = self.geo_prediction(image_hidden)
            target = backproject_depth(batch["intrinsics"].float(), batch["future_depth"][:, 0].float())
            geo_loss = future_geometry_loss(pred, target.reshape(batch_size, frames, *target.shape[1:]))
            losses["geometry_loss"] = geo_loss.detach()
            total = total + geo_loss * 0.4
        losses["loss"] = total
        return losses
