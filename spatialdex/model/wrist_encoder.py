import timm
import torch
import torch.nn as nn

WRIST_VIEWS = 2
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


def prepare_dino_pixels_from_siglip(pixel_values):
    mean = pixel_values.new_tensor(SIGLIP_MEAN).view(1, 3, 1, 1)
    std = pixel_values.new_tensor(SIGLIP_STD).view(1, 3, 1, 1)
    raw = pixel_values * std + mean
    return (raw - pixel_values.new_tensor(DINO_MEAN).view(1, 3, 1, 1)) / pixel_values.new_tensor(DINO_STD).view(
        1, 3, 1, 1
    )


class FrozenDinoV2PatchEncoder(nn.Module):
    def __init__(self, image_size=224):
        super().__init__()
        self.encoder = timm.create_model(
            "vit_large_patch14_dinov2", pretrained=False, num_classes=0, img_size=image_size
        )
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.feature_dim = int(self.encoder.embed_dim)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, pixels):
        parameter = next(self.encoder.parameters())
        with torch.no_grad():
            features = self.encoder.forward_features(pixels.to(device=parameter.device, dtype=parameter.dtype))
        return features[:, 1:]


class WristActionTokenPooler(nn.Module):
    def __init__(
        self, *, vision_dim, context_dim, action_dim, tokens_per_view=256, pool_dim=512, num_heads=8, state_dim=212
    ):
        super().__init__()
        self.tokens_per_view = int(tokens_per_view)
        self.pool_dim = int(pool_dim)
        self.vision_norm = nn.LayerNorm(vision_dim)
        self.vision_projection = nn.Linear(vision_dim, pool_dim)
        self.view_embeddings = nn.Parameter(torch.zeros(WRIST_VIEWS, pool_dim))
        self.spatial_position_embeddings = nn.Parameter(torch.zeros(self.tokens_per_view, pool_dim))
        self.output_norm = nn.LayerNorm(pool_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(pool_dim, 2 * pool_dim), nn.GELU(approximate="tanh"), nn.Linear(2 * pool_dim, pool_dim)
        )
        self.action_projection = nn.Linear(pool_dim, action_dim)

    def forward(
        self,
        patch_features,
        view_mask,
        *,
        ego_context=None,
        ego_context_mask=None,
        current_state=None,
        current_state_mask=None,
    ):
        batch = patch_features.shape[0]
        view_mask = view_mask.to(device=patch_features.device, dtype=torch.bool)
        normalized = self.vision_norm(patch_features.to(dtype=self.vision_norm.weight.dtype))
        patches = self.vision_projection(normalized.to(dtype=self.vision_projection.weight.dtype))
        role = self.view_embeddings.to(dtype=patches.dtype)[None, :, None, :]
        spatial = self.spatial_position_embeddings.to(dtype=patches.dtype)[None, None]
        pooled = (patches + role + spatial).reshape(batch * WRIST_VIEWS, self.tokens_per_view, self.pool_dim)
        pooled = pooled + self.output_mlp(self.output_norm(pooled))
        tokens = self.action_projection(self.output_norm(pooled)).reshape(batch, WRIST_VIEWS * self.tokens_per_view, -1)
        mask = view_mask[:, :, None].expand(-1, -1, self.tokens_per_view).reshape(batch, -1)
        tokens = torch.where(mask[..., None], tokens, torch.zeros_like(tokens))
        return tokens, mask
