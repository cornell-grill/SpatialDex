import torch
import torch.nn.functional as F
from torch import nn

from spatialdex.model.pi3.layers import BlockRope
from spatialdex.model.pi3.pos_embed import PositionGetter, RoPE2D


class PointDecoder(nn.Module):
    def __init__(self, dim, out_dim, depth=5, num_heads=16, mlp_ratio=4):
        super().__init__()
        rope = RoPE2D(freq=100.0)
        self.blocks = nn.ModuleList([BlockRope(dim, num_heads, mlp_ratio, rope) for _ in range(depth)])
        self.linear_out = nn.Linear(dim, out_dim)

    def forward(self, hidden, xpos):
        for block in self.blocks:
            hidden = block(hidden, xpos)
        return self.linear_out(hidden)


class PointHead(nn.Module):
    def __init__(self, patch_size, dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(dim, 3 * patch_size**2)

    def forward(self, tokens, height, width):
        B = tokens.shape[0]
        feat = self.proj(tokens).transpose(-1, -2).view(B, -1, height // self.patch_size, width // self.patch_size)
        return F.pixel_shuffle(feat, self.patch_size).permute(0, 2, 3, 1)


class FutureGeometryHead(nn.Module):
    def __init__(self, in_channels, patch_size=14, image_size=224, feature_dim=1024):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.position_getter = PositionGetter()
        self.point_decoder = PointDecoder(in_channels, feature_dim)
        self.point_head = PointHead(patch_size, feature_dim)

    def forward(self, hidden):
        batch, frames, tokens = hidden.shape[:3]
        hidden = hidden.reshape(batch * frames, tokens, hidden.shape[-1])
        side = self.image_size // self.patch_size
        pos = self.position_getter(batch * frames, side, side, hidden.device).reshape(batch * frames, tokens, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            point_hidden = self.point_decoder(hidden, pos).float()
            out = self.point_head(point_hidden, self.image_size, self.image_size)
            out = out.reshape(batch, frames, self.image_size, self.image_size, -1)
            xy, z = out.split([2, 1], dim=-1)
            z = torch.exp(z.clamp(max=10.0))
            return torch.cat([xy * z, z], dim=-1)
