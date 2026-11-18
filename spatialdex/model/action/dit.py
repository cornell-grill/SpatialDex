# Modified from DiT: https://github.com/facebookresearch/DiT
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the LICENSE file of the DiT repository.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp, RmsNorm


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, t, max_period=10000):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding.to(next(self.mlp.parameters()).dtype))


class Projector(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.projector = nn.Sequential(*layers)

    def forward(self, x):
        return self.projector(x.to(next(self.projector.parameters()).dtype))


class LabelEmbedder(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(in_size, hidden_size)
        self.uncondition = nn.Parameter(torch.zeros(1, in_size))

    def forward(self, conditions, drop_ids=None):
        if drop_ids is not None:
            drop = drop_ids.unsqueeze(1).unsqueeze(1).expand(conditions.shape[0], *self.uncondition.shape)
            conditions = torch.where(drop, self.uncondition, conditions)
        return self.linear(conditions.to(self.linear.weight.dtype))


class SelfAttention(Attention):
    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key_value = nn.Linear(hidden_size, 2 * hidden_size)
        self.query_norm = RmsNorm(self.head_dim)
        self.key_norm = RmsNorm(self.head_dim)
        self.memory_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, memory, memory_mask):
        batch, query_len, hidden = x.shape
        memory_len = memory.shape[1]
        memory_mask = memory_mask.to(device=memory.device, dtype=torch.bool)
        has_memory = memory_mask.any(dim=-1)
        safe_mask = memory_mask.clone()
        safe_mask[~has_memory, 0] = True
        query = self.query(x).reshape(batch, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_value = self.key_value(self.memory_norm(memory))
        key, value = (
            key_value.reshape(batch, memory_len, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        )
        attended = F.scaled_dot_product_attention(
            self.query_norm(query),
            self.key_norm(key),
            value,
            attn_mask=safe_mask[:, None, None, :],
            dropout_p=0.0,
            scale=self.head_dim ** (-0.5),
        )
        projected = self.proj(attended.transpose(1, 2).reshape(batch, query_len, hidden))
        return projected * has_memory[:, None, None].to(projected.dtype)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, cross_attention=True, causal=False):
        super().__init__()
        self.cross_attention = cross_attention
        self.causal = causal
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=RmsNorm)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio), act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        if cross_attention:
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
            self.cross_attn = CrossAttention(hidden_size, num_heads)
            self.cross_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size))

    def forward(self, x, c, memory=None, memory_mask=None, wrist=None, wrist_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        mask = None
        if self.causal:
            seq_len = x.shape[1]
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype), 1)
            mask = mask.masked_fill(mask == 1, float("-inf"))
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=mask)
        if not self.cross_attention:
            return x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        shift_cross, scale_cross, gate_cross = self.cross_adaLN_modulation(c).chunk(3, dim=1)
        query = modulate(self.norm_cross(x), shift_cross, scale_cross)
        x = x + gate_cross.unsqueeze(1) * self.cross_attn(query, memory, memory_mask)
        x = x + self.cross_attn(query, wrist, wrist_mask)
        return x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
    def __init__(
        self,
        action_dim,
        token_size,
        state_dim,
        horizon=16,
        hidden_size=768,
        depth=12,
        num_heads=12,
        cross_attention=True,
        causal=False,
        condition_dropout=0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.cross_attention = cross_attention
        self.condition_dropout = condition_dropout
        self.x_embedder = Projector(
            [nn.Linear(2 * action_dim, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size)]
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.z_embedder = LabelEmbedder(token_size, hidden_size)
        if cross_attention:
            self.reasoning_type_embeddings = nn.ParameterDict({"wrist": nn.Parameter(torch.zeros(hidden_size))})
        self.state_embedder = Projector(
            [
                nn.Linear(2 * state_dim, 4 * hidden_size),
                nn.GELU(),
                nn.Linear(4 * hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            ]
        )
        self.positional_embedding = nn.Parameter(torch.zeros(horizon + 2, hidden_size))
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, cross_attention=cross_attention, causal=causal) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, action_dim)

    def forward(self, x, t, z, state, state_mask, condition_mask=None, wrist=None, wrist_mask=None, x_mask=None):
        if x.shape[-1] == self.action_dim:
            x_mask = x_mask.to(device=x.device, dtype=x.dtype)
            x = torch.cat([x * x_mask, x_mask], dim=2)
        state = state * state_mask
        s = self.state_embedder(torch.cat([state, state_mask.to(state.dtype)], dim=-1))
        x = self.x_embedder(x)
        t = self.t_embedder(t)
        drop_ids = None
        if self.training and self.condition_dropout > 0:
            drop_ids = torch.rand(z.shape[0], device=z.device) < self.condition_dropout
        z = self.z_embedder(z, drop_ids)
        if not self.cross_attention:
            c = z.mean(dim=1) + t
            x = torch.cat((z, s, x), dim=1)
            pos = self.positional_embedding.to(device=x.device, dtype=x.dtype)
            x = x + torch.cat([pos[:1].expand(z.shape[1], -1), pos[1:]], dim=0)
            for block in self.blocks:
                x = block(x, c)
            return self.final_layer(x, c)[:, -self.horizon :, :]
        condition_mask = condition_mask.to(device=z.device, dtype=torch.bool)
        z = torch.where(condition_mask.unsqueeze(-1), z, torch.zeros_like(z))
        c = z[:, 0] * condition_mask[:, :1].to(dtype=z.dtype) + t
        wrist_mask = wrist_mask.to(device=wrist.device, dtype=torch.bool)
        wrist = self.z_embedder.linear(wrist.to(dtype=self.z_embedder.linear.weight.dtype))
        wrist = wrist + self.reasoning_type_embeddings["wrist"].to(dtype=wrist.dtype)
        wrist = torch.where(wrist_mask.unsqueeze(-1), wrist, torch.zeros_like(wrist))
        x = torch.cat((s, x), dim=1)
        x = x + self.positional_embedding[1:].to(device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x = block(x, c, z, condition_mask, wrist, wrist_mask)
        return self.final_layer(x, c)[:, -self.horizon :, :]

    def forward_with_cfg(
        self, x, t, z, x_mask, cfg_scale, state, state_mask, condition_mask=None, wrist=None, wrist_mask=None
    ):
        uncondition = self.z_embedder.uncondition[None].expand_as(z)
        half = x[: len(x) // 2] * x_mask
        half = torch.cat([half, x_mask], dim=2)
        state = torch.cat([state, state], dim=0)
        state_mask = torch.cat([state_mask, state_mask], dim=0)
        combined = torch.cat([half, half], dim=0).to(next(self.x_embedder.parameters()).dtype)
        out = self.forward(
            combined,
            t,
            torch.cat([z, uncondition], dim=0),
            state * state_mask,
            state_mask,
            condition_mask,
            wrist,
            wrist_mask,
        )
        cond, uncond = torch.split(out, len(out) // 2, dim=0)
        guided = uncond + cfg_scale * (cond - uncond)
        return torch.cat([guided, guided], dim=0)
