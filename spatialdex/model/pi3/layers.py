# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class AttentionRope(nn.Module):
    def __init__(self, dim, num_heads, rope):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rope = rope

    def forward(self, x, xpos):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        q, k = self.rope(q, xpos), self.rope(k, xpos)
        if q.dtype == torch.bfloat16:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = scaled_dot_product_attention(q, k, v)
        else:
            x = scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape([B, N, C]))


class BlockRope(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, rope):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-06)
        self.attn = AttentionRope(dim, num_heads, rope)
        self.norm2 = nn.LayerNorm(dim, eps=1e-06)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, xpos):
        x = x + self.attn(self.norm1(x), xpos)
        return x + self.mlp(self.norm2(x))
