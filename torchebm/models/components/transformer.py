from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: (B,N,D), shift/scale: (B,D)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MultiheadSelfAttention(nn.Module):
    """Self-attention with fused QKV projection, dispatching to PyTorch SDPA.

    Uses ``F.scaled_dot_product_attention`` so the flash-attention and
    memory-efficient kernels are selected automatically on supported hardware
    (Ampere+ GPUs in fp16/bf16, any CUDA device via mem-efficient fallback).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=False)
        y = y.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaLNZeroBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning.

    Takes a per-sample conditioning vector `cond` (B, cond_dim) and applies it
    to modulate norms + gate residuals.

    This is a reusable block; it does not assume anything about what `cond`
    represents (time, labels, text, etc.).
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        cond_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        attn: Optional[nn.Module] = None,
        mlp: Optional[nn.Module] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.cond_dim = int(cond_dim) if cond_dim is not None else int(embed_dim)

        self.norm1 = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=eps)
        self.attn = attn if attn is not None else MultiheadSelfAttention(self.embed_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=eps)
        self.mlp = mlp if mlp is not None else FeedForward(self.embed_dim, mlp_ratio=mlp_ratio)

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.cond_dim, 6 * self.embed_dim, bias=True),
        )

        # Zero-init to start near identity.
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B,N,D), cond: (B,cond_dim)
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(cond).chunk(6, dim=1)

        x = x + gate1.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift1, scale1))
        x = x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x
