from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def patchify2d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert (B,C,H,W) into patch tokens (B, N, C*P*P)."""
    b, c, h, w = x.shape
    p = int(patch_size)
    if h % p != 0 or w % p != 0:
        raise ValueError(f"H,W must be divisible by patch_size={p}, got {(h, w)}")

    gh, gw = h // p, w // p
    x = x.reshape(b, c, gh, p, gw, p)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, gh, gw, p, p, C)
    return x.view(b, gh * gw, p * p * c)


def unpatchify2d(tokens: torch.Tensor, patch_size: int, *, out_channels: int) -> torch.Tensor:
    """Convert patch tokens (B,N,P*P*C) back to (B,C,H,W)."""
    b, n, d = tokens.shape
    p = int(patch_size)

    c = int(out_channels)
    if d != p * p * c:
        raise ValueError(f"Token dim {d} != patch_size^2*out_channels ({p*p*c})")

    grid = int(n ** 0.5)
    if grid * grid != n:
        raise ValueError("Number of tokens must be a perfect square for 2D unpatchify.")

    x = tokens.view(b, grid, grid, p, p, c)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B,C,gh,p,gw,p)
    return x.view(b, c, grid * p, grid * p)


class ConvPatchEmbed2d(nn.Module):
    """Patch embedding via strided conv.

    This is a lightweight replacement for timm's PatchEmbed.
    """

    def __init__(self, *, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        p = int(patch_size)
        self.patch_size = p
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=p, stride=p, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W) -> (B, N, D)
        y = self.proj(x)
        b, d, gh, gw = y.shape
        return y.flatten(2).transpose(1, 2).contiguous()
