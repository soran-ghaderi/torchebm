from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .patch import unpatchify2d
from .transformer import modulate


class AdaLNZeroPatchHead(nn.Module):
    """Final layer that maps token features to patch pixels with adaLN-Zero.

    `grid_size` is the `(gh, gw)` token grid; `None` infers a square grid at
    forward time (rectangular grids must set it).
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        cond_dim: Optional[int] = None,
        patch_size: int,
        out_channels: int,
        grid_size: Optional[Tuple[int, int]] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.cond_dim = int(cond_dim) if cond_dim is not None else int(embed_dim)
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        self.grid_size = tuple(int(s) for s in grid_size) if grid_size is not None else None

        self.norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=eps)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.cond_dim, 2 * self.embed_dim, bias=True),
        )
        self.proj = nn.Linear(self.embed_dim, self.patch_size * self.patch_size * self.out_channels, bias=True)

        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(cond).chunk(2, dim=1)
        tokens = modulate(self.norm(tokens), shift, scale)
        patches = self.proj(tokens)
        return unpatchify2d(
            patches,
            self.patch_size,
            out_channels=self.out_channels,
            grid_size=self.grid_size,
        )


class AdaLNZeroLinearHead(nn.Module):
    """Final layer mapping a vector stream to `out_dim` with adaLN-Zero.

    The vector (B, D) sibling of `AdaLNZeroPatchHead`: modulation and
    projection are zero-initialized so the output starts at zero.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        cond_dim: Optional[int] = None,
        out_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.cond_dim = int(cond_dim) if cond_dim is not None else int(embed_dim)
        self.out_dim = int(out_dim)

        self.norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=eps)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.cond_dim, 2 * self.embed_dim, bias=True),
        )
        self.proj = nn.Linear(self.embed_dim, self.out_dim, bias=True)

        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B,D), cond: (B,cond_dim)
        shift, scale = self.modulation(cond).chunk(2, dim=1)
        return self.proj(self.norm(x) * (1 + scale) + shift)
