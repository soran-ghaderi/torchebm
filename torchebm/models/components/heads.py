from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .patch import unpatchify2d
from .transformer import modulate


class AdaLNZeroPatchHead(nn.Module):
    """Final layer that maps token features to patch pixels with adaLN-Zero."""

    def __init__(
        self,
        *,
        embed_dim: int,
        cond_dim: Optional[int] = None,
        patch_size: int,
        out_channels: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.cond_dim = int(cond_dim) if cond_dim is not None else int(embed_dim)
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)

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
        return unpatchify2d(patches, self.patch_size, out_channels=self.out_channels)
