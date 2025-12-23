from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from torchebm.models.components import (
    AdaLNZeroBlock,
    AdaLNZeroPatchHead,
    ConvPatchEmbed2d,
    build_2d_sincos_pos_embed,
)


class ConditionalTransformer2D(nn.Module):
    """Generic conditional 2D Transformer backbone.

    This module is intentionally *loss-agnostic*.

    Inputs:
      - `x`: image-like tensor (B,C,H,W)
      - `cond`: conditioning vector (B, cond_dim)

    Output:
      - image-like tensor (B, out_channels, H, W)

    You can plug this into EqM, diffusion, score matching, etc. by choosing:
      - how `cond` is produced (time, labels, text, ...)
      - `out_channels` and head behavior
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        input_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        cond_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        use_sincos_pos_embed: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.cond_dim = int(cond_dim) if cond_dim is not None else int(embed_dim)

        self.patch_embed = ConvPatchEmbed2d(
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
        )

        num_patches_per_side = self.input_size // self.patch_size
        if num_patches_per_side * self.patch_size != self.input_size:
            raise ValueError("input_size must be divisible by patch_size")

        self.use_sincos_pos_embed = bool(use_sincos_pos_embed)
        if self.use_sincos_pos_embed:
            pe = build_2d_sincos_pos_embed(self.embed_dim, num_patches_per_side)
            self.register_buffer("pos_embed", pe.unsqueeze(0), persistent=False)  # (1,N,D)
        else:
            self.pos_embed = None

        self.blocks = nn.ModuleList(
            [
                AdaLNZeroBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    cond_dim=self.cond_dim,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(self.depth)
            ]
        )

        self.head = AdaLNZeroPatchHead(
            embed_dim=self.embed_dim,
            cond_dim=self.cond_dim,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)  # (B,N,D)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)

        for block in self.blocks:
            tokens = block(tokens, cond)

        return self.head(tokens, cond)
