from __future__ import annotations

import math
from typing import Tuple

import torch


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    # pos: (M,)
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    out = pos[:, None].float() * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def build_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create 2D sin/cos positional embeddings.

    Returns tensor with shape (grid_size*grid_size, embed_dim).
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    dev = device if device is not None else torch.device("cpu")
    grid_h = torch.arange(grid_size, device=dev, dtype=torch.float32)
    grid_w = torch.arange(grid_size, device=dev, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0).reshape(2, -1)  # (2, M)

    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = torch.cat([emb_h, emb_w], dim=1).to(dtype=dtype)
    return emb
