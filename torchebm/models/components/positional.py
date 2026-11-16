from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    # pos: (M,). Computed in float64 like the reference MAE/DiT tables, so the
    # float32 result is bit-identical to theirs; the caller casts at the end.
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float64)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    out = pos[:, None].double() * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def build_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int | Tuple[int, int],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create 2D sin/cos positional embeddings.

    `grid_size` is an int for square grids or an `(gh, gw)` tuple. Returns a
    tensor of shape (gh*gw, embed_dim) in row-major token order.
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    if isinstance(grid_size, int):
        gh = gw = int(grid_size)
    else:
        gh, gw = (int(s) for s in grid_size)

    dev = device if device is not None else torch.device("cpu")
    grid_h = torch.arange(gh, device=dev, dtype=torch.float32)
    grid_w = torch.arange(gw, device=dev, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0).reshape(2, -1)  # (2, M)

    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = torch.cat([emb_h, emb_w], dim=1).to(dtype=dtype)
    return emb


class RotaryEmbedding2d(nn.Module):
    r"""2D rotary position embedding (RoPE) of image tokens.

    The 2D RoPE of EVA-02 (Fang et al., 2023) that LightningDiT (Yao et al.,
    2025) applies to queries and keys. The first half of the head dimension
    rotates with the row \(h\) of a token, the second half with its column
    \(w\); within each half, the pair \((2i, 2i + 1)\) rotates by the angle
    \(p\, \theta^{-4i/D}\) for position \(p\) and head dimension \(D\):

    \[
    x'_{2i} = x_{2i} \cos(p\,\omega_i) - x_{2i+1} \sin(p\,\omega_i), \qquad
    x'_{2i+1} = x_{2i+1} \cos(p\,\omega_i) + x_{2i} \sin(p\,\omega_i).
    \]

    The tables are built once in float32 as in the reference code
    (github.com/hustvl/LightningDiT, ``VisionRotaryEmbeddingFast``) and
    stored as non-persistent buffers: ``cos``, and ``sin`` with the sign of
    the rotation folded in (\(-\sin\) on even, \(+\sin\) on odd entries), so
    the rotation is one pair swap and one ``addcmul``.

    Args:
        head_dim: Head dimension \(D\) of the attention, a multiple of 4.
        grid_size: Token grid, an int for square grids or ``(gh, gw)``, in the
            row-major token order of `ConvPatchEmbed2d`.
        theta: Base \(\theta\) of the frequencies.

    Example:
        ```python
        import torch
        from torchebm.models.components import RotaryEmbedding2d

        rope = RotaryEmbedding2d(head_dim=64, grid_size=16)
        q = rope(torch.randn(2, 16, 256, 64))
        ```
    """

    def __init__(self, head_dim: int, grid_size: int | Tuple[int, int], theta: float = 10000.0):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim must be a multiple of 4 for 2D RoPE, got {head_dim}")
        gh, gw = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
        dim = head_dim // 2
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        angles_h = (torch.arange(gh).float()[:, None] * freqs).repeat_interleave(2, dim=-1)
        angles_w = (torch.arange(gw).float()[:, None] * freqs).repeat_interleave(2, dim=-1)
        angles = torch.cat(
            [angles_h[:, None].expand(gh, gw, dim), angles_w[None].expand(gh, gw, dim)], dim=-1
        ).reshape(gh * gw, head_dim)
        sign = torch.tensor([-1.0, 1.0]).repeat(head_dim // 2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin() * sign, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Rotate ``x`` of shape ``(..., gh * gw, head_dim)``."""
        swapped = x.unflatten(-1, (-1, 2)).flip(-1).flatten(-2)
        return torch.addcmul(x * self.cos, swapped, self.sin)
