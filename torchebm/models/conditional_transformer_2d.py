from __future__ import annotations

from typing import Optional

import torch

from torchebm._deprecation import deprecated
from torchebm.models.dit import DiT


@deprecated(
    since="0.8.4",
    deprecated_on="2026-08-30",
    replacement="torchebm.models.DiT",
    message=(
        "ConditionalTransformer2D is deprecated; use torchebm.models.DiT "
        "(forward(x, cond=...) reproduces this model exactly)"
    ),
    removal=(
        "torchebm/models/conditional_transformer_2d.py, its export in "
        "torchebm/models/__init__.py, "
        "tests/models/test_conditional_transformer_2d.py, and its entry in "
        "tests/models/test_constructor_kwargs.py"
    ),
)
class ConditionalTransformer2D(DiT):
    """Deprecated: use `DiT` instead.

    ``DiT(..., pos_embed="sincos" | None).forward(x, cond=...)`` reproduces
    this model exactly. This shim keeps the original constructor and
    ``forward(x, cond)`` contract while emitting a `DeprecationWarning`.

    State dicts saved from previous versions load with ``strict=False``: the
    shim carries `DiT`'s default timestep embedder, which this class never
    uses.
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
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            cond_dim=cond_dim,
            mlp_ratio=mlp_ratio,
            pos_embed="sincos" if use_sincos_pos_embed else None,
        )
        self.use_sincos_pos_embed = bool(use_sincos_pos_embed)

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        *,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Run the backbone with a single conditioning vector.

        The conditioning signal may be passed positionally (``forward(x,
        cond)``) or by keyword as ``cond=`` or ``t=``; either way it is used
        as the raw adaLN vector, matching the original contract (``t`` is
        *not* embedded).
        """
        c = cond if cond is not None else t
        if c is None:
            raise ValueError(
                "ConditionalTransformer2D.forward requires a conditioning tensor "
                "via `cond` (positional) or the `cond=`/`t=` keyword."
            )
        return super().forward(x, cond=c)
