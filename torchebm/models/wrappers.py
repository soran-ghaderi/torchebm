from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LabelClassifierFreeGuidance(nn.Module):
    """Classifier-free guidance wrapper for label-conditioned models.

    This wrapper is intentionally small and generic:
    - assumes the base model accepts `y` (labels) and supports a *null label id*
    - performs two forward passes (cond and uncond)
    - applies guidance to the first `guide_channels` channels by default

    It does **not** assume a specific loss (EqM/diffusion/etc).

    Expected base signature:
      `base(x, t, y=..., **kwargs) -> Tensor[B,C,H,W]`

    You can use it with `FlowSampler` by wrapping your model instance.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        null_label_id: int,
        cfg_scale: float = 1.0,
        guide_channels: int = 3,
    ):
        super().__init__()
        self.base = base
        self.null_label_id = int(null_label_id)
        self.cfg_scale = float(cfg_scale)
        self.guide_channels = int(guide_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor, *, y: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.cfg_scale <= 1.0:
            return self.base(x, t, y=y, **kwargs)

        y_null = torch.full_like(y, fill_value=self.null_label_id)

        cond = self.base(x, t, y=y, **kwargs)
        uncond = self.base(x, t, y=y_null, **kwargs)

        c = min(self.guide_channels, cond.shape[1])
        guided = uncond[:, :c] + self.cfg_scale * (cond[:, :c] - uncond[:, :c])

        if c == cond.shape[1]:
            return guided
        return torch.cat([guided, uncond[:, c:]], dim=1)
