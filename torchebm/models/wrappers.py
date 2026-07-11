from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from torchebm.core import BaseModel, BaseScheduler, Schedulable


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


class InteractionModel(Schedulable, BaseModel):
    r"""Potential with pairwise repulsive interaction for diverse sampling.

    Wraps a scalar potential \(V(x)\) with the Energy Matching interaction
    energy (Balcerak et al., 2025, arXiv:2504.10612):

    \[
    E_i = V(x_i) - W_i, \qquad
    W_i = \frac{1}{2} \frac{s}{\sigma_W^2} \sum_{j} \|x_i - x_j\|^2
    \]

    so that \(\sum_i W_i = \tfrac{1}{2} \tfrac{s}{\sigma_W^2}
    \sum_{i \neq j} \|x_i - x_j\|^2\). Because `BaseModel.gradient()`
    differentiates the summed batch energy, sampling with this model yields
    the batch-coupled repulsive Langevin drift of the paper: samples are
    pushed apart, increasing diversity (used for inverse design and
    conditional generation).

    The strength \(s\) is a `Schedulable` parameter. Pass a
    `TemperatureScheduler(..., sqrt=False)` to reproduce the paper's
    \(\epsilon(t)\)-scaled interaction: the wrapper sits inside the sampler's
    module subtree, so `LangevinDynamics.sample()` resets and advances it in
    lockstep with the noise schedule.

    Note:
        The squared distances use the expansion
        \(\sum_j \|x_i - x_j\|^2 = B \|x_i\|^2 + \sum_j \|x_j\|^2
        - 2 x_i \cdot \sum_j x_j\), which is exact, O(batch x dim), and
        differentiable everywhere (`torch.cdist` has a NaN derivative on the
        zero diagonal).

    Note:
        The repulsive drift on each sample scales as
        \(2 s B / \sigma_W^2 \cdot (x_i - \bar{x})\) for batch size \(B\).
        Keep \(2 s B \Delta t / \sigma_W^2 \ll 1\) or the chains expand
        exponentially; for \(s = 0.15\), \(B = 64\), \(\Delta t = 0.01\)
        this means \(\sigma_W \gtrsim 4\).

    Args:
        model: Base potential \(V(x)\) returning shape (batch_size,).
        sigma_w: Interaction bandwidth \(\sigma_W\). Must be positive.
        strength: Interaction strength \(s\), typically the temperature
            \(\epsilon(t)\). Float or `BaseScheduler`.

    Example:
        ```python
        from torchebm.core import TemperatureScheduler
        from torchebm.models import InteractionModel
        from torchebm.samplers import LangevinDynamics

        temp_noise = TemperatureScheduler(0.15, 0.8, n_steps=325, t_end=3.25)
        temp_strength = TemperatureScheduler(
            0.15, 0.8, n_steps=325, t_end=3.25, sqrt=False
        )
        repulsive = InteractionModel(V, sigma_w=4.0, strength=temp_strength)
        sampler = LangevinDynamics(
            model=repulsive, step_size=0.01, noise_scale=temp_noise
        )
        samples = sampler.sample(x=torch.randn(64, 2), n_steps=325)
        ```
    """

    def __init__(
        self,
        model: BaseModel,
        sigma_w: float,
        strength: Union[float, BaseScheduler] = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if sigma_w <= 0:
            raise ValueError(f"sigma_w must be positive, got {sigma_w}")
        self.model = model
        self.sigma_w = float(sigma_w)
        self._register_param("strength", strength)

    @property
    def strength(self) -> float:
        return self.get_scheduled_value("strength")

    @strength.setter
    def strength(self, value: Union[float, BaseScheduler]) -> None:
        self._register_param("strength", value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the interacting energy \(V(x_i) - W_i\) per sample."""
        batch = x.shape[0]
        flat = x.reshape(batch, -1)
        sq_norms = flat.square().sum(dim=1)
        pair_sq = batch * sq_norms + sq_norms.sum() - 2.0 * flat @ flat.sum(dim=0)
        w = 0.5 * (self.strength / self.sigma_w**2) * pair_sq
        return self.model(x) - w

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={type(self.model).__name__}, sigma_w={self.sigma_w})"
        )
