r"""Flow Matching (FM) loss.

Standard conditional flow matching: regress a time-conditioned velocity field
onto the interpolant velocity,

\[
L_{FM} = \|v_\theta(x_t, t) - u_t\|^2, \qquad
x_t, u_t = \mathrm{interpolant}(x_0, x_1, t)
\]

with \(x_0\) noise (or any source batch) and \(x_1\) data. Sampling integrates
the learned velocity forward with `FlowSampler` (no ``negate_velocity``).

Relation to Equilibrium Matching: with ``ct="constant"`` and
``ct_multiplier=1`` the EqM objective is exactly this loss with the field
negated and the time conditioning zeroed.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Optional, Union

import torch
from torch import nn

from torchebm.core import BaseCoupling, BaseInterpolant, BaseScheduler
from torchebm.core.base_loss import BaseInterpolantLoss
from torchebm.losses import mean_flat


class FlowMatchingLoss(BaseInterpolantLoss):
    r"""Conditional Flow Matching training loss.

    Trains ``model(x_t, t, **model_kwargs)`` to predict the interpolant
    velocity \(u_t\). The model receives the true training time (unlike
    `EquilibriumMatchingLoss`, whose field is time-invariant).

    Args:
        model: Neural network predicting velocity, called as ``model(x, t)``.
        interpolant: Interpolant name (e.g. 'linear', 'cosine', 'vp') or
            BaseInterpolant instance.
        coupling: Minibatch coupling: a name ('independent' (default,
            identity), 'ot'/'exact_ot', 'sinkhorn', ...) or a BaseCoupling
            instance. Pairs the source and target batches before
            interpolation; per-pair weights are honored in the reduction.
        train_eps: Epsilon for training time interval stability.
        t_sampler: Training-time distribution: 'uniform' (default),
            'lognormal' (EDM timestep skew), or a callable
            ``(batch, *, device, dtype, generator) -> t``.
        t_p_mean: Lognormal skew location $p_{mean}$. Default: -1.2.
        t_p_std: Lognormal skew scale $p_{std}$, positive. Default: 1.2.
        loss_weight_fn: Optional per-timestep weight hook ``t -> w(t)``
            multiplied into the per-sample loss.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.losses import FlowMatchingLoss
        from torchebm.samplers import FlowSampler

        loss_fn = FlowMatchingLoss(model=velocity_net, interpolant="linear")
        loss = loss_fn(x_batch)
        # ... train, then:
        sampler = FlowSampler(velocity_net, interpolant="linear")
        samples = sampler.sample(n_samples=64, dim=2, n_steps=50)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        interpolant: Union[str, BaseInterpolant] = "linear",
        coupling: Union[str, BaseCoupling, None] = None,
        train_eps: Union[float, BaseScheduler] = 0.0,
        t_sampler: Union[
            Literal["uniform", "lognormal"],
            Callable[..., torch.Tensor],
        ] = "uniform",
        t_p_mean: float = -1.2,
        t_p_std: float = 1.2,
        loss_weight_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            interpolant=interpolant,
            coupling=coupling,
            train_eps=train_eps,
            t_sampler=t_sampler,
            t_p_mean=t_p_mean,
            t_p_std=t_p_std,
            loss_weight_fn=loss_weight_fn,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )
        self.model = model

    def _probe_forward(self, px: torch.Tensor, pmk: dict) -> torch.Tensor:
        r"""Field convention for the conditioning probe: fixed mid-path time."""
        t = torch.full((px.shape[0],), 0.5, device=px.device, dtype=px.dtype)
        return self.model(px, t, **pmk)

    def forward(
        self,
        x: torch.Tensor,
        *args,
        y: Optional[torch.Tensor] = None,
        x0: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Compute the FM loss (nn.Module interface).

        Args:
            x: Data samples of shape (batch_size, ...).
            *args: Additional positional arguments.
            y: Optional conditioning tensor forwarded to the model; shorthand
                for ``model_kwargs={'y': y}``.
            x0: Optional source samples of shape (batch_size, ...). Defaults to
                standard Gaussian noise.
            model_kwargs: Conditioning arguments forwarded to the model.
            **kwargs: Deprecated bare model kwargs; pass ``model_kwargs={...}``.

        Returns:
            Scalar loss value.
        """
        model_kwargs = self._merge_condition(model_kwargs, y)
        if (x.device != self.device) or (x.dtype != self.dtype):
            x = x.to(device=self.device, dtype=self.dtype)

        with self.autocast_context():
            loss = self.compute_loss(
                x, *args, x0=x0, model_kwargs=model_kwargs, generator=generator, **kwargs
            )
        return loss

    def training_losses(
        self,
        x1: torch.Tensor,
        model_kwargs: Optional[Dict[str, Any]] = None,
        x0: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Compute training losses with detailed outputs.

        Args:
            x1: Data samples of shape (batch_size, ...).
            model_kwargs: Conditioning forwarded to the model.
            x0: Optional source samples; standard Gaussian noise when None.
                Paired against ``x1`` by the configured coupling.
            generator: RNG for the source draw, the coupling and the time
                sampling; the global RNG when ``None``.

        Returns:
            Dictionary with 'loss' (per-sample), 'pred', and 'weights'
            (per-pair coupling weights or None).
        """
        if model_kwargs is None:
            model_kwargs = {}

        x1 = x1.to(device=self.device, dtype=self.dtype)
        batch = x1.shape[0]

        if x0 is None:
            x0 = torch.randn_like(x1, generator=generator)
        else:
            x0 = x0.to(device=self.device, dtype=self.dtype)
            if x0.shape != x1.shape:
                raise ValueError(
                    f"x0 shape {tuple(x0.shape)} must match x1 shape {tuple(x1.shape)}"
                )

        coupled = self.coupling(x0, x1, generator=generator, **model_kwargs)
        x0, x1 = coupled

        t = self._sample_t(batch, generator)
        xt, ut = self.interpolant.interpolate(x0, x1, t)

        with self.autocast_context():
            pred = self.model(xt, t, **model_kwargs)
        if isinstance(pred, tuple):
            pred = pred[0]

        loss = mean_flat((pred - ut).square())
        if self.loss_weight_fn is not None:
            loss = loss * self.loss_weight_fn(t)

        return {"loss": loss, "pred": pred, "weights": coupled.weights}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"interpolant={type(self.interpolant).__name__}, "
            f"coupling={type(self.coupling).__name__}, "
            f"t_sampler={self.t_sampler!r})"
        )


__all__ = ["FlowMatchingLoss"]
