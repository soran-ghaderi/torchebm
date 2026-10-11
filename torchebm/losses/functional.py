r"""Functional building blocks for matching losses."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch

from torchebm.core import BaseCoupling, BaseInterpolant
from torchebm.losses.loss_utils import mean_flat


@dataclass(frozen=True)
class MatchingBatch:
    r"""Prepared tensors for an externally evaluated matching objective.

    Attributes:
        xt: Interpolated model input.
        t: Sampled interpolation time.
        model_time: Time passed to the model. This equals ``t`` for flow
            matching; objectives with a time-invariant field may replace it.
        target: Tensor prediction target.
        weights: Optional per-pair coupling weights.
    """

    xt: torch.Tensor
    t: torch.Tensor
    model_time: torch.Tensor
    target: torch.Tensor
    weights: Optional[torch.Tensor] = None


def prepare_flow_matching(
    x1: torch.Tensor,
    *,
    interpolant: BaseInterpolant,
    coupling: BaseCoupling,
    x0: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    model_kwargs: Optional[Mapping[str, Any]] = None,
    t_sampler: Optional[Callable[..., torch.Tensor]] = None,
    negate_velocity: bool = False,
) -> MatchingBatch:
    r"""Prepare a conditional flow-matching batch without evaluating a model.

    Random draws retain the class API's order: source noise first, stochastic
    coupling second, and interpolation time last. Passing ``x0`` or ``t``
    skips the corresponding draw. When ``t`` is omitted, ``t_sampler`` is used
    if provided; otherwise time is sampled uniformly on ``[0, 1)``.

    Args:
        x1: Target samples of shape ``(batch, ...)``.
        interpolant: Interpolant used to construct ``xt`` and its velocity.
        coupling: Source-target coupling.
        x0: Optional source samples. Defaults to standard Gaussian noise.
        t: Optional explicit interpolation times of shape ``(batch,)``.
        generator: Generator shared by source, coupling, and time draws.
        model_kwargs: Optional conditioning forwarded to the coupling.
        t_sampler: Optional time sampler called after coupling as
            ``t_sampler(batch, device=, dtype=, generator=)``, the same
            convention as the loss classes' ``t_sampler``. When omitted, times
            are drawn uniformly on ``[0, 1)``.
        negate_velocity: Use ``-ut`` rather than ``ut`` as the target.

    Returns:
        Prepared matching tensors and optional coupling weights.
    """
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    batch = x1.shape[0]

    if x0 is None:
        x0 = torch.randn_like(x1, generator=generator)
    else:
        x0 = x0.to(device=x1.device, dtype=x1.dtype)
        if x0.shape != x1.shape:
            raise ValueError(
                f"x0 shape {tuple(x0.shape)} must match x1 shape {tuple(x1.shape)}"
            )

    coupled = coupling(x0, x1, generator=generator, **model_kwargs)
    x0, x1 = coupled

    if t is None:
        if t_sampler is None:
            t = torch.rand(
                batch,
                device=x1.device,
                dtype=x1.dtype,
                generator=generator,
            )
        else:
            t = t_sampler(batch, device=x1.device, dtype=x1.dtype, generator=generator)
    else:
        t = t.to(device=x1.device, dtype=x1.dtype)

    if t.shape != (batch,):
        raise ValueError(f"t shape {tuple(t.shape)} must be ({batch},)")

    xt, ut = interpolant.interpolate(x0, x1, t)
    target = -ut if negate_velocity else ut
    return MatchingBatch(
        xt=xt,
        t=t,
        model_time=t,
        target=target,
        weights=coupled.weights,
    )


def weighted_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
    reduction: Literal["none", "mean"] = "mean",
) -> torch.Tensor:
    r"""Compute per-sample MSE with optional objective and coupling weights.

    ``loss_weights`` multiply individual per-sample losses. ``weights`` are
    coupling masses and therefore also define the denominator of the weighted
    mean, matching the class API's existing reduction.

    Args:
        prediction: Model prediction with a leading batch dimension.
        target: Prediction target, with the same shape as ``prediction``.
        weights: Optional per-sample coupling masses. These affect both the
            numerator and denominator when ``reduction="mean"``.
        loss_weights: Optional per-sample objective multipliers. These affect
            the numerator only.
        reduction: ``"none"`` returns the per-sample losses after applying
            ``loss_weights``; ``"mean"`` returns their plain or
            coupling-weighted mean.

    Returns:
        A per-sample tensor for ``reduction="none"`` or a scalar mean for
        ``reduction="mean"``.

    Raises:
        ValueError: If ``reduction`` is not ``"none"`` or ``"mean"``.
    """
    loss = mean_flat((prediction - target).square())
    if loss_weights is not None:
        loss = loss * loss_weights

    if reduction == "none":
        return loss
    if reduction != "mean":
        raise ValueError(f"Unknown reduction: {reduction}")
    if weights is not None:
        return (weights * loss).sum() / weights.sum().clamp_min(1e-12)
    return loss.mean()


__all__ = ["MatchingBatch", "prepare_flow_matching", "weighted_mse_loss"]
