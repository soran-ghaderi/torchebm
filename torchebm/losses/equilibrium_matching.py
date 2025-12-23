r"""Equilibrium Matching (EqM) loss.

Implements equilibrium training objective that matches a time-invariant
velocity/score/noise target along a coupling path between x0~N(0,I) and x1~data.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Any, Union

import torch
from torch import nn

from torchebm.core.base_loss import BaseLoss
from torchebm.interpolants import expand_t_like_x
from torchebm.losses.loss_utils import (
    mean_flat,
    get_interpolant,
    compute_eqm_ct,
    dispersive_loss,
)


class EquilibriumMatchingLoss(BaseLoss):
    r"""
    Equilibrium Matching (EqM) training loss.

    Implements the flow matching objective with optional dispersive regularization
    for training generative models on unnormalized densities.

    Args:
        model: Neural network predicting velocity/score/noise.
        prediction: Network prediction type ('velocity', 'score', or 'noise').
        interpolant: Interpolant type ('linear', 'cosine', or 'vp').
        loss_weight: Loss weighting scheme ('velocity', 'likelihood', or None).
        train_eps: Epsilon for training time interval stability.
        apply_dispersion: Whether to apply dispersive regularization.
        dispersion_weight: Weight for dispersive loss term.
        dtype: Data type for computations.
        device: Device for computations.
        use_mixed_precision: Whether to use mixed precision.
        clip_value: Optional value to clamp the loss.

    Example:
        ```python
        from torchebm.losses import EquilibriumMatchingLoss
        import torch.nn as nn
        import torch

        model = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 2))
        loss_fn = EquilibriumMatchingLoss(
            model=model,
            prediction="velocity",
            interpolant="linear",
        )
        x = torch.randn(32, 2)
        loss = loss_fn(x)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        prediction: Literal["velocity", "score", "noise"] = "velocity",
        interpolant: Literal["linear", "cosine", "vp"] = "linear",
        loss_weight: Optional[Literal["velocity", "likelihood"]] = None,
        train_eps: float = 0.0,
        apply_dispersion: bool = False,
        dispersion_weight: float = 0.5,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        use_mixed_precision: bool = False,
        clip_value: Optional[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            dtype=dtype,
            device=device,
            use_mixed_precision=use_mixed_precision,
            clip_value=clip_value,
            *args,
            **kwargs,
        )
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.prediction = prediction
        self.loss_weight = loss_weight
        self.train_eps = train_eps
        self.apply_dispersion = apply_dispersion
        self.dispersion_weight = dispersion_weight
        self.interpolant = get_interpolant(interpolant)

    def _check_interval(self) -> tuple[float, float]:
        r"""Get training time interval respecting epsilon."""
        t0 = self.train_eps
        t1 = 1.0 - self.train_eps
        return t0, t1

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Compute EqM loss (nn.Module interface).

        Args:
            x: Data samples of shape (batch_size, ...).
            *args: Additional positional arguments.
            **kwargs: Additional model arguments.

        Returns:
            Scalar loss value.
        """
        if (x.device != self.device) or (x.dtype != self.dtype):
            x = x.to(device=self.device, dtype=self.dtype)

        with self.autocast_context():
            loss = self.compute_loss(x, *args, **kwargs)

        return loss

    def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Compute the equilibrium matching loss.

        Args:
            x: Data samples of shape (batch_size, ...).
            *args: Additional positional arguments.
            **kwargs: Additional model arguments passed to the network.

        Returns:
            Scalar loss value.
        """
        terms = self.training_losses(x, model_kwargs=kwargs)
        return terms["loss"].mean()

    def training_losses(
        self,
        x1: torch.Tensor,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Compute training losses with detailed outputs.

        This method provides the full loss dictionary including predictions,
        useful for logging and debugging.

        Args:
            x1: Data samples of shape (batch_size, ...).
            model_kwargs: Additional model arguments.

        Returns:
            Dictionary with 'loss' (per-sample) and 'pred' tensors.
        """
        if model_kwargs is None:
            model_kwargs = {}

        x1 = x1.to(device=self.device, dtype=self.dtype)
        batch = x1.shape[0]

        x0 = torch.randn_like(x1)
        t0, t1 = self._check_interval()
        t = torch.rand(batch, device=self.device, dtype=self.dtype) * (t1 - t0) + t0

        xt, ut = self.interpolant.interpolate(x0, x1, t)

        # Apply energy-compatible target scaling
        ct = compute_eqm_ct(t)
        ct = ct.view(batch, *([1] * (xt.ndim - 1)))
        ut = ut * ct

        with self.autocast_context():
            model_output = self.model(xt, t, **model_kwargs)

        if isinstance(model_output, tuple):
            model_output, act = model_output
        else:
            act = []

        disp_loss = 0.0
        if self.apply_dispersion and len(act) > 0:
             if isinstance(act, list):
                disp_loss = dispersive_loss(act[-1])
             else:
                 # Handle case where act might be a single tensor
                 disp_loss = dispersive_loss(act)

        terms = {"pred": model_output}

        if self.prediction == "velocity":
            terms["loss"] = mean_flat((model_output - ut) ** 2)
        else:
            t_expanded = expand_t_like_x(t, xt)
            _, drift_var = self.interpolant.compute_drift(xt, t)
            sigma_t, _ = self.interpolant.compute_sigma_t(t_expanded)

            if self.loss_weight == "velocity":
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_weight == "likelihood":
                weight = drift_var / (sigma_t**2)
            else:
                weight = 1.0

            if self.prediction == "noise":
                terms["loss"] = mean_flat(weight * (model_output - x0) ** 2)
            elif self.prediction == "score":
                terms["loss"] = mean_flat(weight * (model_output * sigma_t + x0) ** 2)
            else:
                raise ValueError(f"Unknown prediction type: {self.prediction}")

        # Add dispersive regularization
        if self.apply_dispersion:
            terms["loss"] = terms["loss"] + self.dispersion_weight * disp_loss

        return terms

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"prediction={self.prediction!r}, "
            f"interpolant={type(self.interpolant).__name__})"
        )


__all__ = ["EquilibriumMatchingLoss"]
