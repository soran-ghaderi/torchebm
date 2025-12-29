r"""Equilibrium Matching (EqM) loss.

Implements time-invariant equilibrium training objectives for learning energy
landscapes, following the EqM paper:

- **Implicit EqM** ($L_{EqM}$): Learns gradient field directly
  
  \[
  L_{EqM} = \|f(x_\gamma) - (\epsilon - x) \cdot c(\gamma)\|^2
  \]

- **Explicit EqM-E** ($L_{EqM-E}$): Learns scalar energy via gradient matching

  \[
  L_{EqM-E} = \|\nabla g(x_\gamma) - (\epsilon - x) \cdot c(\gamma)\|^2
  \]

Key differences from Flow Matching:
- Time-invariant: Model zeros out time conditioning internally
- Gradient direction: EqM learns (noise - data), FM learns (data - noise)
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
    r"""Equilibrium Matching (EqM) training loss.

    Implements gradient matching for learning equilibrium energy landscapes.
    Supports both implicit (vector field) and explicit (energy-based) formulations,
    with multiple prediction types and loss weighting schemes.

    The target gradient is $(\epsilon - x) \cdot c(\gamma)$ where:
    - $\epsilon$ is noise (x0)
    - $x$ is data (x1)  
    - $c(\gamma) = \lambda \cdot \min(1, (1-\gamma)/(1-a))$ is truncated decay

    Args:
        model: Neural network predicting velocity/score/noise.
        prediction: Network prediction type ('velocity', 'score', or 'noise').
        energy_type: Energy formulation type:
            - 'none': Implicit EqM, model predicts gradient directly
            - 'dot': $g(x) = x \cdot f(x)$, dot product energy formulation
            - 'l2': $g(x) = -\frac{1}{2}\|f(x)\|^2$ (experimental)
            - 'mean': Same as dot (alias)
        interpolant: Interpolant type ('linear', 'cosine', or 'vp').
        loss_weight: Loss weighting scheme ('velocity', 'likelihood', or None).
        train_eps: Epsilon for training time interval stability.
        ct_threshold: Decay threshold $a$ for $c(t)$. Decay starts after $t > a$. Default: 0.8.
        ct_multiplier: Gradient multiplier $\lambda$ for $c(t)$. Default: 4.0.
        apply_dispersion: Whether to apply dispersive regularization.
        dispersion_weight: Weight for dispersive loss term.
        time_invariant: If True, pass zeros for time to model (EqM default).
        dtype: Data type for computations.
        device: Device for computations.
        use_mixed_precision: Whether to use mixed precision.
        clip_value: Optional value to clamp the loss.

    Example:
        ```python
        from torchebm.losses import EquilibriumMatchingLoss
        import torch.nn as nn
        import torch

        # Implicit EqM with velocity prediction (default)
        model = MyTimeConditionedModel()
        loss_fn = EquilibriumMatchingLoss(
            model=model,
            prediction="velocity",
            energy_type="none",
        )

        # Explicit EqM-E with dot product (for OOD detection)
        loss_fn_explicit = EquilibriumMatchingLoss(
            model=model,
            prediction="velocity",
            energy_type="dot",
        )

        x = torch.randn(32, 2)
        loss = loss_fn(x)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        prediction: Literal["velocity", "score", "noise"] = "velocity",
        energy_type: Literal["none", "dot", "l2", "mean"] = "none",
        interpolant: Literal["linear", "cosine", "vp"] = "linear",
        loss_weight: Optional[Literal["velocity", "likelihood"]] = None,
        train_eps: float = 0.0,
        ct_threshold: float = 0.8,
        ct_multiplier: float = 4.0,
        apply_dispersion: bool = False,
        dispersion_weight: float = 0.5,
        time_invariant: bool = True,
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
        self.energy_type = energy_type
        self.loss_weight = loss_weight
        self.train_eps = train_eps
        self.ct_threshold = ct_threshold
        self.ct_multiplier = ct_multiplier
        self.apply_dispersion = apply_dispersion
        self.dispersion_weight = dispersion_weight
        self.time_invariant = time_invariant
        self.interpolant = get_interpolant(interpolant)

    def _check_interval(self) -> tuple[float, float]:
        r"""Get training time interval respecting epsilon."""
        t0 = self.train_eps
        t1 = 1.0 - self.train_eps
        return t0, t1

    def _compute_explicit_energy_gradient(
        self,
        xt: torch.Tensor,
        model_output: torch.Tensor,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute explicit energy and its gradient.

        Args:
            xt: Interpolated samples with requires_grad=True.
            model_output: Raw model output (vector field).
            training: Whether to create computation graph.

        Returns:
            Tuple of (gradient field, energy scalar per sample).
        """
        if self.energy_type == "dot" or self.energy_type == "mean":
            # g(x) = x · f(x)
            energy = (xt * model_output).sum(dim=tuple(range(1, xt.ndim)))
        elif self.energy_type == "l2":
            # g(x) = -0.5 ||f(x)||^2
            energy = -0.5 * (model_output**2).sum(dim=tuple(range(1, model_output.ndim)))
        else:
            raise ValueError(f"Unknown energy type: {self.energy_type}")

        # Compute gradient of energy w.r.t. input
        if xt.requires_grad:
            grad = torch.autograd.grad(
                energy.sum(),
                xt,
                create_graph=training,
            )[0]
        else:
            grad = model_output  # Fallback if no grad required

        return grad, energy

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
        r"""Compute training losses with detailed outputs.

        Implements gradient matching with EqM target direction:
        - Target: $(\epsilon - x) \cdot c(\gamma)$ (noise toward data)
        - Time-invariant: zeros out time if time_invariant=True

        Args:
            x1: Data samples of shape (batch_size, ...).
            model_kwargs: Additional model arguments.

        Returns:
            Dictionary with 'loss' (per-sample), 'pred', and optionally 'energy'.
        """
        if model_kwargs is None:
            model_kwargs = {}

        x1 = x1.to(device=self.device, dtype=self.dtype)
        batch = x1.shape[0]

        # Sample noise and time
        x0 = torch.randn_like(x1)
        t0, t1 = self._check_interval()
        t = torch.rand(batch, device=self.device, dtype=self.dtype) * (t1 - t0) + t0

        # Interpolate: xt between x0 (noise) and x1 (data)
        xt, ut = self.interpolant.interpolate(x0, x1, t)

        # EqM target: (noise - data) * c(t), opposite of Flow Matching
        ct = compute_eqm_ct(t, threshold=self.ct_threshold, multiplier=self.ct_multiplier)
        ct = ct.view(batch, *([1] * (xt.ndim - 1)))
        target = (x0 - x1) * ct  # Gradient direction: noise - data

        # For explicit energy, we need gradients w.r.t. xt
        if self.energy_type != "none":
            xt = xt.detach().requires_grad_(True)

        # EqM: zero out time for time-invariance (model still receives t for API compat)
        t_model = torch.zeros_like(t) if self.time_invariant else t

        with self.autocast_context():
            model_output = self.model(xt, t_model, **model_kwargs)

        if isinstance(model_output, tuple):
            model_output, act = model_output
        else:
            act = []

        # Compute dispersive loss if enabled
        disp_loss = 0.0
        if self.apply_dispersion and len(act) > 0:
            if isinstance(act, list):
                disp_loss = dispersive_loss(act[-1])
            else:
                disp_loss = dispersive_loss(act)

        terms = {"pred": model_output}

        # Compute loss based on prediction type
        if self.prediction == "velocity":
            if self.energy_type == "none":
                # Implicit EqM: model directly predicts gradient field
                terms["loss"] = mean_flat((model_output - target) ** 2)
            else:
                # Explicit EqM-E: compute gradient of energy function
                grad, energy = self._compute_explicit_energy_gradient(
                    xt, model_output, training=self.model.training
                )
                terms["loss"] = mean_flat((grad - target) ** 2)
                terms["energy"] = energy
        else:
            # Score or noise prediction with optional weighting
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
            f"energy_type={self.energy_type!r}, "
            f"interpolant={type(self.interpolant).__name__})"
        )


__all__ = ["EquilibriumMatchingLoss"]