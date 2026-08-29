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

where $\epsilon$ is noise (x0), $x$ is data (x1), and the target $(\epsilon - x)$
points from data toward noise (opposite of FM velocity).

Key differences from Flow Matching:

- Time-invariant: Model zeros out time conditioning internally
- Gradient direction: EqM learns $(\epsilon - x)$, FM learns $(x - \epsilon)$
- Sampling: Use ``negate_velocity=True`` with FlowSampler for ODE sampling
"""

from __future__ import annotations

from typing import Callable, Dict, Literal, Optional, Any, Union

import torch
from torch import nn

from torchebm.core import (
    BaseCoupling,
    BaseLoss,
    BaseInterpolant,
    BaseScheduler,
    expand_t_like_x,
)
from torchebm.core.base_loss import _has_dtensor_params
from torchebm.couplings import resolve_coupling
from torchebm.interpolants import resolve_interpolant
from torchebm.losses import (
    mean_flat,
    compute_eqm_ct,
    dispersive_loss,
)


class EquilibriumMatchingLoss(BaseLoss):
    r"""Equilibrium Matching (EqM) training loss.

    Implements gradient matching for learning equilibrium energy landscapes.
    Supports both implicit (vector field) and explicit (energy-based) formulations,
    with multiple prediction types and loss weighting schemes.

    The target is $(\epsilon - x) \cdot c(\gamma)$ where:
    - $\epsilon$ is noise (x0), $x$ is data (x1)
    - For linear interpolant: target is $(x_0 - x_1) \cdot c(t)$ (noise - data)
    - $c(\gamma) = \lambda \cdot \min(1, (1-\gamma)/(1-a))$ is truncated decay

    For ODE sampling, use ``negate_velocity=True`` in FlowSampler since
    velocity $v = -f(x) = x - \epsilon$.

    Args:
        model: Neural network predicting velocity/score/noise.
        prediction: Network prediction type ('velocity', 'score', or 'noise').
        energy_type: Energy formulation type:
            - 'none': Implicit EqM, model predicts gradient directly
            - 'dot': $g(x) = x \cdot f(x)$, dot product energy formulation
            - 'l2': $g(x) = -\frac{1}{2}\|f(x)\|^2$ (experimental)
            - 'mean': Same as dot (alias)
        interpolant: Interpolant name (e.g. 'linear', 'cosine', 'vp') or BaseInterpolant instance.
        coupling: Minibatch coupling: a name ('independent' (default, identity),
            'ot'/'exact_ot', 'sinkhorn', ...) or a BaseCoupling instance. Pairs
            the source and target batches before interpolation.
        loss_weight: Loss weighting scheme ('velocity', 'likelihood', or None).
        train_eps: Epsilon for training time interval stability.
        t_sampler: Training-time distribution:
            - 'uniform' (default): uniform over the training interval
            - 'lognormal': EDM timestep skew, $\sigma = e^{z p_{std} + p_{mean}}$
              with $z \sim \mathcal{N}(0, 1)$ and $t = 1/(1+\sigma)$ clamped to
              [1e-4, 1] (intersected with the ``train_eps`` interval)
            - a callable ``(batch, *, device, dtype, generator) -> t`` returning
              shape (batch_size,)
        t_p_mean: Lognormal skew location $p_{mean}$ (EDM $P_{mean}$). Default: -1.2.
        t_p_std: Lognormal skew scale $p_{std}$ (EDM $P_{std}$), positive. Default: 1.2.
        loss_weight_fn: Optional per-timestep weight hook ``t -> w(t)`` (shape
            (batch_size,)), multiplied into the per-sample loss before the
            dispersion term; the mechanism behind min-SNR / EDM
            $\lambda(\sigma)$ style weightings. None (default) keeps the loss
            unweighted.
        ct: Weight family for the target scaling $c(t)$, always multiplied by
            ``ct_multiplier``:
            - 'truncated' (default): $\min(1, (1-t)/(1-a))$, the EqM truncated decay
            - 'linear': $1 - t$, the $a \to 0$ endpoint of the truncated dial
            - 'constant': $1$, the $a \to 1$ endpoint; with ``ct_multiplier=1``
              this is exactly the negated Flow Matching objective
            - a callable ``t -> c(t)`` mapping a (batch_size,) time tensor to
              weights of the same shape
        ct_threshold: Decay threshold $a$ for ct='truncated', strictly inside
            (0, 1); the endpoints are the 'linear' and 'constant' variants.
            Decay starts after $t > a$. Default: 0.8.
        ct_multiplier: Gradient multiplier $\lambda$ applied to every ct
            variant. Samplers that rescale velocities divide it back out, so it
            is recorded on the loss (attribute and ``repr``). Default: 4.0.
        apply_dispersion: Whether to apply dispersive regularization.
        dispersion_weight: Weight for dispersive loss term.
        dtype: Data type for computations.
        device: Device for computations.

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
        interpolant: Union[str, BaseInterpolant] = "linear",
        coupling: Union[str, BaseCoupling, None] = None,
        loss_weight: Optional[Literal["velocity", "likelihood"]] = None,
        train_eps: Union[float, BaseScheduler] = 0.0,
        t_sampler: Union[
            Literal["uniform", "lognormal"],
            Callable[..., torch.Tensor],
        ] = "uniform",
        t_p_mean: float = -1.2,
        t_p_std: float = 1.2,
        loss_weight_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        ct: Union[
            Literal["truncated", "linear", "constant"],
            Callable[[torch.Tensor], torch.Tensor],
        ] = "truncated",
        ct_threshold: float = 0.8,
        ct_multiplier: float = 4.0,
        apply_dispersion: bool = False,
        dispersion_weight: float = 0.5,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )
        if not callable(t_sampler) and t_sampler not in ("uniform", "lognormal"):
            raise ValueError(
                "t_sampler must be 'uniform', 'lognormal', or a callable "
                f"(batch, *, device, dtype, generator) -> t, got {t_sampler!r}"
            )
        if t_p_std <= 0:
            raise ValueError(f"t_p_std must be positive, got {t_p_std}")
        if loss_weight_fn is not None and not callable(loss_weight_fn):
            raise TypeError(
                "loss_weight_fn must be callable or None, got "
                f"{type(loss_weight_fn).__name__}"
            )
        if not callable(ct) and ct not in ("truncated", "linear", "constant"):
            raise ValueError(
                "ct must be 'truncated', 'linear', 'constant', or a callable "
                f"t -> c(t), got {ct!r}"
            )
        if ct == "truncated" and not 0.0 < ct_threshold < 1.0:
            raise ValueError(
                f"ct_threshold must be in (0, 1) for ct='truncated', got "
                f"{ct_threshold}; use ct='linear' for the threshold -> 0 "
                "endpoint or ct='constant' for the threshold -> 1 endpoint"
            )
        self.model = model
        self.prediction = prediction
        self.energy_type = energy_type
        self.loss_weight = loss_weight
        self.t_sampler = t_sampler
        self.t_p_mean = t_p_mean
        self.t_p_std = t_p_std
        self.loss_weight_fn = loss_weight_fn
        self._register_param("train_eps", train_eps)
        self.ct = ct
        self.ct_threshold = ct_threshold
        self.ct_multiplier = ct_multiplier
        self.apply_dispersion = apply_dispersion
        self.dispersion_weight = dispersion_weight
        self.interpolant = resolve_interpolant(
            interpolant, default="linear", owner="EquilibriumMatchingLoss"
        )
        self.coupling = resolve_coupling(
            coupling, default="independent", owner="EquilibriumMatchingLoss"
        )

    @property
    def train_eps(self) -> float:
        return self.get_scheduled_value("train_eps")

    @train_eps.setter
    def train_eps(self, value: Union[float, BaseScheduler]) -> None:
        self._register_param("train_eps", value)

    def _check_interval(self) -> tuple[float, float]:
        r"""Get training time interval respecting epsilon."""
        eps = self.train_eps
        return eps, 1.0 - eps

    def _sample_t(
        self, batch: int, generator: Optional[torch.Generator]
    ) -> torch.Tensor:
        r"""Draw training times for the configured `t_sampler`.

        'lognormal' is the EDM timestep skew: \(\sigma = e^{z p_{std} + p_{mean}}\)
        with \(z \sim \mathcal{N}(0, 1)\), \(t = 1/(1+\sigma)\), clamped into the
        training interval with a 1e-4 floor. A callable receives
        ``(batch, device=, dtype=, generator=)`` and must return shape (batch,).
        """
        if callable(self.t_sampler):
            return self.t_sampler(
                batch, device=self.device, dtype=self.dtype, generator=generator
            )
        t0, t1 = self._check_interval()
        if self.t_sampler == "lognormal":
            z = torch.randn(
                batch, device=self.device, dtype=self.dtype, generator=generator
            )
            sigma = torch.exp(z * self.t_p_std + self.t_p_mean)
            return (1.0 / (1.0 + sigma)).clamp(min=max(1.0e-4, t0), max=t1)
        return (
            torch.rand(
                batch, device=self.device, dtype=self.dtype, generator=generator
            )
            * (t1 - t0)
            + t0
        )

    def _compute_ct(self, t: torch.Tensor) -> torch.Tensor:
        r"""Target scaling c(t) for the configured variant, times `ct_multiplier`."""
        if callable(self.ct):
            return self.ct(t) * self.ct_multiplier
        if self.ct == "truncated":
            return compute_eqm_ct(
                t, threshold=self.ct_threshold, multiplier=self.ct_multiplier
            )
        if self.ct == "linear":
            return (1.0 - t) * self.ct_multiplier
        return torch.full_like(t, self.ct_multiplier)

    def _reduce_dims(self, ndim: int) -> tuple:
        r"""Cached `tuple(range(1, ndim))` reduction dims (avoids per-call construction)."""
        cache = getattr(self, "_reduce_dims_cache", None)
        if cache is None or cache[0] != ndim:
            self._reduce_dims_cache = (ndim, tuple(range(1, ndim)))
        return self._reduce_dims_cache[1]

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
            energy = (xt * model_output).sum(dim=self._reduce_dims(xt.ndim))
        elif self.energy_type == "l2":
            # g(x) = -0.5 ||f(x)||^2
            energy = -0.5 * model_output.square().sum(dim=self._reduce_dims(model_output.ndim))
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
        r"""
        Compute EqM loss (nn.Module interface).

        Args:
            x: Data samples of shape (batch_size, ...).
            *args: Additional positional arguments.
            y: Optional conditioning tensor (class labels, embeddings, ...)
                forwarded to the model as ``model(x, t, y=y)``; shorthand for
                ``model_kwargs={'y': y}``. ``None`` keeps the unconditional path.
            x0: Optional source samples of shape (batch_size, ...). Defaults to
                standard Gaussian noise; pass a batch from any source
                distribution for arbitrary source-to-target transport.
            model_kwargs: Conditioning arguments (e.g. class labels) forwarded to
                the model. ``None`` (default) is the unconditional path.
            **kwargs: Deprecated. Bare keyword arguments are still forwarded to
                the model for one release but emit a ``DeprecationWarning``; pass
                ``model_kwargs={...}`` instead.

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

    def compute_loss(
        self,
        x: torch.Tensor,
        *args,
        x0: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Compute the equilibrium matching loss.

        Args:
            x: Data samples of shape (batch_size, ...).
            *args: Additional positional arguments.
            x0: Optional source samples (see `forward`).
            model_kwargs: Conditioning arguments forwarded to the model.
            **kwargs: Deprecated bare model kwargs (see `forward`).

        Returns:
            Scalar loss value.
        """
        mk = self._apply_cfg_dropout(
            self._resolve_model_kwargs(
                model_kwargs, kwargs, warn_key="eqm-bare-model-kwargs"
            ),
            generator,
        )
        terms = self.training_losses(
            x, model_kwargs=mk, x0=x0, generator=generator
        )
        loss = terms["loss"]
        weights = terms.get("weights")
        if weights is not None:
            return (weights * loss).sum() / weights.sum().clamp_min(1e-12)
        return loss.mean()

    def training_losses(
        self,
        x1: torch.Tensor,
        model_kwargs: Optional[Dict[str, Any]] = None,
        x0: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Compute training losses with detailed outputs.

        Implements gradient matching with EqM target:
        - Target: $(\epsilon - x) \cdot c(t) = (x_0 - x_1) \cdot c(t)$
        - Time-invariant: the model always receives zeroed time

        Args:
            x1: Data samples of shape (batch_size, ...).
            model_kwargs: Additional model arguments.
            x0: Optional source samples of shape (batch_size, ...); standard
                Gaussian noise when None. Paired against ``x1`` by the configured
                coupling before interpolation.
            generator: RNG for the source draw, the coupling and the time
                sampling; the global RNG when ``None``.

        Returns:
            Dictionary with 'loss' (per-sample), 'pred', 'weights' (per-pair
            coupling weights or None), and optionally 'energy'.
        """
        if model_kwargs is None:
            model_kwargs = {}

        if (
            self.energy_type != "none"
            and self.model.training
            and _has_dtensor_params(self.model)
        ):
            raise RuntimeError(
                "Explicit EqM energies (energy_type != 'none') backpropagate "
                "through the energy's input-gradient (a second-order "
                "backward), which cannot run with FSDP-managed (DTensor) "
                "parameters: resharding hooks free storage the second-order "
                "backward still references. Use energy_type='none' (implicit "
                "EqM) or train with DDP or unsharded parameters."
            )

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

        # Interpolate: xt between x0 (noise) and x1 (data)
        xt, ut = self.interpolant.interpolate(x0, x1, t)

        # EqM target: -ut * c(t) where ut = d_alpha*x1 + d_sigma*x0
        # For linear interpolant, -ut = x0 - x1 (equivalent to original formulation).
        # For VP/cosine, ut encodes the schedule-specific velocity coefficients.
        # Sampling with negate_velocity=True recovers the positive velocity ut*c(t).
        ct = self._compute_ct(t)
        ct = ct.view(batch, *([1] * (xt.ndim - 1)))
        target = -ut * ct

        # For explicit energy, we need gradients w.r.t. xt
        if self.energy_type != "none":
            xt = xt.detach().requires_grad_(True)

        # EqM: zero out time for time-invariance (model still receives t for API compat)
        t_model = torch.zeros_like(t)

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

        terms = {"pred": model_output, "weights": coupled.weights}

        # Compute loss based on prediction type
        if self.prediction == "velocity":
            if self.energy_type == "none":
                # Implicit EqM: model directly predicts gradient field
                terms["loss"] = mean_flat((model_output - target).square())
            else:
                # Explicit EqM-E: compute gradient of energy function
                grad, energy = self._compute_explicit_energy_gradient(
                    xt, model_output, training=self.model.training
                )
                terms["loss"] = mean_flat((grad - target).square())
                terms["energy"] = energy
        else:
            # Score or noise prediction with optional weighting
            t_expanded = expand_t_like_x(t, xt)
            _, drift_var = self.interpolant.compute_drift(xt, t)
            sigma_t, _ = self.interpolant.compute_sigma_t(t_expanded)

            if self.loss_weight == "velocity":
                weight = (drift_var / sigma_t).square()
            elif self.loss_weight == "likelihood":
                weight = drift_var / sigma_t.square()
            else:
                weight = 1.0

            if self.prediction == "noise":
                terms["loss"] = mean_flat(weight * (model_output - x0).square())
            elif self.prediction == "score":
                terms["loss"] = mean_flat(weight * (model_output * sigma_t + x0).square())
            else:
                raise ValueError(f"Unknown prediction type: {self.prediction}")

        if self.loss_weight_fn is not None:
            terms["loss"] = terms["loss"] * self.loss_weight_fn(t)

        # Add dispersive regularization
        if self.apply_dispersion:
            terms["loss"] = terms["loss"] + self.dispersion_weight * disp_loss

        return terms

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"prediction={self.prediction!r}, "
            f"energy_type={self.energy_type!r}, "
            f"interpolant={type(self.interpolant).__name__}, "
            f"coupling={type(self.coupling).__name__}, "
            f"ct={self.ct!r}, "
            f"ct_multiplier={self.ct_multiplier})"
        )


__all__ = ["EquilibriumMatchingLoss"]