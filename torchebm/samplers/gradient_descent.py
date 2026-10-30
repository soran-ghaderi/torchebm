r"""Gradient-based optimization samplers.

First-order optimization methods for sampling from energy-based models by
minimizing the energy function through gradient descent.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from torchebm._deprecation import declare_deprecation
from torchebm.core import BaseModel, BaseSampler, BaseScheduler

_MODEL_BOUND_DEPRECATION = declare_deprecation(
    module=__name__,
    name="model-bound optimization samplers",
    since="0.8.10",
    deprecated_on="2026-09-29",
    until="0.12.0",
    grace_months=4,
    replacement="sampler(model, x, n_steps=...)",
    message=(
        "GradientDescentSampler(model=...), NesterovSampler(model=...) and "
        "sample() are deprecated: construct the sampler without a model and "
        "call sampler(model, x, ...)."
    ),
    removal=(
        "the model argument, sample and _require_model, and the model field of "
        "__repr__ in GradientDescentSampler and NesterovSampler"
    ),
)


class GradientDescentSampler(BaseSampler):
    r"""
    Gradient descent sampler for energy-based models.

    Generates samples by iteratively minimizing the energy function:

    \[
    x_{k+1} = x_k - \eta \nabla_x E(x_k)
    \]

    This is a deterministic optimization-based sampler that finds low-energy
    configurations by following the negative gradient of the energy function.
    The call consumes the score \(-\nabla_x E(x_k)\); see `BaseSampler` for the
    call contract.

    A model whose output is a field \(f\) that is not the gradient of an
    energy, such as an implicit equilibrium-matching model, gets the same
    update \(x_{k+1} = x_k + \eta f(x_k)\): a fixed-point iteration toward the
    stable zeros of \(f\), and the explicit Euler step of \(\dot x = f(x)\).

    Args:
        model: Deprecated. Energy-based model sampled by `sample()`; construct
            without it and call ``sampler(model, x, ...)``.
        step_size: Step size \(\eta\) or scheduler.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        The sampler of Equilibrium Matching (Wang and Du, 2025) on EqM-XL/2,
        a class-conditional model on ImageNet 256 latents (see
        `EquilibriumMatchingLoss`):

        ```python
        from torchebm.samplers import GradientDescentSampler

        sampler = GradientDescentSampler(step_size=0.0017)
        samples = sampler(model, torch.randn(64, 4, 32, 32), n_steps=250, model_kwargs={"y": y})
        ```
    """

    def __init__(
        self,
        model: Optional[BaseModel] = None,
        step_size: Union[float, BaseScheduler] = 1e-3,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(
            model=model,
            dtype=dtype,
            device=device,
        )
        if model is not None:
            _MODEL_BOUND_DEPRECATION.warn(type(self).__name__)
        self._register_param("step_size", step_size, positive=True)

    @torch.no_grad()
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        *,
        n_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        r"""Run `n_steps` gradient descent updates of `model` from `x`; see
        `BaseSampler.forward` for the arguments. `generator` is unused: the
        updates are deterministic."""
        score = self._model_fn(model, model_kwargs)
        self.reset_schedulers(model)
        trajectory = None
        if return_trajectory:
            trajectory = x.new_empty((x.shape[0], n_steps, *x.shape[1:]))
        for i in range(n_steps):
            x = torch.add(x, score(x), alpha=self.get_scheduled_value("step_size"))
            self.step_schedulers(model)
            if trajectory is not None:
                trajectory[:, i] = x
        return x if trajectory is None else trajectory

    @torch.no_grad()
    def sample(
        self,
        x: Optional[torch.Tensor] = None,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        n_steps: int = 100,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
        reset_schedulers: bool = True,
        *,
        y: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        r"""Generate samples via gradient descent optimization.

        Args:
            x: Initial state. If None, samples from N(0, I).
            dim: State dimension (int) or shape (tuple), used when `x is None`.
            n_steps: Number of gradient descent steps.
            n_samples: Number of parallel chains/samples.
            thin: Keep every `thin`-th sample. Final length `n_steps // thin`.
            return_trajectory: If True, return the full kept trajectory of
                shape `[n_samples, n_steps // thin, *data_shape]`.
            return_diagnostics: If True, also return a dict with key
                ``"energy"`` (`[n_kept]`).
            reset_schedulers: If True (default), reset registered schedulers.
            y: Optional conditioning tensor forwarded to the model; shorthand
                for ``model_kwargs={'y': y}``.
            model_kwargs: Conditioning arguments (e.g. class labels) forwarded to
                the model at every step. Normalized to the sampler device once at
                entry; ``None`` (default) is the exact unconditional path.
            generator: RNG for the initial state when `x is None`; the global
                RNG when ``None``. The updates themselves are deterministic.

        Raises:
            ValueError: If `thin < 1`, or if `x` and `dim` are both `None`.
        """
        self._require_model()
        if thin < 1:
            raise ValueError("thin must be >= 1")
        if reset_schedulers:
            self.reset_schedulers()

        x = self._init_state(x, dim, n_samples, generator)
        model_kwargs = self._prepare_model_kwargs(
            self._merge_condition(model_kwargs, y)
        )

        n_kept = n_steps // thin
        if return_trajectory:
            trajectory = torch.empty(
                x.shape[0], n_kept, *x.shape[1:], device=x.device, dtype=x.dtype
            )

        diagnostics: Optional[Dict[str, torch.Tensor]] = None
        if return_diagnostics:
            diagnostics = {
                "energy": torch.empty(n_kept, dtype=self.dtype, device=self.device),
            }

        keep_idx = 0
        with self.autocast_context():
            for i in range(n_steps):
                eta = self.get_scheduled_value("step_size")
                grad = self._model_gradient(x, model_kwargs)
                x = torch.sub(x, grad, alpha=eta)

                if (i + 1) % thin == 0:
                    if return_trajectory:
                        trajectory[:, keep_idx] = x
                    if return_diagnostics:
                        diagnostics["energy"][keep_idx] = self._model_energy(
                            x, model_kwargs
                        ).mean()
                    keep_idx += 1

                self.step_schedulers()

        output = trajectory if return_trajectory else x
        return (output, diagnostics) if return_diagnostics else output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={type(self.model).__name__}, "
            f"step_size={self.get_scheduled_value('step_size')})"
        )


class NesterovSampler(BaseSampler):
    r"""
    Nesterov accelerated gradient sampler for energy-based models.

    Uses Nesterov momentum to accelerate convergence to low-energy states:

    \[
    v_{k+1} = \mu v_k - \eta \nabla_x E(x_k + \mu v_k)
    \]

    \[
    x_{k+1} = x_k + v_{k+1}
    \]

    where \(\mu\) is the momentum coefficient and \(\eta\) is the step size.

    With ``lookahead_only=True`` the momentum only moves the point where the
    score is evaluated, \(v_{k+1} = -\eta \nabla_x E(x_k + \mu v_k)\), so
    \(x_{k+1} = x_k - \eta \nabla_x E(x_k + \mu (x_k - x_{k-1}))\): the NAG-GD
    sampler of Equilibrium Matching (Wang and Du, 2025, Eq. 9).

    Called as ``sampler(model, x, n_steps=...)``, the model giving
    the score \(-\nabla_x E\) at the lookahead point \(x_k + \mu v_k\); see
    `BaseSampler`. On a field that is not a gradient, the update is
    momentum-accelerated fixed-point iteration toward the field's stable zeros.

    Args:
        model: Deprecated. Energy-based model sampled by `sample()`; construct
            without it and call ``sampler(model, x, ...)``.
        step_size: Step size \(\eta\) or scheduler.
        momentum: Momentum coefficient \(\mu \in [0, 1)\).
        lookahead_only: Step with the lookahead score alone, without the
            momentum term. Default False, standard Nesterov momentum.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        The NAG-GD sampler of Equilibrium Matching (Wang and Du, 2025) on
        EqM-XL/2, a class-conditional model on ImageNet 256 latents (see
        `EquilibriumMatchingLoss`):

        ```python
        from torchebm.samplers import NesterovSampler

        sampler = NesterovSampler(step_size=0.0017, momentum=0.3, lookahead_only=True)
        samples = sampler(model, torch.randn(64, 4, 32, 32), n_steps=250, model_kwargs={"y": y})
        ```
    """

    def __init__(
        self,
        model: Optional[BaseModel] = None,
        step_size: Union[float, BaseScheduler] = 1e-3,
        momentum: float = 0.9,
        lookahead_only: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(
            model=model,
            dtype=dtype,
            device=device,
        )
        if model is not None:
            _MODEL_BOUND_DEPRECATION.warn(type(self).__name__)
        if not (0 <= momentum < 1):
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = momentum
        self.lookahead_only = bool(lookahead_only)

        self._register_param("step_size", step_size, positive=True)

    @torch.no_grad()
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        *,
        n_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        r"""Run `n_steps` Nesterov updates of `model` from `x`; see
        `BaseSampler.forward` for the arguments. `generator` is unused: the
        updates are deterministic."""
        score = self._model_fn(model, model_kwargs)
        self.reset_schedulers(model)
        v = torch.zeros_like(x)
        mu = self.momentum
        carry = 0.0 if self.lookahead_only else mu
        trajectory = None
        if return_trajectory:
            trajectory = x.new_empty((x.shape[0], n_steps, *x.shape[1:]))
        for i in range(n_steps):
            eta = self.get_scheduled_value("step_size")
            lookahead_score = score(torch.add(x, v, alpha=mu))
            v.mul_(carry).add_(lookahead_score, alpha=eta)
            x = x + v
            self.step_schedulers(model)
            if trajectory is not None:
                trajectory[:, i] = x
        return x if trajectory is None else trajectory

    @torch.no_grad()
    def sample(
        self,
        x: Optional[torch.Tensor] = None,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        n_steps: int = 100,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
        reset_schedulers: bool = True,
        *,
        y: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        r"""Generate samples via Nesterov accelerated gradient descent.

        Args:
            x: Initial state. If None, samples from N(0, I).
            dim: State dimension (int) or shape (tuple), used when `x is None`.
            n_steps: Number of optimization steps.
            n_samples: Number of parallel chains/samples.
            thin: Keep every `thin`-th sample. Final length `n_steps // thin`.
            return_trajectory: If True, return the full kept trajectory of
                shape `[n_samples, n_steps // thin, *data_shape]`.
            return_diagnostics: If True, also return a dict with key
                ``"energy"`` (`[n_kept]`).
            reset_schedulers: If True (default), reset registered schedulers.
            y: Optional conditioning tensor forwarded to the model; shorthand
                for ``model_kwargs={'y': y}``.
            model_kwargs: Conditioning arguments (e.g. class labels) forwarded to
                the model at every step. Normalized to the sampler device once at
                entry; ``None`` (default) is the exact unconditional path.
            generator: RNG for the initial state when `x is None`; the global
                RNG when ``None``. The updates themselves are deterministic.

        Raises:
            ValueError: If `thin < 1`, or if `x` and `dim` are both `None`.
        """
        self._require_model()
        if thin < 1:
            raise ValueError("thin must be >= 1")
        if reset_schedulers:
            self.reset_schedulers()

        x = self._init_state(x, dim, n_samples, generator)
        model_kwargs = self._prepare_model_kwargs(
            self._merge_condition(model_kwargs, y)
        )

        v = torch.zeros_like(x)
        n_kept = n_steps // thin
        if return_trajectory:
            trajectory = torch.empty(
                x.shape[0], n_kept, *x.shape[1:], device=x.device, dtype=x.dtype
            )

        diagnostics: Optional[Dict[str, torch.Tensor]] = None
        if return_diagnostics:
            diagnostics = {
                "energy": torch.empty(n_kept, dtype=self.dtype, device=self.device),
            }

        mu = self.momentum
        carry = 0.0 if self.lookahead_only else mu
        keep_idx = 0
        with self.autocast_context():
            for i in range(n_steps):
                eta = self.get_scheduled_value("step_size")
                lookahead = torch.add(x, v, alpha=mu)
                grad = self._model_gradient(lookahead, model_kwargs)
                v.mul_(carry).sub_(grad, alpha=eta)
                x = x + v

                if (i + 1) % thin == 0:
                    if return_trajectory:
                        trajectory[:, keep_idx] = x
                    if return_diagnostics:
                        diagnostics["energy"][keep_idx] = self._model_energy(
                            x, model_kwargs
                        ).mean()
                    keep_idx += 1

                self.step_schedulers()

        output = trajectory if return_trajectory else x
        return (output, diagnostics) if return_diagnostics else output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={type(self.model).__name__}, "
            f"step_size={self.get_scheduled_value('step_size')}, "
            f"momentum={self.momentum})"
        )


__all__ = [
    "GradientDescentSampler",
    "NesterovSampler",
]
