r"""Gradient-based optimization samplers.

First-order optimization methods for sampling from energy-based models by
minimizing the energy function through gradient descent.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch

from torchebm.core import BaseModel, BaseSampler, BaseScheduler


class GradientDescentSampler(BaseSampler):
    r"""
    Gradient descent sampler for energy-based models.

    Generates samples by iteratively minimizing the energy function:

    \[
    x_{k+1} = x_k - \eta \nabla_x E(x_k)
    \]

    This is a deterministic optimization-based sampler that finds low-energy
    configurations by following the negative gradient of the energy function.

    Args:
        model: Energy-based model with `gradient()` method.
        step_size: Step size \(\eta\) or scheduler.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.samplers import GradientDescentSampler
        from torchebm.core import DoubleWellModel
        import torch

        energy = DoubleWellModel()
        sampler = GradientDescentSampler(energy, step_size=0.1)
        samples = sampler.sample(n_samples=100, dim=2, n_steps=500)
        ```
    """

    def __init__(
        self,
        model: BaseModel,
        step_size: Union[float, BaseScheduler] = 1e-3,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(
            model=model,
            dtype=dtype,
            device=device,
        )
        self._register_param("step_size", step_size, positive=True)

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
            model_kwargs: Conditioning arguments (e.g. class labels) forwarded to
                the model at every step. Normalized to the sampler device once at
                entry; ``None`` (default) is the exact unconditional path.
            generator: RNG for the initial state when `x is None`; the global
                RNG when ``None``. The updates themselves are deterministic.

        Raises:
            ValueError: If `thin < 1`, or if `x` and `dim` are both `None`.
        """
        if thin < 1:
            raise ValueError("thin must be >= 1")
        if reset_schedulers:
            self.reset_schedulers()

        x = self._init_state(x, dim, n_samples, generator)
        model_kwargs = self._prepare_model_kwargs(model_kwargs)

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

    Args:
        model: Energy-based model with `gradient()` method.
        step_size: Step size \(\eta\) or scheduler.
        momentum: Momentum coefficient \(\mu \in [0, 1)\).
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.samplers import NesterovSampler
        from torchebm.core import DoubleWellModel
        import torch

        energy = DoubleWellModel()
        sampler = NesterovSampler(energy, step_size=0.1, momentum=0.9)
        samples = sampler.sample(n_samples=100, dim=2, n_steps=500)
        ```
    """

    def __init__(
        self,
        model: BaseModel,
        step_size: Union[float, BaseScheduler] = 1e-3,
        momentum: float = 0.9,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(
            model=model,
            dtype=dtype,
            device=device,
        )
        if not (0 <= momentum < 1):
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = momentum

        self._register_param("step_size", step_size, positive=True)

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
            model_kwargs: Conditioning arguments (e.g. class labels) forwarded to
                the model at every step. Normalized to the sampler device once at
                entry; ``None`` (default) is the exact unconditional path.
            generator: RNG for the initial state when `x is None`; the global
                RNG when ``None``. The updates themselves are deterministic.

        Raises:
            ValueError: If `thin < 1`, or if `x` and `dim` are both `None`.
        """
        if thin < 1:
            raise ValueError("thin must be >= 1")
        if reset_schedulers:
            self.reset_schedulers()

        x = self._init_state(x, dim, n_samples, generator)
        model_kwargs = self._prepare_model_kwargs(model_kwargs)

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
        keep_idx = 0
        with self.autocast_context():
            for i in range(n_steps):
                eta = self.get_scheduled_value("step_size")
                lookahead = torch.add(x, v, alpha=mu)
                grad = self._model_gradient(lookahead, model_kwargs)
                v.mul_(mu).sub_(grad, alpha=eta)
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


__all__ = [
    "GradientDescentSampler",
    "NesterovSampler",
]
