r"""Gradient-based optimization samplers.

First-order optimization methods for sampling from energy-based models by
minimizing the energy function through gradient descent.
"""

from __future__ import annotations

from typing import Optional, Union, Tuple, List

import torch

from torchebm.core import BaseModel
from torchebm.core.base_sampler import BaseSampler
from torchebm.core import BaseScheduler, ConstantScheduler


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
        use_mixed_precision: Whether to use mixed precision.

    Example:
        ```python
        from torchebm.samplers import GradientDescentSampler
        from torchebm.core import DoubleWellEnergy
        import torch

        energy = DoubleWellEnergy()
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
        use_mixed_precision: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            dtype=dtype,
            device=device,
            use_mixed_precision=use_mixed_precision,
        )
        if isinstance(step_size, BaseScheduler):
            self.register_scheduler("step_size", step_size)
        else:
            if step_size <= 0:
                raise ValueError("step_size must be positive")
            self.register_scheduler("step_size", ConstantScheduler(step_size))

    @torch.no_grad()
    def sample(
        self,
        x: Optional[torch.Tensor] = None,
        dim: int = 10,
        n_steps: int = 100,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
        *args,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
        r"""
        Generate samples via gradient descent optimization.

        Args:
            x: Initial state. If None, samples from N(0, I).
            dim: Dimension of state space (used if x is None).
            n_steps: Number of gradient descent steps.
            n_samples: Number of parallel chains/samples.
            thin: Thinning factor (not currently supported).
            return_trajectory: Whether to return full trajectory.
            return_diagnostics: Whether to return diagnostics.

        Returns:
            Final samples or (samples, diagnostics) if return_diagnostics=True.
        """
        self.reset_schedulers()

        if x is None:
            x = torch.randn(n_samples, dim, device=self.device, dtype=self.dtype)
        else:
            x = x.to(device=self.device, dtype=self.dtype)

        diagnostics = self._setup_diagnostics() if return_diagnostics else None
        trajectory = [x.clone()] if return_trajectory else None

        with self.autocast_context():
            for _ in range(n_steps):
                self.step_schedulers()
                eta = self.get_scheduled_value("step_size")
                grad = self.model.gradient(x)
                x = x - eta * grad

                if return_trajectory:
                    trajectory.append(x.clone())

        if return_diagnostics:
            return (
                torch.stack(trajectory, dim=1) if return_trajectory else x,
                [diagnostics],
            )
        return torch.stack(trajectory, dim=1) if return_trajectory else x


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
        use_mixed_precision: Whether to use mixed precision.

    Example:
        ```python
        from torchebm.samplers import NesterovSampler
        from torchebm.core import DoubleWellEnergy
        import torch

        energy = DoubleWellEnergy()
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
        use_mixed_precision: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            dtype=dtype,
            device=device,
            use_mixed_precision=use_mixed_precision,
        )
        if not (0 <= momentum < 1):
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = momentum

        if isinstance(step_size, BaseScheduler):
            self.register_scheduler("step_size", step_size)
        else:
            if step_size <= 0:
                raise ValueError("step_size must be positive")
            self.register_scheduler("step_size", ConstantScheduler(step_size))

    @torch.no_grad()
    def sample(
        self,
        x: Optional[torch.Tensor] = None,
        dim: int = 10,
        n_steps: int = 100,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
        *args,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
        r"""
        Generate samples via Nesterov accelerated gradient descent.

        Args:
            x: Initial state. If None, samples from N(0, I).
            dim: Dimension of state space (used if x is None).
            n_steps: Number of optimization steps.
            n_samples: Number of parallel chains/samples.
            thin: Thinning factor (not currently supported).
            return_trajectory: Whether to return full trajectory.
            return_diagnostics: Whether to return diagnostics.

        Returns:
            Final samples or (samples, diagnostics) if return_diagnostics=True.
        """
        self.reset_schedulers()

        if x is None:
            x = torch.randn(n_samples, dim, device=self.device, dtype=self.dtype)
        else:
            x = x.to(device=self.device, dtype=self.dtype)

        v = torch.zeros_like(x)
        diagnostics = self._setup_diagnostics() if return_diagnostics else None
        trajectory = [x.clone()] if return_trajectory else None

        mu = self.momentum
        with self.autocast_context():
            for _ in range(n_steps):
                self.step_schedulers()
                eta = self.get_scheduled_value("step_size")
                lookahead = x + mu * v
                grad = self.model.gradient(lookahead)
                v = mu * v - eta * grad
                x = x + v

                if return_trajectory:
                    trajectory.append(x.clone())

        if return_diagnostics:
            return (
                torch.stack(trajectory, dim=1) if return_trajectory else x,
                [diagnostics],
            )
        return torch.stack(trajectory, dim=1) if return_trajectory else x


__all__ = [
    "GradientDescentSampler",
    "NesterovSampler",
]
