r"""Langevin Dynamics Sampler Module."""

import time
from typing import Optional, Union, Tuple, List

import torch

from torchebm.core.base_model import BaseModel
from torchebm.core.base_sampler import BaseSampler
from torchebm.core import (
    BaseScheduler,
    ConstantScheduler,
)
from torchebm.integrators import EulerMaruyamaIntegrator


class LangevinDynamics(BaseSampler):
    r"""
    Langevin Dynamics sampler.

    Update rule:

    \[
    x_{t+1} = x_t - \eta \nabla_x U(x_t) + \sqrt{2\eta} \epsilon_t
    \]

    Args:
        model: Energy-based model to sample from.
        step_size: Step size for gradient descent.
        noise_scale: Scale of Gaussian noise injection.
        decay: Damping coefficient (not supported).
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        ```python
        from torchebm.samplers import LangevinDynamics
        from torchebm.core import DoubleWellEnergy
        import torch

        energy = DoubleWellEnergy()
        sampler = LangevinDynamics(energy, step_size=0.01, noise_scale=1.0)
        samples = sampler.sample(n_samples=100, dim=2, n_steps=500)
        ```
    """

    def __init__(
        self,
        model: BaseModel,
        step_size: Union[float, BaseScheduler] = 1e-3,
        noise_scale: Union[float, BaseScheduler] = 1.0,
        decay: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(model=model, dtype=dtype, device=device)

        if isinstance(step_size, BaseScheduler):
            self.register_scheduler("step_size", step_size)
        else:
            if step_size <= 0:
                raise ValueError("step_size must be positive")
            self.register_scheduler("step_size", ConstantScheduler(step_size))

        if isinstance(noise_scale, BaseScheduler):
            self.register_scheduler("noise_scale", noise_scale)
        else:
            if noise_scale <= 0:
                raise ValueError("noise_scale must be positive")
            self.register_scheduler("noise_scale", ConstantScheduler(noise_scale))

        self.decay = decay
        self.integrator = EulerMaruyamaIntegrator(device=self.device, dtype=self.dtype)

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
        """
        Generates samples using Langevin dynamics.

        Args:
            x (Optional[torch.Tensor]): The initial state to start sampling from. If `None`,
                a random state is created.
            dim (int): The dimension of the state space (if `x` is not provided).
            n_steps (int): The number of MCMC steps to perform.
            n_samples (int): The number of parallel chains/samples to generate.
            thin (int): The thinning factor (not currently supported).
            return_trajectory (bool): Whether to return the full sample trajectory.
            return_diagnostics (bool): Whether to return sampling diagnostics.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
                - The final samples.
                - If `return_trajectory` is `True`, the full trajectory.
                - If `return_diagnostics` is `True`, a tuple of samples and diagnostics.
        """

        self.reset_schedulers()

        if x is None:
            x = torch.randn(n_samples, dim, dtype=self.dtype, device=self.device)
        else:
            x = x.to(device=self.device, dtype=self.dtype)
            dim = x.shape[-1]
            n_samples = x.shape[0]

        if return_trajectory:
            trajectory = torch.empty(
                (n_samples, n_steps, dim), dtype=self.dtype, device=self.device
            )

        if return_diagnostics:
            diagnostics = self._setup_diagnostics(dim, n_steps, n_samples=n_samples)

        with self.autocast_context():
            for i in range(n_steps):
                self.step_schedulers()
                state = {"x": x}
                x = self.integrator.step(
                    state=state,
                    model=self.model,
                    step_size=self.get_scheduled_value("step_size"),
                    noise_scale=self.get_scheduled_value("noise_scale"),
                )["x"]

                if return_trajectory:
                    trajectory[:, i, :] = x

                if return_diagnostics:
                    if n_samples > 1:
                        mean_x = x.mean(dim=0, keepdim=True)
                        var_x = torch.clamp(
                            x.var(dim=0, unbiased=False, keepdim=True),
                            min=1e-10,
                            max=1e10,
                        )
                    else:
                        mean_x = x
                        var_x = torch.zeros_like(x)
                    energy = self.model(x)
                    diagnostics[i, 0, :, :] = (
                        mean_x if n_samples > 1 else mean_x.unsqueeze(0)
                    )
                    diagnostics[i, 1, :, :] = (
                        var_x if n_samples > 1 else var_x.unsqueeze(0)
                    )
                    diagnostics[i, 2, :, :] = energy.view(-1, 1).expand(n_samples, dim)

        if return_trajectory:
            if return_diagnostics:
                return trajectory, diagnostics
            return trajectory
        if return_diagnostics:
            return x, diagnostics
        return x

    def _setup_diagnostics(
        self, dim: int, n_steps: int, n_samples: int = None
    ) -> torch.Tensor:
        if n_samples is not None:
            return torch.empty(
                (n_steps, 3, n_samples, dim), device=self.device, dtype=self.dtype
            )
        return torch.empty((n_steps, 3, dim), device=self.device, dtype=self.dtype)
