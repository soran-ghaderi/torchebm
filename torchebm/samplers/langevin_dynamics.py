r"""Langevin Dynamics Sampler Module."""

import time
from typing import Optional, Union, Tuple, List
from functools import partial

import torch

from torchebm.core.base_model import BaseModel, GaussianEnergy
from torchebm.core.base_sampler import BaseSampler
from torchebm.core import BaseScheduler, ConstantScheduler, ExponentialDecayScheduler


class LangevinDynamics(BaseSampler):
    r"""
    Langevin Dynamics sampler using discretized gradient-based MCMC.

    This sampler uses a stochastic update rule that combines gradient descent on the
    energy landscape with Gaussian noise to generate samples.

    Args:
        model (BaseModel): The energy-based model to sample from.
        step_size (Union[float, BaseScheduler]): The step size for the Langevin update.
        noise_scale (Union[float, BaseScheduler]): The scale of the Gaussian noise.
        decay (float): Damping coefficient (not currently supported).
        dtype (torch.dtype): The data type for computations.
        device (Optional[Union[str, torch.device]]): The device for computations.
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

        # Register schedulers for step_size and noise_scale
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

        # if device is not None:
        #     self.device = torch.device(device)
        #     energy_function = energy_function.to(self.device)
        # else:
        #     self.device = torch.device("cpu")
        # Respect dtype from BaseSampler; do not override based on device
        self.model = model
        self.step_size = step_size
        self.noise_scale = noise_scale
        self.decay = decay

    def langevin_step(self, prev_x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        r"""
        Performs a single Langevin dynamics update step.

        The update rule is:
        \(x_{t+1} = x_t - \eta \nabla_x U(x_t) + \sqrt{2\eta} \epsilon_t\)

        Args:
            prev_x (torch.Tensor): The current state tensor.
            noise (torch.Tensor): A tensor of Gaussian noise.

        Returns:
            torch.Tensor: The updated state tensor.
        """

        step_size = self.get_scheduled_value("step_size")
        noise_scale = self.get_scheduled_value("noise_scale")

        gradient = self.model.gradient(prev_x)

        # Apply noise scaling
        scaled_noise = noise_scale * noise

        # Apply proper step size and noise scaling
        new_x = (
            prev_x
            - step_size * gradient
            + torch.sqrt(torch.tensor(2.0 * step_size, device=prev_x.device))
            * scaled_noise
        )
        return new_x

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
            x = x.to(self.device)  # Initial batch
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
                # todo: Add decay logic
                # Generate fresh noise for each step
                noise = torch.randn_like(x, device=self.device)

                # Step all schedulers before each MCMC step
                scheduler_values = self.step_schedulers()

                x = self.langevin_step(x, noise)

                if return_trajectory:
                    trajectory[:, i, :] = x

                if return_diagnostics:
                    # Handle mean and variance safely regardless of batch size
                    if n_samples > 1:
                        mean_x = x.mean(dim=0, keepdim=True)
                        var_x = x.var(dim=0, unbiased=False, keepdim=True)
                        var_x = torch.clamp(var_x, min=1e-10, max=1e10)
                    else:
                        # For single sample, just use the value and zeros for variance
                        mean_x = x.clone()
                        var_x = torch.zeros_like(x)

                    # Compute energy values
                    energy = self.model(x)

                    # Store the diagnostics safely
                    for b in range(n_samples):
                        diagnostics[i, 0, b, :] = mean_x[b if n_samples > 1 else 0]
                        diagnostics[i, 1, b, :] = var_x[b if n_samples > 1 else 0]
                        diagnostics[i, 2, b, :] = energy[b].reshape(-1)

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
        else:
            return torch.empty((n_steps, 3, dim), device=self.device, dtype=self.dtype)
