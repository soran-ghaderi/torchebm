import time
from typing import Optional, Union, Tuple, List
from functools import partial

import torch

from torchebm.core.energy_function import EnergyFunction, GaussianEnergy
from torchebm.core.basesampler import BaseSampler


class LangevinDynamics(BaseSampler):
    """
    Langevin dynamics sampler.

    Inherits from :class:`BaseSampler`.

    Args:
        energy_function (EnergyFunction): Energy function to sample from.
        step_size (float): Step size for updates.
        noise_scale (float): Scale of the noise.
        decay (float): Damping coefficient (not supported yet).
        dtype (torch.dtype): Data type to use for the computations.
        device (str): Device to run the computations on (e.g., "cpu" or "cuda").

    Methods:
        langevin_step(prev_x, noise): Perform a Langevin step.
        sample_chain(x, dim, n_steps, n_samples, return_trajectory, return_diagnostics): Run the sampling process.
        _setup_diagnostics(dim, n_steps, n_samples): Initialize the diagnostics
    """

    def __init__(
        self,
        energy_function: EnergyFunction,
        step_size: float = 1e-3,
        noise_scale: float = 1.0,
        decay: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize Langevin dynamics sampler.

        Args:
            energy_function: The energy function to sample from
            step_size: The step size for updates
            noise_scale: Scale of the noise
            decay: Damping coefficient (not supported yet)
            dtype: Tensor dtype to use
            device: Device to run on
        """

        super().__init__(energy_function, dtype, device)

        if step_size <= 0 or noise_scale <= 0:
            raise ValueError("step_size and noise_scale must be positive")
        if not 0 <= decay <= 1:
            raise ValueError("decay must be between 0 and 1")

        self.step_size = step_size
        self.noise_scale = noise_scale
        self.decay = decay
        self.energy_function = energy_function
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32

    def langevin_step(self, prev_x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        gradient_fn = partial(self.energy_function.gradient)
        new_x = (
            prev_x
            - self.step_size * gradient_fn(prev_x)
            + torch.sqrt(torch.tensor(2.0 * self.step_size, device=prev_x.device))
            * noise
        )
        return new_x

    @torch.no_grad()
    def sample_chain(
        self,
        x: Optional[torch.Tensor] = None,
        dim: int = 10,
        n_steps: int = 100,
        n_samples: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
        """
        Generates samples using Langevin dynamics.

        This method simulates a Markov chain using Langevin dynamics, where each step updates
        the state `x_t` according to the discretized Langevin equation:

        .. math::

            x_{t+1} = x_t - \eta \\nabla_x U(x_t) + \sqrt{2\eta} \epsilon_t

        This process generates samples that asymptotically follow the Boltzmann distribution:

        .. math::

            p(x) \propto e^{-U(x)}

        where :math:`U(x)` defines the energy landscape.

        ### Algorithm:
        1. If `x` is not provided, initialize it with Gaussian noise.
        2. Iteratively update `x` for `n_steps` using `self.langevin_step()`.
        3. Optionally track trajectory (`return_trajectory=True`).
        4. Optionally collect diagnostics such as mean, variance, and energy gradients.



        Args:
            x: Initial state to start the sampling from.
            dim: Dimension of the state space.
            n_steps: Number of steps to take between samples.
            n_samples: Number of samples to generate.
            return_trajectory: Whether to return the trajectory of the samples.
            return_diagnostics: Whether to return the diagnostics of the sampling process.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
                - If `return_trajectory=False` and `return_diagnostics=False`, returns the final
                  samples of shape `(n_samples, dim)`.
                - If `return_trajectory=True`, returns a tensor of shape `(n_samples, n_steps, dim)`,
                  containing the sampled trajectory.
                - If `return_diagnostics=True`, returns a tuple `(samples, diagnostics)`, where
                  `diagnostics` is a list of dictionaries storing per-step statistics.
        """
        if x is None:
            x = torch.randn(n_samples, dim, dtype=self.dtype, device=self.device)
        else:
            x = x.to(self.device)  # Initial batch

        if return_trajectory:
            trajectory = torch.empty(
                (n_samples, n_steps, dim), dtype=self.dtype, device=self.device
            )

        if return_diagnostics:
            diagnostics = self._setup_diagnostics(dim, n_steps, n_samples=n_samples)

        with torch.amp.autocast("cuda"):
            noise = torch.randn_like(x, device=self.device)
            for i in range(n_steps):
                x = self.langevin_step(x, noise)
                if return_trajectory:
                    trajectory[:, i, :] = x

                if return_diagnostics:
                    mean_x = x.mean(dim=0)
                    var_x = x.var(dim=0)
                    energy = self.energy_function.gradient(x)

                    # Stack the diagnostics along the second dimension (index 1)
                    diagnostics[i, 0, :, :] = mean_x
                    diagnostics[i, 1, :, :] = var_x
                    diagnostics[i, 2, :, :] = energy

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
