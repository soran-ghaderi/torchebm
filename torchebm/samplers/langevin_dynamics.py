import time
from typing import Optional, Union, Tuple, List
from functools import partial

import torch

from torchebm.core.energy_function import EnergyFunction, GaussianEnergy
from torchebm.core.sampler import Sampler


class LangevinDynamics(Sampler):

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
        print(
            "Sampling chain, dim: ", dim, "n_steps: ", n_steps, "n_samples: ", n_samples
        )
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
        """Initialize the diagnostics dictionary."""
        if n_samples is not None:
            return torch.empty(
                (n_steps, 3, n_samples, dim), device=self.device, dtype=self.dtype
            )
        else:
            return torch.empty((n_steps, 3, dim), device=self.device, dtype=self.dtype)


energy_fn = GaussianEnergy(mean=torch.zeros(10), cov=torch.eye(10))
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Langevin dynamics model
langevin_sampler = LangevinDynamics(
    energy_function=energy_fn, step_size=5e-3, device=device
).to(device)

# Initial state: batch of 100 samples, 10-dimensional space
ts = time.time()
# Run Langevin sampling for 500 steps
final_x = langevin_sampler.sample_chain(
    dim=10, n_steps=500, n_samples=10000, return_trajectory=False
)

print(final_x.shape)  # Output: (100, 10)  (final state)
# print(xs.shape)  # Output: (500, 100, 10)  (history of all states)
print("Time taken: ", time.time() - ts)

n_samples = 250
n_steps = 500
dim = 10
final_samples, diagnostics = langevin_sampler.sample_chain(
    n_samples=n_samples,
    n_steps=n_steps,
    dim=dim,
    return_trajectory=True,
    return_diagnostics=True,
)
print(final_samples.shape)  # Output: (100, 10)  (final state)
print(diagnostics.shape)  # (500, 3, 100, 10) -> All diagnostics

x_init = torch.randn(n_samples, dim, dtype=torch.float32, device="cuda")
samples = langevin_sampler.sample(x=x_init, n_steps=100)
print(samples.shape)  # Output: (100, 10)  (final state)
