import time
from typing import Optional, Union, Tuple, List
from functools import partial

from functorch import vmap

import torch

from torchebm.core.energy_function import EnergyFunction, GaussianEnergy
from torchebm.core.sampler import Sampler


class LangevinDynamics(Sampler):
    """Langevin dynamics sampler implementation."""

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
            decay: Damping coefficient
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

    # @torch.jit.script
    def langevin_step(self, prev_x: torch.Tensor):
        draw = torch.randn_like(prev_x)
        gradient_fn = partial(self.energy_function.gradient)
        new_x = (
            prev_x
            - self.step_size * gradient_fn(prev_x)
            + torch.sqrt(torch.tensor(2.0 * self.step_size, device=prev_x.device))
            * draw
        )
        return new_x

    @torch.no_grad()
    def sample_chain(
        self,
        dim: int = 10,
        n_steps: int = 100,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
        """
        Return a chain of Langevin samples over n-steps.
        Args:
            x:
            n_steps:
            n_samples:
            thin:
            return_diagnostics:

        Returns:

        """
        # configurations for CUDA
        # device = x.device
        # dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        x = torch.randn(
            n_samples, dim, dtype=self.dtype, device=self.device
        )  # Initial batch

        xs = torch.empty(
            (n_samples, n_steps, dim), dtype=self.dtype, device=self.device
        )  # Storage for samples

        # xs = torch.empty((n_steps, *x.shape), dtype=dtype, device=x.device)

        with torch.cuda.amp.autocast():
            for i in range(n_steps):
                x = self.langevin_step(x)
                # xs[i] = x
                xs[:, i, :] = x

        return x, xs

    @torch.no_grad()
    def sample(
        self,
        initial_state: torch.Tensor,
        n_steps: int,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Implementation of Langevin dynamics sampling."""
        current_state = initial_state.to(device=self.device, dtype=self.dtype)
        trajectory = [current_state.clone()] if return_trajectory else None
        diagnostics = self._setup_diagnostics() if return_diagnostics else None

        for _ in range(n_steps):
            gradient = self.energy_function.gradient(current_state)

            if return_diagnostics:
                # diagnostics['energies'].append(
                #     self.energy_function(current_state).item()
                # )
                diagnostics["energies"] = torch.cat(
                    [
                        diagnostics["energies"],
                        self.energy_function(current_state).unsqueeze(0),
                    ]
                )

            if self.decay > 0:
                current_state = current_state * (1 - self.decay)

            noise = torch.randn_like(current_state) * self.noise_scale
            current_state = current_state - self.step_size * gradient + noise

            if return_trajectory:
                trajectory.append(current_state.clone())

        result = torch.stack(trajectory) if return_trajectory else current_state

        if return_diagnostics:
            # if diagnostics["energies"]:
            if diagnostics["energies"].nelement() > 0:
                diagnostics["energies"] = diagnostics["energies"]
            return result, diagnostics
        return result

    @torch.no_grad()
    def sample_parallel(
        self,
        initial_states: torch.Tensor,
        n_steps: int,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Implementation of parallel Langevin dynamics sampling."""
        current_states = initial_states.to(device=self.device, dtype=self.dtype)
        diagnostics = (
            {"mean_energies": [], "mean_gradient_norms": []}
            if return_diagnostics
            else None
        )

        for _ in range(n_steps):
            gradients = self.energy_function.gradient(current_states)

            if return_diagnostics:
                diagnostics["mean_energies"].append(
                    self.energy_function(current_states).mean().item()
                )
                diagnostics["mean_gradient_norms"].append(
                    gradients.norm(dim=-1).mean().item()
                )

            if self.decay > 0:
                current_states = current_states * (1 - self.decay)

            noise = torch.randn_like(current_states) * self.noise_scale
            current_states = current_states - self.step_size * gradients + noise

        if return_diagnostics:
            if diagnostics["mean_energies"]:
                diagnostics["mean_energies"] = torch.tensor(
                    diagnostics["mean_energies"]
                )
                diagnostics["mean_gradient_norms"] = torch.tensor(
                    diagnostics["mean_gradient_norms"]
                )
            return current_states, diagnostics
        return current_states

    def _setup_diagnostics(self) -> dict:
        """Initialize the diagnostics dictionary."""
        return {
            "energies": torch.empty(0, device=self.device, dtype=self.dtype),
        }


# class EF(EnergyFunction):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x**2
#
#     def gradient(self, x: torch.Tensor) -> torch.Tensor:
#         return 2 * x

energy_fn = GaussianEnergy(mean=torch.zeros(10), cov=torch.eye(10))
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Langevin dynamics model
langevin_chain = LangevinDynamics(energy_function=energy_fn, step_size=5e-3).to(device)

# Initial state: batch of 100 samples, 10-dimensional space
x0 = torch.randn(100, 10, device=device)
ts = time.time()
# Run Langevin sampling for 500 steps
final_x, xs = langevin_chain.sample_chain(dim=10, n_steps=500, n_samples=10000)

print(final_x.shape)  # Output: (100, 10)  (final state)
print(xs.shape)  # Output: (500, 100, 10)  (history of all states)
print("Time taken: ", time.time() - ts)
