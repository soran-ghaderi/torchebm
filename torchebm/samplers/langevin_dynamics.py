from typing import Optional, Union, Tuple

import torch

from torchebm.core.energy_function import EnergyFunction
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
            device: Optional[Union[str, torch.device]] = None
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

    def sample(
            self,
            initial_state: torch.Tensor,
            n_steps: int,
            return_trajectory: bool = False,
            return_diagnostics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Implementation of Langevin dynamics sampling."""
        current_state = initial_state.to(device=self.device, dtype=self.dtype)
        trajectory = [current_state.clone()] if return_trajectory else None
        diagnostics = self._setup_diagnostics() if return_diagnostics else None

        for _ in range(n_steps):
            gradient = self.energy_function.gradient(current_state)

            if return_diagnostics:
                diagnostics['energies'].append(
                    self.energy_function(current_state).item()
                )

            if self.decay > 0:
                current_state = current_state * (1 - self.decay)

            noise = torch.randn_like(current_state) * self.noise_scale
            current_state = current_state - self.step_size * gradient + noise

            if return_trajectory:
                trajectory.append(current_state.clone())

        result = torch.stack(trajectory) if return_trajectory else current_state

        if return_diagnostics:
            if diagnostics['energies']:
                diagnostics['energies'] = torch.tensor(diagnostics['energies'])
            return result, diagnostics
        return result

    @torch.no_grad()
    def sample_parallel(
            self,
            initial_states: torch.Tensor,
            n_steps: int,
            return_diagnostics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Implementation of parallel Langevin dynamics sampling."""
        current_states = initial_states.to(device=self.device, dtype=self.dtype)
        diagnostics = {
            'mean_energies': [],
            'mean_gradient_norms': []
        } if return_diagnostics else None

        for _ in range(n_steps):
            gradients = self.energy_function.gradient(current_states)

            if return_diagnostics:
                diagnostics['mean_energies'].append(
                    self.energy_function(current_states).mean().item()
                )
                diagnostics['mean_gradient_norms'].append(
                    gradients.norm(dim=-1).mean().item()
                )

            if self.decay > 0:
                current_states = current_states * (1 - self.decay)

            noise = torch.randn_like(current_states) * self.noise_scale
            current_states = current_states - self.step_size * gradients + noise

        if return_diagnostics:
            if diagnostics['mean_energies']:
                diagnostics['mean_energies'] = torch.tensor(diagnostics['mean_energies'])
                diagnostics['mean_gradient_norms'] = torch.tensor(diagnostics['mean_gradient_norms'])
            return current_states, diagnostics
        return current_states


