import time
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, List

import torch

from torchebm.core.energy_function import EnergyFunction


class Sampler(ABC):
    def __init__(
        self,
        energy_function: EnergyFunction,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.energy_function = energy_function
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def sample(self, initial_state: torch.Tensor, n_steps: int) -> torch.Tensor:
        pass

    def sample_chain(
        self,
        initial_state: torch.Tensor,
        n_steps: int,
        n_samples: int,
        thin: int = 1,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
        """
        Generate multiple independent samples using the sampler.

        Args:
            initial_state: Starting point
            n_steps: Steps between samples
            n_samples: Number of samples to generate
            thin: Keep every nth sample
            return_diagnostics: Whether to return diagnostic info

        Returns:
            - samples tensor of shape (n_samples, *state_shape)
            - optionally with list of diagnostics dicts if return_diagnostics=True
        """
        samples = []
        diagnostics_list = [] if return_diagnostics else None
        current_state = initial_state.to(device=self.device, dtype=self.dtype)

        for _ in range(n_samples):
            if return_diagnostics:
                current_state, diag = self.sample(
                    current_state, n_steps, return_diagnostics=True
                )
                diagnostics_list.append(diag)
            else:
                current_state = self.sample(current_state, n_steps)
            samples.append(current_state.clone())

        samples = torch.stack(samples)[::thin]

        if return_diagnostics:
            return samples, diagnostics_list[::thin]
        return samples

    @torch.no_grad()
    def sample_parallel(
        self,
        initial_states: torch.Tensor,
        n_steps: int,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Run parallel sampling from the given initial states.

        Args:
            initial_states: Starting points of shape (batch_size, *state_shape)
            n_steps: Number of simulation steps
            return_diagnostics: Whether to return diagnostic info

        Returns:
            final states tensor of shape (batch_size, *state_shape)
            optionally with diagnostics dict if return_diagnostics=True
        """
        raise NotImplementedError("Parallel sampling not implemented for this sampler")

    def _setup_diagnostics(self) -> dict:
        """Initialize the diagnostics dictionary."""
        return {
            "energies": torch.tensor([], device=self.device, dtype=self.dtype),
            "acceptance_rate": torch.tensor(
                0.0, device=self.device, dtype=self.dtype
            ),  # todo: this should be only included in methods that use acceptance rate
        }

    def to(self, device: Union[str, torch.device]) -> "Sampler":
        """Move sampler to specified device."""
        self.device = device
        return self
