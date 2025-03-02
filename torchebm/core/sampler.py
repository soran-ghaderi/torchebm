from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, List

import torch
from torchebm.core.energy_function import EnergyFunction


class Sampler(ABC):
    """
    Base class for samplers.
    """

    def __init__(
        self,
        energy_function: EnergyFunction,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize the sampler.

        Args:
            energy_function:
            dtype:
            device:
        """
        self.energy_function = energy_function
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def sample(
        self,
        x: Optional[torch.Tensor] = None,
        dim: int = 10,
        n_steps: int = 100,
        n_samples: int = 1,
        thin: int = 1,  # not supported yet
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
        return self.sample_chain(
            x=x,
            dim=dim,
            n_steps=n_steps,
            n_samples=n_samples,
            return_trajectory=return_trajectory,
            return_diagnostics=return_diagnostics,
        )

    @abstractmethod
    def sample_chain(
        self,
        dim: int = 10,
        n_steps: int = 100,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
        raise NotImplementedError

    def _setup_diagnostics(self) -> dict:
        """Initialize the diagnostics dictionary."""
        return {
            "energies": torch.empty(0, device=self.device, dtype=self.dtype),
            "acceptance_rate": torch.tensor(0.0, device=self.device, dtype=self.dtype),
        }

    def to(self, device: Union[str, torch.device]) -> "Sampler":
        """Move sampler to specified device."""
        self.device = device
        return self
