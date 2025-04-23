from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, List, Dict

import torch

from torchebm.core import BaseEnergyFunction, BaseScheduler


class BaseSampler(ABC):
    """
    Base class for samplers.

    Args:
        energy_function (BaseEnergyFunction): Energy function to sample from.
        dtype (torch.dtype): Data type to use for the computations.
        device (str): Device to run the computations on (e.g., "cpu" or "cuda").

    Methods:
        sample(x, dim, k_steps, n_samples, thin, return_trajectory, return_diagnostics): Run the sampling process.
        sample_chain(dim, k_steps, n_samples, thin, return_trajectory, return_diagnostics): Run the sampling process.
        _setup_diagnostics(): Initialize the diagnostics dictionary.
        to(device): Move sampler to specified device.
    """

    def __init__(
        self,
        energy_function: BaseEnergyFunction,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.energy_function = energy_function
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.schedulers: Dict[str, BaseScheduler] = {}

    @abstractmethod
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
        Run the sampling process.

        Args:
            x: Initial state to start the sampling from.
            dim: Dimension of the state space.
            k_steps: Number of steps to take between samples.
            n_samples: Number of samples to generate.
            thin: Thinning factor (not supported yet).
            return_trajectory: Whether to return the trajectory of the samples.
            return_diagnostics: Whether to return the diagnostics of the sampling process.

        Returns:
            torch.Tensor: Samples from the sampler.
            List[dict]: Diagnostics of the sampling process.
        """
        raise NotImplementedError

    def register_scheduler(self, name: str, scheduler: BaseScheduler) -> None:
        """
        Register a parameter scheduler.

        Args:
            name: Name of the parameter to schedule
            scheduler: Scheduler instance to use
        """
        self.schedulers[name] = scheduler

    def get_scheduled_value(self, name: str) -> float:
        """
        Get current value for a scheduled parameter.

        Args:
            name: Name of the scheduled parameter

        Returns:
            Current value of the parameter

        Raises:
            KeyError: If no scheduler exists for the parameter
        """
        if name not in self.schedulers:
            raise KeyError(f"No scheduler registered for parameter '{name}'")
        return self.schedulers[name].get_value()

    def step_schedulers(self) -> Dict[str, float]:
        """
        Advance all schedulers by one step.

        Returns:
            Dictionary mapping parameter names to their updated values
        """
        return {name: scheduler.step() for name, scheduler in self.schedulers.items()}

    def reset_schedulers(self) -> None:
        """Reset all schedulers to their initial state."""
        for scheduler in self.schedulers.values():
            scheduler.reset()

    # @abstractmethod
    def _setup_diagnostics(self) -> dict:
        """
        Initialize the diagnostics dictionary.

            .. deprecated:: 1.0
               This method is deprecated and will be removed in a future version.
        """
        return {
            "energies": torch.empty(0, device=self.device, dtype=self.dtype),
            "acceptance_rate": torch.tensor(0.0, device=self.device, dtype=self.dtype),
        }
        # raise NotImplementedError

    def to(self, device: Union[str, torch.device]) -> "BaseSampler":
        """Move sampler to specified device."""
        self.device = device
        return self
