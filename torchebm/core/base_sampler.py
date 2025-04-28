from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, List, Dict
import warnings

import torch

from torchebm.core import BaseEnergyFunction, BaseScheduler


class BaseSampler(ABC):
    """
    Base class for samplers.

    Args:
        energy_function (BaseEnergyFunction): Energy function to sample from.
        dtype (torch.dtype): Data type to use for the computations.
        device (Union[str, torch.device]): Device to run the computations on (e.g., "cpu" or "cuda").
        use_mixed_precision (bool): Whether to use mixed precision for sampling operations.

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
        use_mixed_precision: bool = False,
    ):
        self.energy_function = energy_function
        self.dtype = dtype
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = use_mixed_precision
        
        # Check if mixed precision is available
        if self.use_mixed_precision:
            try:
                from torch.cuda.amp import autocast
                self.autocast_available = True
                # Ensure device is CUDA for mixed precision
                if not self.device.type.startswith('cuda'):
                    warnings.warn(
                        f"Mixed precision requested but device is {self.device}. "
                        f"Mixed precision requires CUDA. Falling back to full precision.",
                        UserWarning,
                    )
                    self.use_mixed_precision = False
                    self.autocast_available = False
            except ImportError:
                warnings.warn(
                    "Mixed precision requested but torch.cuda.amp not available. "
                    "Falling back to full precision. Requires PyTorch 1.6+.",
                    UserWarning,
                )
                self.use_mixed_precision = False
                self.autocast_available = False
        else:
            self.autocast_available = False

        self.schedulers: Dict[str, BaseScheduler] = {}
        
        # Ensure the energy function has matching precision settings
        if hasattr(self.energy_function, 'use_mixed_precision'):
            self.energy_function.use_mixed_precision = self.use_mixed_precision

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

    def get_schedulers(self) -> Dict[str, BaseScheduler]:
        """
        Get all registered schedulers.

        Returns:
            Dictionary mapping parameter names to their schedulers
        """
        return self.schedulers

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

    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None) -> "BaseSampler":
        """
        Move sampler to the specified device and optionally change its dtype.
        
        Args:
            device: Target device for computations
            dtype: Optional data type to convert to
            
        Returns:
            The sampler instance moved to the specified device/dtype
        """
        if isinstance(device, str):
            device = torch.device(device)
            
        self.device = device
        
        if dtype is not None:
            self.dtype = dtype
            
        # Update mixed precision availability if device changed
        if self.use_mixed_precision and not self.device.type.startswith('cuda'):
            warnings.warn(
                f"Mixed precision active but moving to {self.device}. "
                f"Mixed precision requires CUDA. Disabling mixed precision.",
                UserWarning,
            )
            self.use_mixed_precision = False
            
        # Move energy function if it has a to method
        if hasattr(self.energy_function, "to") and callable(getattr(self.energy_function, "to")):
            self.energy_function = self.energy_function.to(device=self.device, dtype=self.dtype)
            
        return self
        
    def apply_mixed_precision(self, func):
        """
        Decorator to apply mixed precision context to a method.
        
        Args:
            func: Function to wrap with mixed precision
            
        Returns:
            Wrapped function with mixed precision support
        """
        def wrapper(*args, **kwargs):
            if self.use_mixed_precision and self.autocast_available:
                from torch.cuda.amp import autocast
                with autocast():
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
