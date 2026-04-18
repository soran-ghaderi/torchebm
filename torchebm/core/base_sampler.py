from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from torchebm.core import BaseScheduler, TorchEBMModule, safe_to


class BaseSampler(TorchEBMModule, ABC):
    """
    Abstract base class for samplers.

    Args:
        model (nn.Module): The model to sample from. For MCMC samplers, this is
            typically a `BaseModel` energy function; for learned samplers it may be
            any `nn.Module`.
        dtype (torch.dtype): The data type for computations.
        device (Optional[Union[str, torch.device]]): The device for computations.
    """

    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)
        self.model = model
        self.setup_mixed_precision(use_mixed_precision)

        self.schedulers: Dict[str, BaseScheduler] = {}

        # Align child components using the helper
        self.model = safe_to(
            self.model, device=self.device, dtype=self.dtype
        )

        # Ensure the energy function has matching precision settings
        if hasattr(self.model, "use_mixed_precision"):
            self.model.use_mixed_precision = self.use_mixed_precision

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
        reset_schedulers: bool = True,
        *args,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
        """
        Runs the sampling process.

        Args:
            x (Optional[torch.Tensor]): The initial state to start sampling from.
            dim (int): The dimension of the state space.
            n_steps (int): The number of MCMC steps to perform.
            n_samples (int): The number of samples to generate.
            thin (int): The thinning factor for samples (currently not supported).
            return_trajectory (bool): Whether to return the full trajectory of the samples.
            return_diagnostics (bool): Whether to return diagnostics of the sampling process.
            reset_schedulers (bool): If True (default), reset registered schedulers
                before sampling so each call starts from step 0. Pass False for
                lifetime schedules driven by an outer training loop.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
                - A tensor of samples from the model.
                - If `return_diagnostics` is `True`, a tuple containing the samples
                  and a list of diagnostics dictionaries.
        """
        raise NotImplementedError

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

    # def to(
    #     self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None
    # ) -> "BaseSampler":
    #     """
    #     Move sampler to the specified device and optionally change its dtype.
    #
    #     Args:
    #         device: Target device for computations
    #         dtype: Optional data type to convert to
    #
    #     Returns:
    #         The sampler instance moved to the specified device/dtype
    #     """
    #     if isinstance(device, str):
    #         device = torch.device(device)
    #
    #     self.device = device
    #
    #     if dtype is not None:
    #         self.dtype = dtype
    #
    #     # Update mixed precision availability if device changed
    #     if self.use_mixed_precision and not self.device.type.startswith("cuda"):
    #         warnings.warn(
    #             f"Mixed precision active but moving to {self.device}. "
    #             f"Mixed precision requires CUDA. Disabling mixed precision.",
    #             UserWarning,
    #         )
    #         self.use_mixed_precision = False
    #
    #     # Move energy function if it has a to method
    #     if hasattr(self.model, "to") and callable(
    #         getattr(self.model, "to")
    #     ):
    #         self.model = self.model.to(
    #             device=self.device, dtype=self.dtype
    #         )
    #
    #     return self

    def apply_mixed_precision(self, func):
        """
        A decorator to apply the mixed precision context to a method.

        Args:
            func: The function to wrap.

        Returns:
            The wrapped function.
        """

        def wrapper(*args, **kwargs):
            with self.autocast_context():
                return func(*args, **kwargs)

        return wrapper

    def to(self, *args, **kwargs):
        """Moves the sampler and its components to the specified device and/or dtype."""
        # Let DeviceMixin update internal state and parent class handle movement
        result = super().to(*args, **kwargs)
        # After move, make sure energy_function follows
        self.model = safe_to(
            self.model, device=self.device, dtype=self.dtype
        )
        return result
