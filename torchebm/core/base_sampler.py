from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from torchebm.core import Schedulable, TorchEBMModule


class BaseSampler(Schedulable, TorchEBMModule, ABC):
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
