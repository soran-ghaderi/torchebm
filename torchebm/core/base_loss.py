from abc import abstractmethod, ABC
from typing import Tuple, Union, Optional, Dict, Any

import torch
from torch import nn

from torchebm.core import BaseEnergyFunction
from torchebm.core import BaseSampler


class BaseLoss(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass

    def to(self, device):
        self.device = device
        return self


class BaseContrastiveDivergence(BaseLoss):

    def __init__(
        self,
        energy_function: BaseEnergyFunction,
        sampler: BaseSampler,
        n_steps: int = 1,
        persistent: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.energy_function = energy_function
        self.sampler = sampler
        self.n_steps = n_steps
        self.persistent = persistent
        self.dtype = dtype
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # self.chain = None  # For persistent CD
        self.register_buffer("chain", None)  # For persistent CD

    def __call__(self, x, *args, **kwargs):
        """
        Call the forward method of the loss function.

        Args:
            x: Real data samples (positive samples).
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            torch.Tensor: The computed loss.
        """
        return self.forward(x, *args, **kwargs)

    @abstractmethod
    def forward(
        self, x: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CD loss given real data samples.

        Args:
            x: Real data samples (positive samples).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - loss: The contrastive divergence loss
                - pred_x: Generated negative samples
        """
        pass

    def initialize_persistent_chain(self, shape: Tuple[int, ...]):
        """
        Initialize the persistent chain with random noise.

        Args:
            shape: Shape of the initial chain state.
        """

        if self.chain is None or self.chain.shape != shape:
            self.chain = torch.randn(*shape, dtype=self.dtype, device=self.device)

        return self.chain

    @abstractmethod
    def compute_loss(
        self, x: torch.Tensor, pred_x: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """

        Args:
            x: Real data samples (positive samples).
            pred_x: Generated negative samples.

        Returns:
            torch.Tensor: The contrastive divergence loss

        """

    def to(self, device: Union[str, torch.device]) -> "BaseContrastiveDivergence":
        """Move loss to specified device."""
        self.device = device
        return self
