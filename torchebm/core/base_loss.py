"""
Base Loss Classes for Energy-Based Models

This module provides abstract base classes for defining loss functions for training
energy-based models (EBMs). It includes the general BaseLoss class for arbitrary loss
functions and the more specialized BaseContrastiveDivergence for contrastive divergence
based training methods.

Loss functions in TorchEBM are designed to work with energy functions and samplers
to define the training objective for energy-based models.
"""

from abc import abstractmethod, ABC
from typing import Tuple, Union, Optional, Dict, Any

import torch
from torch import nn

from torchebm.core import BaseEnergyFunction
from torchebm.core import BaseSampler


class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for loss functions used in energy-based models.

    This class builds on torch.nn.Module to allow loss functions to be part of PyTorch's
    computational graph and have trainable parameters if needed. It serves as the foundation
    for all loss functions in TorchEBM.

    Inheriting from torch.nn.Module ensures compatibility with PyTorch's training
    infrastructure, including device placement, parameter management, and gradient
    computation.

    Subclasses must implement the forward method to define the loss computation.
    """

    def __init__(self):
        """Initialize the base loss class."""
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Compute the loss value given input data.

        Args:
            x: Input data tensor, typically real samples from the target distribution.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed scalar loss value.
        """
        pass

    def to(self, device):
        """
        Move the loss function to the specified device.

        Args:
            device: Target device for computations.

        Returns:
            The loss function instance moved to the specified device.
        """
        self.device = device
        return self


class BaseContrastiveDivergence(BaseLoss):
    """
    Abstract base class for Contrastive Divergence (CD) based loss functions.

    Contrastive Divergence is a family of methods for training energy-based models that
    approximate the gradient of the log-likelihood by comparing the energy between real
    data samples (positive phase) and model samples (negative phase) generated through
    MCMC sampling.

    This class provides the common structure for CD variants, including standard CD,
    Persistent CD (PCD), and others.

    Attributes:
        energy_function: The energy function being trained
        sampler: MCMC sampler for generating negative samples
        n_steps: Number of MCMC steps to perform for each update
        persistent: Whether to use persistent chains (PCD)
        dtype: Data type for computations
        device: Device for computations
        chain: Buffer for persistent chains (when using PCD)
    """

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
        """
        Initialize the ContrastiveDivergence loss.

        Args:
            energy_function: Energy function to train
            sampler: MCMC sampler for generating negative samples
            n_steps: Number of MCMC steps for generating negative samples
            persistent: Whether to use persistent CD (maintain chains between updates)
            dtype: Data type for computations
            device: Device for computations
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.energy_function = energy_function
        self.sampler = sampler
        self.n_steps = n_steps
        self.persistent = persistent
        self.dtype = dtype
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Register buffer for persistent chains (if using PCD)
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

        This method should implement the specifics of the contrastive divergence
        variant, typically:
        1. Generate negative samples using the MCMC sampler
        2. Compute energies for real and negative samples
        3. Calculate the contrastive loss

        Args:
            x: Real data samples (positive samples).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - loss: The contrastive divergence loss
                - pred_x: Generated negative samples
        """
        pass

    def initialize_persistent_chain(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Initialize the persistent chain with random noise.

        For persistent CD variants, this method initializes the persistent chain
        buffer with random noise. This is typically called the first time the loss
        is computed or when the batch size changes.

        Args:
            shape: Shape of the initial chain state.

        Returns:
            The initialized chain.
        """

        if self.chain is None or self.chain.shape != shape:
            self.chain = torch.randn(*shape, dtype=self.dtype, device=self.device)

        return self.chain

    @abstractmethod
    def compute_loss(
        self, x: torch.Tensor, pred_x: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Compute the contrastive divergence loss from positive and negative samples.

        This method defines how the loss is calculated given real samples (positive phase)
        and samples from the model (negative phase). Typical implementations compute
        the difference between mean energies of positive and negative samples.

        Args:
            x: Real data samples (positive samples).
            pred_x: Generated negative samples.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The contrastive divergence loss
        """
        pass

    def to(self, device: Union[str, torch.device]) -> "BaseContrastiveDivergence":
        """
        Move loss to specified device.

        Args:
            device: Target device for computations.

        Returns:
            The loss function instance moved to the specified device.
        """
        self.device = device
        return self
