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

    def to(self, device: Union[str, torch.device]) -> "BaseLoss":
        """
        Move the loss function to the specified device.

        Args:
            device: Target device for computations.

        Returns:
            The loss function instance moved to the specified device.
        """
        self.device = device
        return self

    def __repr__(self):
        """Return a string representation of the loss function."""
        return f"{self.__class__.__name__}()"

    def __str__(self):
        """Return a string representation of the loss function."""
        return self.__repr__()

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Call the forward method of the loss function.

        Args:
            x: Input data tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed loss value.
        """
        return self.forward(x, *args, **kwargs)


class BaseContrastiveDivergence(BaseLoss):
    """
    Abstract base class for Contrastive Divergence (CD) based loss functions.

    Contrastive Divergence is a family of methods for training energy-based models that
    approximate the gradient of the log-likelihood by comparing the energy between real
    data samples (positive phase) and model samples (negative phase) generated through
    MCMC sampling.

    This class provides the common structure for CD variants, including standard CD,
    Persistent CD (PCD), and others.

    Methods:
        - __call__: Calls the forward method of the loss function.
        - initialize_buffer: Initializes the replay buffer with random noise.
        - get_negative_samples: Generates negative samples using the replay buffer strategy.
        - update_buffer: Updates the replay buffer with new samples.
        - forward: Computes CD loss given real data samples.
        - compute_loss: Computes the contrastive divergence loss from positive and negative samples.

    Args:
        energy_function: The energy function being trained
        sampler: MCMC sampler for generating negative samples
        k_steps: Number of MCMC steps to perform for each update
        persistent: Whether to use replay buffer (PCD)
        buffer_size: Size of the buffer for storing replay buffer
        new_sample_ratio: Ratio of new samples (default 5%)
        init_steps: Number of steps to run when initializing new chain elements
        dtype: Data type for computations
        device: Device for computations
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        energy_function: BaseEnergyFunction,
        sampler: BaseSampler,
        k_steps: int = 1,
        persistent: bool = False,
        buffer_size: int = 100,
        new_sample_ratio: float = 0.05,
        init_steps: int = 0,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.energy_function = energy_function
        self.sampler = sampler
        self.k_steps = k_steps
        self.persistent = persistent
        self.buffer_size = buffer_size
        self.new_sample_ratio = new_sample_ratio
        self.init_steps = init_steps
        self.dtype = dtype
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # for replay buffer
        self.register_buffer("replay_buffer", None)
        self.register_buffer("buffer_ptr", torch.tensor(0, dtype=torch.long))
        self.buffer_initialized = False

    def initialize_buffer(
        self, batch_shape: Tuple[int, ...], buffer_chunk_size: int = 1024
    ) -> torch.Tensor:
        """
        Initialize the replay buffer with random noise.

        For persistent CD variants, this method initializes the replay buffer
        with random noise. This is typically called the first time the loss
        is computed or when the batch size changes.

        Args:
            batch_shape: Shape of the initial chain state.
            buffer_chunk_size: Size of the chunks to process during initialization.

        Returns:
            The initialized chain.
        """
        if not self.persistent:
            return

        if self.buffer_initialized:
            return self.replay_buffer

        if (
            self.replay_buffer is None
            or self.replay_buffer.batch_shape[1:] != batch_shape[1:]
        ):

            if self.buffer_size < batch_shape[0]:
                raise ValueError(
                    f"Replay buffer size {self.buffer_size} is smaller than batch size {batch_shape[0]}, please increase the replay buffer size. "
                    f"Hint: ContrastiveDivergence(...,buffer_size=BATCH_SIZE,...)."
                )
            buffer_shape = (self.buffer_size,) + tuple(
                batch_shape[1:]
            )  # shape: [buffer_size, *data_shape]
            self.replay_buffer = torch.randn(
                buffer_shape, dtype=self.dtype, device=self.device
            )
            self.buffer_ptr = torch.tensor(0, dtype=torch.long, device=self.device)

            if self.init_steps == 0:
                print(f"Buffer initialized with random noise (size {self.buffer_size})")

            elif self.init_steps > 0:
                print(f"Initializing buffer with {self.init_steps} MCMC steps...")
                with torch.no_grad():  # Make sure we don't track gradients
                    # Process in chunks to avoid memory issues
                    chunk_size = min(
                        self.buffer_size, buffer_chunk_size
                    )  # Adjust based on your GPU memory
                    for i in range(0, self.buffer_size, chunk_size):
                        end = min(i + chunk_size, self.buffer_size)
                        self.replay_buffer[i:end] = self.sampler.sample(
                            x=self.replay_buffer[i:end], n_steps=self.init_steps
                        ).detach()

            self.buffer_ptr = torch.tensor(0, dtype=torch.long, device=self.device)
            self.buffer_initialized = True
            print(f"Replay buffer initialized with {self.buffer_size} samples")

        return self.replay_buffer

    def get_negative_samples(self, batch_size, data_shape) -> torch.Tensor:
        """Get negative samples using the replay buffer strategy.

        Args:
            batch_size: Number of samples to generate.
            data_shape: Shape of the data samples (excluding batch size).

        Returns:
            torch.Tensor: Negative samples generated from the replay buffer.

        """
        if not self.persistent or not self.buffer_initialized:
            # For non-persistent CD, just return random noise
            return torch.randn(
                (batch_size,) + data_shape, dtype=self.dtype, device=self.device
            )

        # Calculate how many samples to draw from buffer vs. generate new
        n_new = max(1, int(batch_size * self.new_sample_ratio))
        n_old = batch_size - n_new

        # Create tensor to hold all samples
        all_samples = torch.empty(
            (batch_size,) + data_shape, dtype=self.dtype, device=self.device
        )

        # Generate new random samples (5%)
        if n_new > 0:
            all_samples[:n_new] = torch.randn(
                (n_new,) + data_shape, dtype=self.dtype, device=self.device
            )

        # Get samples from buffer (95%)
        if n_old > 0:
            # Choose random indices from buffer
            indices = torch.randint(0, self.buffer_size, (n_old,), device=self.device)
            all_samples[n_new:] = self.replay_buffer[indices]

        return all_samples

    def update_buffer(self, samples: torch.Tensor) -> None:
        """Update the replay buffer with new samples.

        Args:
            samples: New samples to add to the buffer.
        """
        if not self.persistent:
            return

        batch_size = samples.shape[0]

        # If the buffer isn't fully initialized yet, just append
        if not hasattr(self, "_buffer_filled_size"):
            self._buffer_filled_size = 0

        if self._buffer_filled_size < self.buffer_size:
            # Still filling the buffer
            end_idx = min(self._buffer_filled_size + batch_size, self.buffer_size)
            size_to_add = end_idx - self._buffer_filled_size

            self.replay_buffer[self._buffer_filled_size : end_idx] = samples[
                :size_to_add
            ]
            self._buffer_filled_size = end_idx
            self.buffer_ptr.fill_(self._buffer_filled_size % self.buffer_size)

            if self._buffer_filled_size >= self.buffer_size:
                print("Buffer fully filled with training samples")
            return

        # Update buffer with new samples
        ptr = int(self.buffer_ptr)
        if ptr + batch_size <= self.buffer_size:
            # If we can fit the entire batch
            self.replay_buffer[ptr : ptr + batch_size] = samples
            self.buffer_ptr.fill_((ptr + batch_size) % self.buffer_size)
        else:
            # Handle wrapping around the buffer
            space_left = self.buffer_size - ptr
            self.replay_buffer[ptr:] = samples[:space_left]
            self.replay_buffer[: batch_size - space_left] = samples[space_left:]
            self.buffer_ptr.fill_((ptr + batch_size) % self.buffer_size)

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

    def __repr__(self):
        """Return a string representation of the loss function."""
        return f"{self.__class__.__name__}(energy_function={self.energy_function}, sampler={self.sampler})"

    def __str__(self):
        """Return a string representation of the loss function."""
        return self.__repr__()
