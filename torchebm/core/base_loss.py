"""
Base Loss Classes for Energy-Based Models

This module provides abstract base classes for defining loss functions for training
energy-based models (EBMs). It includes the general BaseLoss class for arbitrary loss
functions and the more specialized BaseContrastiveDivergence for contrastive divergence
based training methods.

Loss functions in TorchEBM are designed to work with energy functions and samplers
to define the training objective for energy-based models.
"""

import warnings
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
        new_sample_ratio: float = 0.0,
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
        self,
        data_shape_no_batch: Tuple[int, ...],
        buffer_chunk_size: int = 1024,
        init_noise_scale: float = 0.1,
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
        if not self.persistent or self.buffer_initialized:
            return

        if self.buffer_size <= 0:
            raise ValueError(
                f"Replay buffer size must be positive, got {self.buffer_size}"
            )

        buffer_shape = (
            self.buffer_size,
        ) + data_shape_no_batch  # shape: [buffer_size, *data_shape]
        print(f"Initializing replay buffer with shape {buffer_shape}...")

        self.replay_buffer = (
            torch.randn(buffer_shape, dtype=self.dtype, device=self.device)
            * init_noise_scale
        )  # Start with small noise

        if self.init_steps > 0:
            print(f"Running {self.init_steps} MCMC steps to populate buffer...")
            with torch.no_grad():  # Make sure we don't track gradients
                # Process in chunks to avoid memory issues
                chunk_size = min(self.buffer_size, buffer_chunk_size)
                for i in range(0, self.buffer_size, chunk_size):
                    end = min(i + chunk_size, self.buffer_size)
                    current_chunk = self.replay_buffer[
                        i:end
                    ].clone()  # Sample from current state
                    try:
                        updated_chunk = self.sampler.sample(
                            x=current_chunk, n_steps=self.init_steps
                        ).detach()
                        # Ensure the output shape matches
                        if updated_chunk.shape == current_chunk.shape:
                            self.replay_buffer[i:end] = updated_chunk
                        else:
                            warnings.warn(
                                f"Sampler output shape mismatch during buffer init. Expected {current_chunk.shape}, got {updated_chunk.shape}. Skipping update for chunk {i}-{end}."
                            )
                    except Exception as e:
                        raise RuntimeError(
                            f"Error during buffer initialization sampling for chunk {i}-{end}: {e}. Keeping noise for this chunk."
                        )
                        # Handle potential sampler errors during init

        # Reset pointer and mark as initialized
        self.buffer_ptr.zero_()
        self.buffer_initialized = True
        print(f"Replay buffer initialized.")

        # if (
        #     self.replay_buffer is None
        #     or self.replay_buffer.batch_shape[1:] != batch_shape[1:]
        # ):
        #
        #     if self.buffer_size < batch_shape[0]:
        #         raise ValueError(
        #             f"Replay buffer size {self.buffer_size} is smaller than batch size {batch_shape[0]}, please increase the replay buffer size. "
        #             f"Hint: ContrastiveDivergence(...,buffer_size=BATCH_SIZE,...)."
        #         )
        #     buffer_shape = (self.buffer_size,) + tuple(
        #         batch_shape[1:]
        #     )  # shape: [buffer_size, *data_shape]
        #     self.replay_buffer = torch.randn(
        #         buffer_shape, dtype=self.dtype, device=self.device
        #     )
        #     self.buffer_ptr = torch.tensor(0, dtype=torch.long, device=self.device)
        #
        #     if self.init_steps == 0:
        #         print(f"Buffer initialized with random noise (size {self.buffer_size})")

        #     elif self.init_steps > 0:
        #         print(f"Initializing buffer with {self.init_steps} MCMC steps...")
        #         with torch.no_grad():  # Make sure we don't track gradients
        #             # Process in chunks to avoid memory issues
        #             chunk_size = min(
        #                 self.buffer_size, buffer_chunk_size
        #             )  # Adjust based on your GPU memory
        #             for i in range(0, self.buffer_size, chunk_size):
        #                 end = min(i + chunk_size, self.buffer_size)
        #                 self.replay_buffer[i:end] = self.sampler.sample(
        #                     x=self.replay_buffer[i:end], n_steps=self.init_steps
        #                 ).detach()
        #
        #     self.buffer_ptr = torch.tensor(0, dtype=torch.long, device=self.device)
        #     self.buffer_initialized = True
        #     print(f"Replay buffer initialized with {self.buffer_size} samples")
        #
        # return self.replay_buffer

    def get_start_points(self, x: torch.Tensor) -> torch.Tensor:
        """Gets the starting points for the MCMC sampler.

        Handles both persistent (PCD) and non-persistent (CD-k) modes.
        Initializes the buffer for PCD on the first call if needed.

        Args:
            x (torch.Tensor): The input data batch. Used directly for non-persistent CD
                              and for shape inference/initialization trigger for PCD.

        Returns:
            torch.Tensor: The tensor of starting points for the sampler.
        """
        batch_size = x.shape[0]
        data_shape_no_batch = x.shape[1:]

        if self.persistent:
            # Initialize buffer if it hasn't been done yet
            if not self.buffer_initialized:
                self.initialize_buffer(data_shape_no_batch)
                # Check again if initialization failed silently (e.g., buffer_size=0)
                if not self.buffer_initialized:
                    raise RuntimeError("Buffer initialization failed.")

            # --- Standard PCD Logic ---
            # Sample random indices from the buffer
            # Ensure buffer_size is at least batch_size for this simple sampling.
            # A more robust approach might sample with replacement if buffer < batch_size,
            # but ideally buffer should be much larger. Let's check here.
            if self.buffer_size < batch_size:
                warnings.warn(
                    f"Buffer size ({self.buffer_size}) is smaller than batch size ({batch_size}). Sampling with replacement.",
                    UserWarning,
                )

                indices = torch.randint(
                    0, self.buffer_size, (batch_size,), device=self.device
                )
            else:
                # Sample without replacement if buffer is large enough (though randint samples with replacement by default)
                # To be precise: sample random indices
                indices = torch.randint(
                    0, self.buffer_size, (batch_size,), device=self.device
                )

            # Retrieve samples and detach
            start_points = self.replay_buffer[indices].detach().clone()

            # --- Optional: Noise Injection (Less standard for START points) ---
            if self.new_sample_ratio > 0.0:
                n_new = max(1, int(batch_size * self.new_sample_ratio))
                noise_indices = torch.randperm(batch_size, device=self.device)[:n_new]
                start_points[noise_indices] = torch.randn_like(
                    start_points[noise_indices]
                )

        else:
            # --- Standard Non-Persistent CD Logic ---
            start_points = x.detach().clone()

        return start_points

    def get_negative_samples(self, x, batch_size, data_shape) -> torch.Tensor:
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
        if not self.persistent or not self.buffer_initialized:
            return

        batch_size = samples.shape[0]
        samples = samples.detach()

        # If the buffer isn't fully initialized yet, just append
        # if not hasattr(self, "_buffer_filled_size"):
        #     self._buffer_filled_size = 0

        # if self._buffer_filled_size < self.buffer_size:
        #     # Still filling the buffer
        #     end_idx = min(self._buffer_filled_size + batch_size, self.buffer_size)
        #     size_to_add = end_idx - self._buffer_filled_size
        #
        #     self.replay_buffer[self._buffer_filled_size : end_idx] = samples[
        #         :size_to_add
        #     ]
        #     self._buffer_filled_size = end_idx
        #     self.buffer_ptr.fill_(self._buffer_filled_size % self.buffer_size)
        #
        #     if self._buffer_filled_size >= self.buffer_size:
        #         print("Buffer fully filled with training samples")
        #     return

        indices_to_replace = torch.randint(
            0, self.buffer_size, (batch_size,), device=self.device
        )

        # Handle potential size mismatch if batch_size > buffer_size (should be rare with checks)
        num_to_replace = min(batch_size, self.buffer_size)
        if num_to_replace < batch_size:
            warnings.warn(
                f"Replacing only {num_to_replace} buffer samples as batch size ({batch_size}) > buffer size ({self.buffer_size})",
                UserWarning,
            )
            indices_to_replace = indices_to_replace[:num_to_replace]
            samples_to_insert = samples[:num_to_replace]
        else:
            samples_to_insert = samples

        self.replay_buffer[indices_to_replace] = samples_to_insert

        # this uses FIFO logic, maybe receive a param whether to use this strategy  or the random slots above
        # # Update buffer with new samples
        # ptr = int(self.buffer_ptr)
        # if ptr + batch_size <= self.buffer_size:
        #     # If we can fit the entire batch
        #     self.replay_buffer[ptr : ptr + batch_size] = samples
        #     self.buffer_ptr.fill_((ptr + batch_size) % self.buffer_size)
        # else:
        #     # Handle wrapping around the buffer
        #     space_left = self.buffer_size - ptr
        #     self.replay_buffer[ptr:] = samples[:space_left]
        #     self.replay_buffer[: batch_size - space_left] = samples[space_left:]
        #     self.buffer_ptr.fill_((ptr + batch_size) % self.buffer_size)

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
