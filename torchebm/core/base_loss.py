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
from typing import Tuple, Union, Optional, Dict, Any, Callable

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

    Args:
        dtype: Data type for computations
        device: Device for computations
        use_mixed_precision: Whether to use mixed precision training (requires PyTorch 1.6+)
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        use_mixed_precision: bool = False,
    ):
        """Initialize the base loss class."""
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)
        self.dtype = dtype
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.use_mixed_precision = use_mixed_precision

        # Check if mixed precision is available
        if self.use_mixed_precision:
            try:
                from torch.cuda.amp import autocast

                self.autocast_available = True
            except ImportError:
                warnings.warn(
                    "Mixed precision training requested but torch.cuda.amp not available. "
                    "Falling back to full precision. Requires PyTorch 1.6+.",
                    UserWarning,
                )
                self.use_mixed_precision = False
                self.autocast_available = False
        else:
            self.autocast_available = False

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

    def to(
        self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None
    ) -> "BaseLoss":
        """
        Move the loss function to the specified device and optionally change its dtype.

        Args:
            device: Target device for computations.
            dtype: Optional data type to convert to.

        Returns:
            The loss function instance moved to the specified device/dtype.
        """
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        return super().to(device)

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
        # Ensure x is on the correct device and has the correct dtype
        x = x.to(device=self.device, dtype=self.dtype)

        # Apply mixed precision context if enabled
        if (
            hasattr(self, "use_mixed_precision")
            and self.use_mixed_precision
            and self.autocast_available
        ):
            from torch.cuda.amp import autocast

            with autocast():
                return self.forward(x, *args, **kwargs)
        else:
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
        new_sample_ratio: Ratio of new samples (default 5%). Adds noise to a fraction of buffer samples for exploration
        init_steps: Number of steps to run when initializing new chain elements
        dtype: Data type for computations
        device: Device for computations
        use_mixed_precision: Whether to use mixed precision training (requires PyTorch 1.6+)
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
        use_mixed_precision: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            dtype=dtype, device=device, use_mixed_precision=use_mixed_precision
        )
        self.energy_function = energy_function
        self.sampler = sampler
        self.k_steps = k_steps
        self.persistent = persistent
        self.buffer_size = buffer_size
        self.new_sample_ratio = new_sample_ratio
        self.init_steps = init_steps

        # Move components to the specified device
        self.energy_function = self.energy_function.to(device=self.device)
        if hasattr(self.sampler, "to") and callable(getattr(self.sampler, "to")):
            self.sampler = self.sampler.to(device=self.device)

        # for replay buffer
        self.register_buffer("replay_buffer", None)
        self.register_buffer(
            "buffer_ptr", torch.tensor(0, dtype=torch.long, device=self.device)
        )
        self.buffer_initialized = False

    def initialize_buffer(
        self,
        data_shape_no_batch: Tuple[int, ...],
        buffer_chunk_size: int = 1024,
        init_noise_scale: float = 0.01,
    ) -> torch.Tensor:
        """
        Initialize the replay buffer with random noise.

        For persistent CD variants, this method initializes the replay buffer
        with random noise. This is typically called the first time the loss
        is computed or when the batch size changes.

        Args:
            data_shape_no_batch: Shape of the data excluding batch dimension.
            buffer_chunk_size: Size of the chunks to process during initialization.
            init_noise_scale: Scale of the initial noise.

        Returns:
            The initialized buffer.
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

        # Initialize with small noise for better starting positions
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
                        # Apply mixed precision context if enabled
                        if self.use_mixed_precision and self.autocast_available:
                            from torch.cuda.amp import autocast

                            with autocast():
                                updated_chunk = self.sampler.sample(
                                    x=current_chunk, n_steps=self.init_steps
                                ).detach()
                        else:
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
                        warnings.warn(
                            f"Error during buffer initialization sampling for chunk {i}-{end}: {e}. Keeping noise for this chunk."
                        )
                        # Handle potential sampler errors during init

        # Reset pointer and mark as initialized
        self.buffer_ptr.zero_()
        self.buffer_initialized = True
        print(f"Replay buffer initialized.")

        return self.replay_buffer

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
        # Ensure x is on the correct device and has the correct dtype
        x = x.to(device=self.device, dtype=self.dtype)

        batch_size = x.shape[0]
        data_shape_no_batch = x.shape[1:]

        if self.persistent:
            # Initialize buffer if it hasn't been done yet
            if not self.buffer_initialized:
                self.initialize_buffer(data_shape_no_batch)
                # Check again if initialization failed silently (e.g., buffer_size=0)
                if not self.buffer_initialized:
                    raise RuntimeError("Buffer initialization failed.")

            # Sample random indices from the buffer
            if self.buffer_size < batch_size:
                warnings.warn(
                    f"Buffer size ({self.buffer_size}) is smaller than batch size ({batch_size}). Sampling with replacement.",
                    UserWarning,
                )
                indices = torch.randint(
                    0, self.buffer_size, (batch_size,), device=self.device
                )
            else:
                # Use stratified sampling to ensure better coverage of the buffer
                stride = self.buffer_size // batch_size
                base_indices = torch.arange(0, batch_size, device=self.device) * stride
                offset = torch.randint(0, stride, (batch_size,), device=self.device)
                indices = (base_indices + offset) % self.buffer_size

            # Retrieve samples and detach
            start_points = self.replay_buffer[indices].detach().clone()

            # Optional noise injection for exploration
            if self.new_sample_ratio > 0.0:
                n_new = max(1, int(batch_size * self.new_sample_ratio))
                noise_indices = torch.randperm(batch_size, device=self.device)[:n_new]
                noise_scale = 0.01  # Small noise scale
                start_points[noise_indices] = (
                    start_points[noise_indices]
                    + torch.randn_like(
                        start_points[noise_indices],
                        device=self.device,
                        dtype=self.dtype,
                    )
                    * noise_scale
                )
        else:
            # For standard CD-k, use the data points as starting points
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
        """Update the replay buffer with new samples using FIFO strategy.

        Args:
            samples: New samples to add to the buffer.
        """
        if not self.persistent or not self.buffer_initialized:
            return

        # Ensure samples are on the correct device and dtype
        samples = samples.to(device=self.device, dtype=self.dtype).detach()

        batch_size = samples.shape[0]

        # FIFO update strategy - replace oldest samples first
        ptr = int(self.buffer_ptr.item())
        # Handle the case where batch_size > buffer_size
        if batch_size >= self.buffer_size:
            # If batch is larger than buffer, just use the latest samples
            self.replay_buffer[:] = samples[-self.buffer_size :].detach()
            self.buffer_ptr[...] = 0  # Use ellipsis to set value without indexing
        else:
            # Calculate indices to be replaced (handling buffer wraparound)
            end_ptr = (ptr + batch_size) % self.buffer_size

            if end_ptr > ptr:
                # No wraparound
                self.replay_buffer[ptr:end_ptr] = samples.detach()
            else:
                # Handle wraparound - split the update into two parts
                first_part = self.buffer_size - ptr
                self.replay_buffer[ptr:] = samples[:first_part].detach()
                self.replay_buffer[:end_ptr] = samples[first_part:].detach()

            # Update pointer - use item assignment instead of indexing
            self.buffer_ptr[...] = end_ptr

    def to(
        self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None
    ) -> "BaseContrastiveDivergence":
        """
        Move the loss function to the specified device and optionally change its dtype.

        Args:
            device: Target device for computations
            dtype: Optional data type to convert to

        Returns:
            The loss function instance moved to the specified device/dtype
        """
        # Call parent method
        super().to(device, dtype)

        # Move components to the specified device
        self.energy_function = self.energy_function.to(device=self.device)
        if hasattr(self.sampler, "to") and callable(getattr(self.sampler, "to")):
            self.sampler = self.sampler.to(device=self.device)

        return self

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


class BaseScoreMatching(BaseLoss):
    """
    Abstract base class for Score Matching based loss functions.

    Score Matching is a family of methods for training energy-based models that
    avoid the sampling problem by directly matching the score function (gradient of log density).
    This class provides the common structure for different score matching variants including
    original score matching (Hyvärinen), denoising score matching, sliced score matching, etc.

    Methods:
        - forward: Computes the score matching loss given data samples
        - compute_score: Computes score function (gradient of energy w.r.t. input)
        - compute_loss: Computes the specific score matching loss variant

    Args:
        energy_function: The energy function being trained
        noise_scale: Scale of noise for perturbation (used in denoising variants)
        regularization_strength: Coefficient for regularization terms
        use_autograd: Whether to use PyTorch autograd for computing derivatives
        hutchinson_samples: Number of random samples for Hutchinson's trick in stochastic variants
        custom_regularization: Optional function for custom regularization (signature: f(loss, x, energy_fn) -> loss)
        use_mixed_precision: Whether to use mixed precision training (requires PyTorch 1.6+)
        dtype: Data type for computations
        device: Device for computations
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        energy_function: BaseEnergyFunction,
        noise_scale: float = 0.01,
        regularization_strength: float = 0.0,
        use_autograd: bool = True,
        hutchinson_samples: int = 1,
        custom_regularization: Optional[Callable] = None,
        use_mixed_precision: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            dtype=dtype, device=device, use_mixed_precision=use_mixed_precision
        )
        self.energy_function = energy_function.to(device=self.device)
        self.noise_scale = noise_scale
        self.regularization_strength = regularization_strength
        self.use_autograd = use_autograd
        self.hutchinson_samples = hutchinson_samples
        self.custom_regularization = custom_regularization
        self.use_mixed_precision = use_mixed_precision

        # Move energy function to specified device and dtype
        # self.energy_function = self.energy_function

        # Check if mixed precision is available
        if self.use_mixed_precision:
            try:
                from torch.cuda.amp import autocast

                self.autocast_available = True
            except ImportError:
                warnings.warn(
                    "Mixed precision training requested but torch.cuda.amp not available. "
                    "Falling back to full precision. Requires PyTorch 1.6+.",
                    UserWarning,
                )
                self.use_mixed_precision = False
                self.autocast_available = False
        else:
            self.autocast_available = False

    def compute_score(
        self, x: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the score function (gradient of energy function w.r.t. input).

        Args:
            x: Input data tensor
            noise: Optional noise tensor for perturbed variants

        Returns:
            torch.Tensor: The score function evaluated at x
        """
        # Ensure x is on the correct device and has the correct dtype
        x = x.to(device=self.device, dtype=self.dtype)

        if noise is not None:
            noise = noise.to(device=self.device, dtype=self.dtype)
            x_perturbed = x + noise
        else:
            x_perturbed = x

        if not x_perturbed.requires_grad:
            x_perturbed.requires_grad_(True)

        # Apply mixed precision context if enabled
        if self.use_mixed_precision and self.autocast_available:
            from torch.cuda.amp import autocast

            with autocast():
                energy = self.energy_function(x_perturbed)
        else:
            energy = self.energy_function(x_perturbed)

        if self.use_autograd:
            # Compute gradient using autograd
            score = torch.autograd.grad(energy.sum(), x_perturbed, create_graph=True)[0]
        else:
            # Allow subclasses to implement custom gradient computation
            raise NotImplementedError(
                "Custom gradient computation must be implemented in subclasses"
            )

        return score

    def perturb_data(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perturb the input data with Gaussian noise for denoising variants.

        Args:
            x: Input data tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Perturbed data and noise tensor
        """
        # Ensure x is on the correct device and has the correct dtype
        x = x.to(device=self.device, dtype=self.dtype)
        noise = (
            torch.randn_like(x, device=self.device, dtype=self.dtype) * self.noise_scale
        )
        x_perturbed = x + noise
        return x_perturbed, noise

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Call the forward method of the loss function.

        Args:
            x: Input data tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            torch.Tensor: The computed loss
        """
        # Ensure x is on the correct device
        x = x.to(device=self.device, dtype=self.dtype)
        return self.forward(x, *args, **kwargs)

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Compute the score matching loss given input data.

        Args:
            x: Input data tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            torch.Tensor: The computed score matching loss
        """
        pass

    @abstractmethod
    def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Compute the specific score matching loss variant.

        Args:
            x: Input data tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            torch.Tensor: The specific score matching loss
        """
        pass

    def add_regularization(
        self,
        loss: torch.Tensor,
        x: torch.Tensor,
        custom_reg_fn: Optional[Callable] = None,
        reg_strength: Optional[float] = None,
    ) -> torch.Tensor:
        """Add regularization terms to the loss.

        Args:
            loss: Current loss value
            x: Input tensor
            custom_reg_fn: Optional custom regularization function with signature f(x, energy_fn) -> tensor
            reg_strength: Optional regularization strength (overrides self.regularization_strength)

        Returns:
            regularized_loss: Loss with regularization
        """
        strength = (
            reg_strength if reg_strength is not None else self.regularization_strength
        )

        if strength <= 0:
            return loss

        # Use custom regularization if provided as parameter
        if custom_reg_fn is not None:
            reg_term = custom_reg_fn(x, self.energy_function)
        # Use class-level custom regularization if available
        elif self.custom_regularization is not None:
            reg_term = self.custom_regularization(x, self.energy_function)
        # Default regularization: L2 norm of score magnitude
        else:
            score = self.compute_score(x)
            reg_term = score.pow(2).sum(dim=list(range(1, len(x.shape)))).mean()

        return loss + strength * reg_term

    def to(
        self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None
    ) -> "BaseScoreMatching":
        """
        Move the loss function to the specified device and optionally change its dtype.

        Args:
            device: Target device for computations
            dtype: Optional data type to convert to

        Returns:
            The loss function instance moved to the specified device/dtype
        """
        self.device = device
        if dtype is not None:
            self.dtype = dtype

        # Move energy function
        self.energy_function = self.energy_function.to(device=self.device)

        return self

    def __repr__(self):
        """Return a string representation of the loss function."""
        return f"{self.__class__.__name__}(energy_function={self.energy_function})"

    def __str__(self):
        """Return a string representation of the loss function."""
        return self.__repr__()
