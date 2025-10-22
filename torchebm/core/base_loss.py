"""
Base Loss Classes for Energy-Based Models
"""

import warnings
from abc import abstractmethod, ABC
from typing import Tuple, Union, Optional, Dict, Any, Callable

import torch
from torch import nn

from torchebm.core import BaseModel
from torchebm.core import BaseSampler
from torchebm.core import DeviceMixin


class BaseLoss(DeviceMixin, nn.Module, ABC):
    """
    Abstract base class for loss functions used in energy-based models.

    Args:
        dtype (torch.dtype): Data type for computations.
        device (Optional[Union[str, torch.device]]): Device for computations.
        use_mixed_precision (bool): Whether to use mixed precision training.
        clip_value (Optional[float]): Optional value to clamp the loss.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        use_mixed_precision: bool = False,
        clip_value: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the base loss class."""
        super().__init__(device=device, *args, **kwargs)

        # if isinstance(device, str):
        #     device = torch.device(device)
        self.dtype = dtype
        self.clip_value = clip_value
        self.setup_mixed_precision(use_mixed_precision)


    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Computes the loss value.

        Args:
            x (torch.Tensor): Input data tensor from the target distribution.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed scalar loss value.
        """
        pass

    def __repr__(self):
        """Return a string representation of the loss function."""
        return f"{self.__class__.__name__}()"

    def __str__(self):
        """Return a string representation of the loss function."""
        return self.__repr__()

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Calls the forward method of the loss function.

        Args:
            x (torch.Tensor): Input data tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed loss value.
        """
        x = x.to(device=self.device, dtype=self.dtype)

        with self.autocast_context():
            loss = self.forward(x, *args, **kwargs)

        if self.clip_value:
            loss = torch.clamp(loss, -self.clip_value, self.clip_value)
        return loss


class BaseContrastiveDivergence(BaseLoss):
    """
    Abstract base class for Contrastive Divergence (CD) based loss functions.

    Args:
        model (BaseModel): The energy-based model to be trained.
        sampler (BaseSampler): The MCMC sampler for generating negative samples.
        k_steps (int): The number of MCMC steps to perform for each update.
        persistent (bool): If `True`, uses a replay buffer for Persistent CD (PCD).
        buffer_size (int): The size of the replay buffer for PCD.
        new_sample_ratio (float): The ratio of new random samples to introduce into the MCMC chain.
        init_steps (int): The number of MCMC steps to run when initializing new chain elements.
        dtype (torch.dtype): Data type for computations.
        device (Optional[Union[str, torch.device]]): Device for computations.
        use_mixed_precision (bool): Whether to use mixed precision training.
        clip_value (Optional[float]): Optional value to clamp the loss.
    """

    def __init__(
        self,
        model: BaseModel,
        sampler: BaseSampler,
        k_steps: int = 1,
        persistent: bool = False,
        buffer_size: int = 100,
        new_sample_ratio: float = 0.0,
        init_steps: int = 0,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        use_mixed_precision: bool = False,
        clip_value: Optional[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            dtype=dtype,
            device=device,
            use_mixed_precision=use_mixed_precision,
            clip_value=clip_value,
            *args,
            **kwargs,
        )
        self.model = model
        self.sampler = sampler
        self.k_steps = k_steps
        self.persistent = persistent
        self.buffer_size = buffer_size
        self.new_sample_ratio = new_sample_ratio
        self.init_steps = init_steps

        self.model = self.model.to(device=self.device)
        if hasattr(self.sampler, "to") and callable(getattr(self.sampler, "to")):
            self.sampler = self.sampler.to(device=self.device)

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
        Initializes the replay buffer with random noise for PCD.

        Args:
            data_shape_no_batch (Tuple[int, ...]): The shape of the data excluding the batch dimension.
            buffer_chunk_size (int): The size of chunks to process during initialization.
            init_noise_scale (float): The scale of the initial noise.

        Returns:
            torch.Tensor: The initialized replay buffer.
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
        )

        if self.init_steps > 0:
            print(f"Running {self.init_steps} MCMC steps to populate buffer...")
            with torch.no_grad():
                chunk_size = min(self.buffer_size, buffer_chunk_size)
                for i in range(0, self.buffer_size, chunk_size):
                    end = min(i + chunk_size, self.buffer_size)
                    current_chunk = self.replay_buffer[i:end].clone()
                    try:
                        with self.autocast_context():
                            updated_chunk = self.sampler.sample(
                                x=current_chunk, n_steps=self.init_steps
                            ).detach()

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

        self.buffer_ptr.zero_()
        self.buffer_initialized = True
        print(f"Replay buffer initialized.")

        return self.replay_buffer

    def get_start_points(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gets the starting points for the MCMC sampler.

        For standard CD, this is the input data. For PCD, it's samples from the replay buffer.

        Args:
            x (torch.Tensor): The input data batch.

        Returns:
            torch.Tensor: The tensor of starting points for the sampler.
        """
        x = x.to(device=self.device, dtype=self.dtype)

        batch_size = x.shape[0]
        data_shape_no_batch = x.shape[1:]

        if self.persistent:
            if not self.buffer_initialized:
                self.initialize_buffer(data_shape_no_batch)
                if not self.buffer_initialized:
                    raise RuntimeError("Buffer initialization failed.")

            if self.buffer_size < batch_size:
                warnings.warn(
                    f"Buffer size ({self.buffer_size}) is smaller than batch size ({batch_size}). Sampling with replacement.",
                    UserWarning,
                )
                indices = torch.randint(
                    0, self.buffer_size, (batch_size,), device=self.device
                )
            else:
                # stratified sampling for better buffer coverage
                stride = self.buffer_size // batch_size
                base_indices = torch.arange(0, batch_size, device=self.device) * stride
                offset = torch.randint(0, stride, (batch_size,), device=self.device)
                indices = (base_indices + offset) % self.buffer_size

            start_points = self.replay_buffer[indices].detach().clone()

            # add some noise for exploration
            if self.new_sample_ratio > 0.0:
                n_new = max(1, int(batch_size * self.new_sample_ratio))
                noise_indices = torch.randperm(batch_size, device=self.device)[:n_new]
                noise_scale = 0.01
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
            # standard CD-k uses data as starting points
            start_points = x.detach().clone()

        return start_points

    def get_negative_samples(self, x, batch_size, data_shape) -> torch.Tensor:
        """
        Gets negative samples using the replay buffer strategy.

        Args:
            x: (Unused) The input data tensor.
            batch_size (int): The number of samples to generate.
            data_shape (Tuple[int, ...]): The shape of the data samples (excluding batch size).

        Returns:
            torch.Tensor: Negative samples.
        """
        if not self.persistent or not self.buffer_initialized:
            # For non-persistent CD, just return random noise
            return torch.randn(
                (batch_size,) + data_shape, dtype=self.dtype, device=self.device
            )

        n_new = max(1, int(batch_size * self.new_sample_ratio))
        n_old = batch_size - n_new

        all_samples = torch.empty(
            (batch_size,) + data_shape, dtype=self.dtype, device=self.device
        )

        # new random samples
        if n_new > 0:
            all_samples[:n_new] = torch.randn(
                (n_new,) + data_shape, dtype=self.dtype, device=self.device
            )

        # samples from buffer
        if n_old > 0:

            indices = torch.randint(0, self.buffer_size, (n_old,), device=self.device)
            all_samples[n_new:] = self.replay_buffer[indices]

        return all_samples

    def update_buffer(self, samples: torch.Tensor) -> None:
        """
        Updates the replay buffer with new samples using a FIFO strategy.

        Args:
            samples (torch.Tensor): New samples to add to the buffer.
        """
        if not self.persistent or not self.buffer_initialized:
            return

        # Ensure samples are on the correct device and dtype
        samples = samples.to(device=self.device, dtype=self.dtype).detach()

        batch_size = samples.shape[0]

        # FIFO strategy
        ptr = int(self.buffer_ptr.item())

        if batch_size >= self.buffer_size:
            # batch larger than buffer, use latest samples
            self.replay_buffer[:] = samples[-self.buffer_size :].detach()
            self.buffer_ptr[...] = 0
        else:
            # handle buffer wraparound
            end_ptr = (ptr + batch_size) % self.buffer_size

            if end_ptr > ptr:
                self.replay_buffer[ptr:end_ptr] = samples.detach()
            else:
                # wraparound case - split update
                first_part = self.buffer_size - ptr
                self.replay_buffer[ptr:] = samples[:first_part].detach()
                self.replay_buffer[:end_ptr] = samples[first_part:].detach()

            self.buffer_ptr[...] = end_ptr

    @abstractmethod
    def forward(
        self, x: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the CD loss given real data samples.

        Args:
            x (torch.Tensor): Real data samples (positive samples).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - The contrastive divergence loss.
                - The generated negative samples.
        """
        pass

    @abstractmethod
    def compute_loss(
        self, x: torch.Tensor, pred_x: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Computes the contrastive divergence loss from positive and negative samples.

        Args:
            x (torch.Tensor): Real data samples (positive samples).
            pred_x (torch.Tensor): Generated negative samples.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The contrastive divergence loss.
        """
        pass

    def __repr__(self):
        """Return a string representation of the loss function."""
        return f"{self.__class__.__name__}(model={self.model}, sampler={self.sampler})"

    def __str__(self):
        """Return a string representation of the loss function."""
        return self.__repr__()


class BaseScoreMatching(BaseLoss):
    """
    Abstract base class for Score Matching based loss functions.

    Args:
        model (BaseModel): The energy-based model to be trained.
        noise_scale (float): The scale of noise for perturbation in denoising variants.
        regularization_strength (float): The coefficient for regularization terms.
        use_autograd (bool): Whether to use `torch.autograd` for computing derivatives.
        hutchinson_samples (int): The number of random samples for Hutchinson's trick.
        custom_regularization (Optional[Callable]): An optional function for custom regularization.
        use_mixed_precision (bool): Whether to use mixed precision training.
        clip_value (Optional[float]): Optional value to clamp the loss.
    """

    def __init__(
        self,
        model: BaseModel,
        noise_scale: float = 0.01,
        regularization_strength: float = 0.0,
        use_autograd: bool = True,
        hutchinson_samples: int = 1,
        custom_regularization: Optional[Callable] = None,
        use_mixed_precision: bool = False,
        clip_value: Optional[float] = None,
        # dtype: torch.dtype = torch.float32,
        # device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            use_mixed_precision=use_mixed_precision,
            clip_value=clip_value,
            *args,
            **kwargs,  # dtype=dtype, device=device,
        )
        self.model = model.to(device=self.device)
        self.noise_scale = noise_scale
        self.regularization_strength = regularization_strength
        self.use_autograd = use_autograd
        self.hutchinson_samples = hutchinson_samples
        self.custom_regularization = custom_regularization
        self.use_mixed_precision = use_mixed_precision

        self.model = self.model.to(device=self.device)

        self.setup_mixed_precision(use_mixed_precision)

    def compute_score(
        self, x: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the score function, \(\nabla_x E(x)\).

        Args:
            x (torch.Tensor): The input data tensor.
            noise (Optional[torch.Tensor]): Optional noise tensor for perturbed variants.

        Returns:
            torch.Tensor: The score function evaluated at `x` or `x + noise`.
        """

        x = x.to(device=self.device, dtype=self.dtype)

        if noise is not None:
            noise = noise.to(device=self.device, dtype=self.dtype)
            x_perturbed = x + noise
        else:
            x_perturbed = x

        if not x_perturbed.requires_grad:
            x_perturbed.requires_grad_(True)

        with self.autocast_context():
            energy = self.model(x_perturbed)

        if self.use_autograd:
            score = torch.autograd.grad(energy.sum(), x_perturbed, create_graph=True)[0]
        else:
            raise NotImplementedError(
                "Custom gradient computation must be implemented in subclasses"
            )

        return score

    def perturb_data(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # todo: add more noise types
        """
        Perturbs the input data with Gaussian noise for denoising variants.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the perturbed data
                and the noise that was added.
        """

        x = x.to(device=self.device, dtype=self.dtype)
        noise = (
            torch.randn_like(x, device=self.device, dtype=self.dtype) * self.noise_scale
        )
        x_perturbed = x + noise
        return x_perturbed, noise

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Calls the forward method of the loss function.

        Args:
            x (torch.Tensor): Input data tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed loss.
        """

        x = x.to(device=self.device, dtype=self.dtype)
        return self.forward(x, *args, **kwargs)

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Computes the score matching loss given input data.

        Args:
            x (torch.Tensor): Input data tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed score matching loss.
        """
        pass

    @abstractmethod
    def compute_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Computes the specific score matching loss variant.

        Args:
            x (torch.Tensor): Input data tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The specific score matching loss.
        """
        pass

    def add_regularization(
        self,
        loss: torch.Tensor,
        x: torch.Tensor,
        custom_reg_fn: Optional[Callable] = None,
        reg_strength: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Adds regularization terms to the loss.

        Args:
            loss (torch.Tensor): The current loss value.
            x (torch.Tensor): The input tensor.
            custom_reg_fn (Optional[Callable]): An optional custom regularization function.
            reg_strength (Optional[float]): An optional regularization strength.

        Returns:
            torch.Tensor: The loss with the regularization term added.
        """
        strength = (
            reg_strength if reg_strength is not None else self.regularization_strength
        )

        if strength <= 0:
            return loss

        if custom_reg_fn is not None:
            reg_term = custom_reg_fn(x, self.model)

        elif self.custom_regularization is not None:
            reg_term = self.custom_regularization(x, self.model)
        # default: L2 norm of score
        else:
            score = self.compute_score(x)
            reg_term = score.pow(2).sum(dim=list(range(1, len(x.shape)))).mean()

        return loss + strength * reg_term

    def __repr__(self):
        """Return a string representation of the loss function."""
        return f"{self.__class__.__name__}(model={self.model})"

    def __str__(self):
        """Return a string representation of the loss function."""
        return self.__repr__()
