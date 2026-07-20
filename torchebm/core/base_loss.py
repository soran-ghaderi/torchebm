"""
Base Loss Classes for Energy-Based Models
"""

import logging
import warnings
from abc import abstractmethod, ABC
from typing import Tuple, Union, Optional, Dict, Any, Callable

import torch
from torch import nn

from torchebm.core import BaseModel
from torchebm.core import BaseSampler
from torchebm.core import BaseScheduler
from torchebm.core import Schedulable
from torchebm.core import TorchEBMModule
from torchebm.core.base_module import warn_once

logger = logging.getLogger(__name__)


def _dtensor_type():
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return None
    return DTensor


class BaseLoss(Schedulable, TorchEBMModule, ABC):
    """
    Abstract base class for loss functions used in energy-based models.

    Args:
        dtype (torch.dtype): Data type for computations.
        device (Optional[Union[str, torch.device]]): Device for computations.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the base loss class."""
        super().__init__(device=device, dtype=dtype, *args, **kwargs)

    def _resolve_model_kwargs(
        self,
        model_kwargs: Optional[dict],
        legacy_kwargs: Optional[dict] = None,
        *,
        warn_key: str,
    ) -> dict:
        r"""Merge explicit `model_kwargs` with deprecated bare ``**kwargs``.

        Shared shim for losses whose bare ``**kwargs`` historically meant *model*
        conditioning (EqM, EM, score matching). The explicit dict wins on key
        conflicts; a non-empty legacy mapping triggers a one-time
        ``DeprecationWarning`` keyed by `warn_key`. The result is device-
        normalized once (see `_prepare_model_kwargs`) and is a fresh dict, so it
        never aliases the caller's mapping.
        """
        if legacy_kwargs:
            warn_once(
                warn_key,
                "Passing model conditioning as bare keyword arguments is "
                "deprecated; pass model_kwargs={...} instead.",
            )
            merged = {**legacy_kwargs, **(model_kwargs or {})}
        else:
            merged = model_kwargs
        return self._prepare_model_kwargs(merged)

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
        *args,
        **kwargs,
    ):
        super().__init__(
            dtype=dtype,
            device=device,
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

        self.register_buffer("replay_buffer", None)
        self.register_buffer(
            "buffer_ptr", torch.tensor(0, dtype=torch.long, device=self.device)
        )
        self._buffer_ptr_int: int = 0
        self.buffer_initialized = False

    def initialize_buffer(
        self,
        data_shape_no_batch: Tuple[int, ...],
        buffer_chunk_size: int = 1024,
        init_noise_scale: float = 0.01,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Initializes the replay buffer with random noise for PCD.

        Args:
            data_shape_no_batch (Tuple[int, ...]): The shape of the data excluding the batch dimension.
            buffer_chunk_size (int): The size of chunks to process during initialization.
            init_noise_scale (float): The scale of the initial noise.
            generator: RNG for the noise and the warm-up chains; the global RNG
                when `None`.

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
        logger.info("Initializing replay buffer with shape %s...", buffer_shape)

        self.replay_buffer = (
            torch.randn(
                buffer_shape,
                dtype=self.dtype,
                device=self.device,
                generator=generator,
            )
            * init_noise_scale
        )

        if self.init_steps > 0:
            logger.info("Running %d MCMC steps to populate buffer...", self.init_steps)
            with torch.no_grad():
                chunk_size = min(self.buffer_size, buffer_chunk_size)
                for i in range(0, self.buffer_size, chunk_size):
                    end = min(i + chunk_size, self.buffer_size)
                    current_chunk = self.replay_buffer[i:end].clone()
                    try:
                        with self.autocast_context():
                            updated_chunk = self.sampler.sample(
                                x=current_chunk,
                                n_steps=self.init_steps,
                                generator=generator,
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
        self._buffer_ptr_int = 0
        self.buffer_initialized = True
        logger.info("Replay buffer initialized.")

        return self.replay_buffer

    def get_start_points(
        self, x: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Gets the starting points for the MCMC sampler.

        For standard CD, this is the input data. For PCD, it's samples from the replay buffer.

        Args:
            x (torch.Tensor): The input data batch.
            generator: RNG for buffer index draws and exploration noise (PCD
                only); the global RNG when `None`.

        Returns:
            torch.Tensor: The tensor of starting points for the sampler.
        """
        x = x.to(device=self.device, dtype=self.dtype)

        batch_size = x.shape[0]
        data_shape_no_batch = x.shape[1:]

        if self.persistent:
            if not self.buffer_initialized:
                self.initialize_buffer(data_shape_no_batch, generator=generator)
                if not self.buffer_initialized:
                    raise RuntimeError("Buffer initialization failed.")

            if self.buffer_size < batch_size:
                warnings.warn(
                    f"Buffer size ({self.buffer_size}) is smaller than batch size ({batch_size}). Sampling with replacement.",
                    UserWarning,
                )
                indices = torch.randint(
                    0,
                    self.buffer_size,
                    (batch_size,),
                    device=self.device,
                    generator=generator,
                )
            else:
                # stratified sampling for better buffer coverage
                stride = self.buffer_size // batch_size
                base_indices = torch.arange(0, batch_size, device=self.device) * stride
                offset = torch.randint(
                    0, stride, (batch_size,), device=self.device, generator=generator
                )
                indices = (base_indices + offset) % self.buffer_size

            start_points = self.replay_buffer[indices]

            # add some noise for exploration
            if self.new_sample_ratio > 0.0:
                n_new = max(1, int(batch_size * self.new_sample_ratio))
                noise_indices = torch.randperm(
                    batch_size, device=self.device, generator=generator
                )[:n_new]
                noise_scale = 0.01
                start_points[noise_indices] = (
                    start_points[noise_indices]
                    + torch.randn_like(
                        start_points[noise_indices],
                        device=self.device,
                        dtype=self.dtype,
                        generator=generator,
                    )
                    * noise_scale
                )
        else:
            # standard CD-k uses data as starting points
            start_points = x.detach().clone()

        return start_points

    def get_negative_samples(
        self, x, batch_size, data_shape, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Gets negative samples using the replay buffer strategy.

        Args:
            x: (Unused) The input data tensor.
            batch_size (int): The number of samples to generate.
            data_shape (Tuple[int, ...]): The shape of the data samples (excluding batch size).
            generator: RNG for the noise and buffer index draws; the global RNG
                when `None`.

        Returns:
            torch.Tensor: Negative samples.
        """
        if not self.persistent or not self.buffer_initialized:
            # For non-persistent CD, just return random noise
            return torch.randn(
                (batch_size,) + data_shape,
                dtype=self.dtype,
                device=self.device,
                generator=generator,
            )

        n_new = max(1, int(batch_size * self.new_sample_ratio))
        n_old = batch_size - n_new

        all_samples = torch.empty(
            (batch_size,) + data_shape, dtype=self.dtype, device=self.device
        )

        # new random samples
        if n_new > 0:
            all_samples[:n_new] = torch.randn(
                (n_new,) + data_shape,
                dtype=self.dtype,
                device=self.device,
                generator=generator,
            )

        # samples from buffer
        if n_old > 0:

            indices = torch.randint(
                0, self.buffer_size, (n_old,), device=self.device, generator=generator
            )
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

        # FIFO strategy — use cached Python int to avoid GPU sync every step
        ptr = self._buffer_ptr_int

        if batch_size >= self.buffer_size:
            # batch larger than buffer, use latest samples
            self.replay_buffer[:] = samples[-self.buffer_size :]
            self._buffer_ptr_int = 0
            self.buffer_ptr.zero_()
        else:
            # handle buffer wraparound
            end_ptr = (ptr + batch_size) % self.buffer_size

            if end_ptr > ptr:
                self.replay_buffer[ptr:end_ptr] = samples
            else:
                # wraparound case - split update
                first_part = self.buffer_size - ptr
                self.replay_buffer[ptr:] = samples[:first_part]
                self.replay_buffer[:end_ptr] = samples[first_part:]

            self._buffer_ptr_int = end_ptr
            self.buffer_ptr.fill_(end_ptr)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # sync cached int with loaded tensor buffer_ptr
        self._buffer_ptr_int = int(self.buffer_ptr.item())

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
        use_autograd (bool): If True, compute the score by differentiating
            `model` in place. If False, use the functional path:
            `torch.func.functional_call` on a hook-free module with the model's
            current parameters. The functional path is required when the model's
            parameters are sharded DTensors (e.g. FSDP2 `fully_shard`), whose
            forward/backward hooks cannot run the second-order backward score
            matching needs.
        hutchinson_samples (int): The number of random samples for Hutchinson's trick.
        custom_regularization (Optional[Callable]): An optional function for custom regularization.
        functional_model (Optional[nn.Module]): Hook-free module used by the
            functional path in place of `model` for `functional_call`. Required
            when `model` holds FSDP-managed submodules (their hooks fire even
            under `functional_call`); pass an unwrapped instance of the same
            architecture. Ignored by the autograd path. Held as a structural
            template only: not registered as a submodule, its own parameters are
            never used.
    """

    def __init__(
        self,
        model: BaseModel,
        noise_scale: Union[float, BaseScheduler] = 0.01,
        regularization_strength: Union[float, BaseScheduler] = 0.0,
        use_autograd: bool = True,
        hutchinson_samples: int = 1,
        custom_regularization: Optional[Callable] = None,
        functional_model: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self._register_param("noise_scale", noise_scale)
        self._register_param("regularization_strength", regularization_strength)
        self.use_autograd = use_autograd
        self.hutchinson_samples = hutchinson_samples
        self.custom_regularization = custom_regularization
        object.__setattr__(self, "_functional_model", functional_model)

    @property
    def functional_model(self) -> Optional[nn.Module]:
        r"""Template module for the functional score path (never trained)."""
        return self._functional_model

    def _functional_state(self) -> Tuple[dict, dict, Optional[object]]:
        r"""Collect the model's parameters/buffers and their mesh, if sharded.

        Returns:
            `(params, buffers, mesh)` where `mesh` is the 1-D device mesh of
            the DTensor parameters, or None for plain tensors.
        """
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        dtensor = _dtensor_type()
        mesh = None
        if dtensor is not None:
            for p in params.values():
                if isinstance(p, dtensor):
                    mesh = p.device_mesh
                    break
        if mesh is not None and mesh.ndim != 1:
            raise NotImplementedError(
                f"The functional score path supports 1-D device meshes only, "
                f"got a {mesh.ndim}-D mesh."
            )
        return params, buffers, mesh

    def _functional_leaf(self, x: torch.Tensor, mesh) -> torch.Tensor:
        r"""Detached grad-leaf for the functional path, batch-sharded on `mesh`."""
        if mesh is None:
            return x.detach().requires_grad_(True)
        from torch.distributed.tensor import DTensor, Shard

        leaf = DTensor.from_local(x.detach(), mesh, [Shard(0)], run_check=False)
        leaf.requires_grad_(True)
        return leaf

    def _functional_energy(
        self,
        leaf: torch.Tensor,
        params: dict,
        buffers: dict,
        mesh,
        model_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        r"""Energy via `functional_call` on the hook-free template module.

        On a mesh, plain buffers are wrapped as Replicate DTensors and tensor
        `model_kwargs` are wrapped Shard(0) when batch-aligned, Replicate
        otherwise, so no operator mixes DTensor and plain operands.
        """
        module = self._functional_model
        kwargs = dict(model_kwargs or {})
        if mesh is None:
            module = module if module is not None else self.model
        else:
            if module is None:
                raise RuntimeError(
                    "functional_model is required when the model's parameters "
                    "are sharded DTensors: FSDP hooks fire even under "
                    "functional_call. Pass an unwrapped instance of the model "
                    "architecture at construction."
                )
            from torch.distributed.tensor import DTensor, Replicate, Shard

            local_batch = leaf.to_local().shape[0]
            buffers = {
                n: (
                    b
                    if isinstance(b, DTensor)
                    else DTensor.from_local(b, mesh, [Replicate()], run_check=False)
                )
                for n, b in buffers.items()
            }
            for k, v in kwargs.items():
                if torch.is_tensor(v) and not isinstance(v, DTensor):
                    placement = (
                        Shard(0)
                        if v.ndim > 0 and v.shape[0] == local_batch
                        else Replicate()
                    )
                    kwargs[k] = DTensor.from_local(
                        v, mesh, [placement], run_check=False
                    )
        return torch.func.functional_call(
            module, {**params, **buffers}, (leaf,), kwargs=kwargs
        )

    def _functional_localize(self, t: torch.Tensor, mesh) -> torch.Tensor:
        r"""Convert a Shard(0) result to its local shard with averaged gradients.

        `to_local` alone would leave parameter gradients as the cross-rank sum
        (the backward of the parameter all-gather is a summing reduce-scatter);
        the identity-forward rescale below divides them by the world size, so a
        local-mean loss yields global-batch-mean gradients, matching the
        convention of gradient-averaging data parallelism.
        """
        if mesh is None:
            return t
        from torch.distributed.tensor import Shard

        if t.placements != (Shard(0),):
            t = t.redistribute(mesh, [Shard(0)])
        local = t.to_local()
        c = 1.0 / mesh.size()
        return local * c + local.detach() * (1.0 - c)

    def _functional_score(
        self, x_perturbed: torch.Tensor, model_kwargs: Optional[dict] = None
    ) -> torch.Tensor:
        r"""Score \(\nabla_x E(x)\) via the functional path.

        The input is treated as a constant: the returned score is
        differentiable with respect to the model parameters (`create_graph`),
        not with respect to the caller's tensor.
        """
        params, buffers, mesh = self._functional_state()
        leaf = self._functional_leaf(x_perturbed, mesh)
        with self.autocast_context():
            energy = self._functional_energy(leaf, params, buffers, mesh, model_kwargs)
        score = torch.autograd.grad(energy.sum(), leaf, create_graph=True)[0]
        return self._functional_localize(score, mesh)

    def _require_autograd_safe_params(self) -> None:
        r"""Reject the in-place autograd path when parameters are sharded."""
        dtensor = _dtensor_type()
        if dtensor is not None and isinstance(
            next(self.model.parameters(), None), dtensor
        ):
            raise RuntimeError(
                "The autograd score path cannot run with FSDP-managed "
                "(DTensor) parameters: resharding hooks free storage the "
                "second-order backward still references. Construct the loss "
                "with use_autograd=False and functional_model=<unwrapped "
                "instance of the model architecture>."
            )

    @property
    def noise_scale(self) -> float:
        return self.get_scheduled_value("noise_scale")

    @noise_scale.setter
    def noise_scale(self, value: Union[float, BaseScheduler]) -> None:
        self._register_param("noise_scale", value)

    @property
    def regularization_strength(self) -> float:
        return self.get_scheduled_value("regularization_strength")

    @regularization_strength.setter
    def regularization_strength(self, value: Union[float, BaseScheduler]) -> None:
        self._register_param("regularization_strength", value)

    def compute_score(
        self,
        x: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        model_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        r"""
        Computes the score function, \(\nabla_x E(x)\).

        Args:
            x (torch.Tensor): The input data tensor.
            noise (Optional[torch.Tensor]): Optional noise tensor for perturbed variants.
            model_kwargs (Optional[dict]): Conditioning arguments forwarded to the
                model (e.g. class labels). This is the single funnel every
                score-matching variant routes its model call through, so passing
                it here conditions all variants.

        Returns:
            torch.Tensor: The score function evaluated at `x` or `x + noise`.
        """

        x = x.to(device=self.device, dtype=self.dtype)

        if noise is not None:
            noise = noise.to(device=self.device, dtype=self.dtype)
            x_perturbed = x + noise
        else:
            x_perturbed = x

        if not self.use_autograd:
            return self._functional_score(x_perturbed, model_kwargs=model_kwargs)

        self._require_autograd_safe_params()
        if not x_perturbed.requires_grad:
            x_perturbed.requires_grad_(True)

        with self.autocast_context():
            energy = self.model(x_perturbed, **(model_kwargs or {}))

        score = torch.autograd.grad(energy.sum(), x_perturbed, create_graph=True)[0]

        return score

    def perturb_data(
        self, x: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # todo: add more noise types
        """
        Perturbs the input data with Gaussian noise for denoising variants.

        Args:
            x (torch.Tensor): Input data tensor.
            generator: RNG for the Gaussian noise; the global RNG when `None`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the perturbed data
                and the noise that was added.
        """

        x = x.to(device=self.device, dtype=self.dtype)
        noise = (
            torch.randn_like(
                x, device=self.device, dtype=self.dtype, generator=generator
            )
            * self.noise_scale
        )
        x_perturbed = x + noise
        return x_perturbed, noise

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
        model_kwargs: Optional[dict] = None,
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
            score = self.compute_score(x, model_kwargs=model_kwargs)
            reg_term = score.square().sum(dim=list(range(1, len(x.shape)))).mean()

        return loss + strength * reg_term

    def __repr__(self):
        """Return a string representation of the loss function."""
        return f"{self.__class__.__name__}(model={self.model})"

    def __str__(self):
        """Return a string representation of the loss function."""
        return self.__repr__()
