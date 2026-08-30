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
from torchebm.core.base_module import (
    _unexpected_init_args_message,
    substitute_condition,
    warn_once,
)

logger = logging.getLogger(__name__)


def _dtensor_type():
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return None
    return DTensor


def _has_dtensor_params(module: nn.Module) -> bool:
    r"""Whether the module's parameters are sharded DTensors (e.g. FSDP2)."""
    dtensor = _dtensor_type()
    return dtensor is not None and isinstance(
        next(module.parameters(), None), dtensor
    )


class BaseLoss(Schedulable, TorchEBMModule, ABC):
    r"""
    Abstract base class for loss functions used in energy-based models.

    Distributed reduction semantics: losses reduce with means over the local
    batch. Combined with the gradient averaging of data-parallel wrappers
    (DDP, FSDP), parameter gradients equal the global-batch mean exactly when
    per-rank batch sizes are equal; with unequal batches the smaller ranks
    are over-weighted. Batch-global statistics (couplings, trimmed means)
    stay rank-local unless a component takes an explicit ``process_group``.

    Args:
        dtype (torch.dtype): Data type for computations.
        device (Optional[Union[str, torch.device]]): Device for computations.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        cfg_dropout: float = 0.0,
        null_condition: Union[int, float, torch.Tensor, Callable, None] = None,
        check_conditioning: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the base loss class.

        Args:
            dtype: Data type for computations.
            device: Device for computations.
            cfg_dropout: Classifier-free-guidance label dropout: probability of
                replacing the ``y`` conditioning with `null_condition` per
                sample during training. 0 (default) disables dropout; applies
                only in training mode and only when ``y`` is passed.
            null_condition: Null condition for dropped samples: an int (the
                ``num_classes`` label convention), a tensor broadcast over the
                non-batch dims (e.g. a zero embedding), or a callable
                ``(y, mask) -> y``. Required when ``cfg_dropout > 0``.
            check_conditioning: If True (default), verify on the first
                conditional loss call that the model actually consumes ``y``
                (same input, two distinct in-batch y values; identical outputs
                raise instead of silently training an unconditional model).
                Set False for models this probe misjudges, e.g.
                stochastic-in-eval backbones.

        Raises:
            TypeError: If constructor arguments remain that no class in the
                loss's MRO binds; the message lists the supported parameters
                and the installed torchebm version.
            ValueError: If ``cfg_dropout`` is outside [0, 1], or positive
                without a `null_condition`.
        """
        if args or kwargs:
            raise TypeError(
                _unexpected_init_args_message(type(self), args, kwargs, BaseLoss)
            )
        if not 0.0 <= cfg_dropout <= 1.0:
            raise ValueError(f"cfg_dropout must be in [0, 1], got {cfg_dropout}")
        if cfg_dropout > 0 and null_condition is None:
            raise ValueError("cfg_dropout > 0 requires null_condition")
        super().__init__(device=device, dtype=dtype)
        self.cfg_dropout = cfg_dropout
        self.null_condition = null_condition
        self._condition_check_pending = check_conditioning
        self._condition_probe_deferrals = 0

    def _probe_forward(self, px: torch.Tensor, pmk: dict) -> torch.Tensor:
        r"""Model call used by the conditioning probe; energy convention."""
        return self.model(px, **pmk)

    #: Undecided conditioning probes tolerated before concluding the model
    #: ignores y. adaLN-Zero models need 3 optimizer steps for y-dependence
    #: to reach the output; 10 leaves headroom without hiding real bugs long.
    _CONDITION_PROBE_MAX_DEFERRALS = 10

    def _check_condition(
        self, x: torch.Tensor, model_kwargs: Optional[dict]
    ) -> None:
        r"""One-time probe that the model consumes ``y`` when it is passed.

        Runs on the first conditional call: the same one-sample input under
        two distinct in-batch y values (never fabricated labels, which could
        index outside an embedding), model temporarily in eval mode,
        gradients off. Identical nonzero outputs on a fresh model raise.
        Identical all-zero outputs are indeterminate, the signature of a
        zero-initialized output head: adaLN-Zero models emit exactly zero
        until trained, and their y-dependence surfaces only a few optimizer
        steps later (head, then modulations, then conditioning). The probe
        therefore stays armed and retries on later calls, tolerating a
        bounded number of undecided probes before raising; once decided or
        warned, no per-step work remains on the hot path. A first conditional
        batch with no two distinct y values warns and disarms the probe.

        Raises:
            ValueError: If the outputs of the probe pair coincide on a fresh
                model, or remain undecided after the deferral budget.
        """
        if not self._condition_check_pending or not model_kwargs:
            return
        y = model_kwargs.get("y")
        if y is None:
            return
        distinct = (y != y[0]).reshape(y.shape[0], -1).any(dim=1)
        if y.shape[0] < 2 or not bool(distinct.any()):
            self._condition_check_pending = False
            warnings.warn(
                "Conditioning consumption could not be verified: the first "
                "conditional batch carries a single distinct y value. Pass "
                "check_conditioning=False to silence this warning.",
                UserWarning,
            )
            return
        i1 = int(distinct.int().argmax())
        px = x[:1].detach()
        batch = x.shape[0]
        pmk = {
            k: v[:1]
            if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == batch
            else v
            for k, v in model_kwargs.items()
            if k != "y"
        }
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                out_a = self._probe_forward(px, {**pmk, "y": y[0:1]})
                out_b = self._probe_forward(px, {**pmk, "y": y[i1 : i1 + 1]})
        finally:
            self.model.train(was_training)
        if isinstance(out_a, tuple):
            out_a = out_a[0]
        if isinstance(out_b, tuple):
            out_b = out_b[0]
        if torch.allclose(out_a, out_b, rtol=1e-5, atol=1e-6):
            if bool(out_a.any()) and self._condition_probe_deferrals == 0:
                raise ValueError(
                    f"{type(self.model).__name__} returns identical outputs for "
                    "two different y values, so it ignores its conditioning input "
                    "while y was passed. Wire y into the model's forward, or pass "
                    "check_conditioning=False to skip this probe."
                )
            self._condition_probe_deferrals += 1
            if self._condition_probe_deferrals <= self._CONDITION_PROBE_MAX_DEFERRALS:
                return
            raise ValueError(
                f"{type(self.model).__name__} still returns identical outputs "
                f"for two different y values after "
                f"{self._CONDITION_PROBE_MAX_DEFERRALS} deferred probes, so it "
                "ignores its conditioning input while y was passed. Wire y "
                "into the model's forward, or pass check_conditioning=False "
                "to skip this probe."
            )
        self._condition_check_pending = False

    def _apply_cfg_dropout(
        self,
        model_kwargs: Optional[dict],
        generator: Optional[torch.Generator] = None,
    ) -> Optional[dict]:
        r"""Replace ``y`` with the null condition per sample during training.

        No-op unless the loss is in training mode, ``cfg_dropout > 0`` and
        `model_kwargs` carries a ``'y'`` entry. The mask is drawn with
        `generator` for reproducibility.
        """
        if (
            self.cfg_dropout == 0.0
            or not self.training
            or not model_kwargs
            or "y" not in model_kwargs
        ):
            return model_kwargs
        y = model_kwargs["y"]
        mask = (
            torch.rand(y.shape[0], device=y.device, generator=generator)
            < self.cfg_dropout
        )
        return {
            **model_kwargs,
            "y": substitute_condition(y, mask, self.null_condition),
        }

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


class BaseInterpolantLoss(BaseLoss):
    r"""Shared skeleton for interpolant-based losses (EqM, EM, FM).

    Owns the pieces every stochastic-interpolant objective repeats:
    interpolant and minibatch-coupling resolution, the training-time
    distribution (`t_sampler`, uniform or the EDM lognormal skew or a
    callable), the ``train_eps`` interval, and the per-timestep weight hook
    ``loss_weight_fn``. Subclasses set `_default_coupling` and consume
    `_sample_t` / `_check_interval` in their forwards; everything else
    (targets, reductions, regularizers) stays subclass-specific.

    Internal: not exported from ``torchebm.losses``; its constructor
    signature may change as more losses adopt it.

    Args:
        interpolant: Interpolant name (e.g. 'linear', 'cosine', 'vp') or
            BaseInterpolant instance.
        coupling: Minibatch coupling name or BaseCoupling instance; ``None``
            uses the subclass default.
        train_eps: Epsilon for training time interval stability. Float or
            `BaseScheduler`.
        t_sampler: Training-time distribution:

            - 'uniform' (default): uniform over the training interval
            - 'lognormal': EDM timestep skew, $\sigma = e^{z p_{std} + p_{mean}}$
              with $z \sim \mathcal{N}(0, 1)$ and $t = 1/(1+\sigma)$ clamped to
              [1e-4, 1] (intersected with the ``train_eps`` interval)
            - a callable ``(batch, *, device, dtype, generator) -> t``
              returning shape (batch_size,)

        t_p_mean: Lognormal skew location $p_{mean}$ (EDM $P_{mean}$). Default: -1.2.
        t_p_std: Lognormal skew scale $p_{std}$ (EDM $P_{std}$), positive. Default: 1.2.
        loss_weight_fn: Optional per-timestep weight hook ``t -> w(t)`` (shape
            (batch_size,)) multiplied into the per-sample loss; the mechanism
            behind min-SNR / EDM $\lambda(\sigma)$ style weightings. None
            (default) keeps the loss unweighted.
    """

    _default_coupling = "independent"

    def __init__(
        self,
        interpolant="linear",
        coupling=None,
        train_eps: Union[float, "BaseScheduler"] = 0.0,
        t_sampler: Union[str, Callable[..., torch.Tensor]] = "uniform",
        t_p_mean: float = -1.2,
        t_p_std: float = 1.2,
        loss_weight_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if not callable(t_sampler) and t_sampler not in ("uniform", "lognormal"):
            raise ValueError(
                "t_sampler must be 'uniform', 'lognormal', or a callable "
                f"(batch, *, device, dtype, generator) -> t, got {t_sampler!r}"
            )
        if t_p_std <= 0:
            raise ValueError(f"t_p_std must be positive, got {t_p_std}")
        if loss_weight_fn is not None and not callable(loss_weight_fn):
            raise TypeError(
                "loss_weight_fn must be callable or None, got "
                f"{type(loss_weight_fn).__name__}"
            )
        self.t_sampler = t_sampler
        self.t_p_mean = t_p_mean
        self.t_p_std = t_p_std
        self.loss_weight_fn = loss_weight_fn
        self._register_param("train_eps", train_eps)
        from torchebm.couplings import resolve_coupling
        from torchebm.interpolants import resolve_interpolant

        owner = type(self).__name__
        self.interpolant = resolve_interpolant(
            interpolant, default="linear", owner=owner
        )
        self.coupling = resolve_coupling(
            coupling, default=self._default_coupling, owner=owner
        )

    @property
    def train_eps(self) -> float:
        return self.get_scheduled_value("train_eps")

    @train_eps.setter
    def train_eps(self, value) -> None:
        self._register_param("train_eps", value)

    def compute_loss(
        self,
        x: torch.Tensor,
        *args,
        x0: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Resolve conditioning, run `training_losses`, reduce with weights.

        Shared pipeline: model_kwargs resolution (deprecated bare kwargs
        included), the conditioning-consumption probe, cfg label dropout,
        then the subclass `training_losses`; per-pair coupling weights, when
        present, replace the plain mean.

        Args:
            x: Data samples of shape (batch_size, ...).
            *args: Additional positional arguments.
            x0: Optional source samples of shape (batch_size, ...).
            model_kwargs: Conditioning arguments forwarded to the model.
            **kwargs: Deprecated bare model kwargs.

        Returns:
            Scalar loss value.
        """
        mk = self._resolve_model_kwargs(
            model_kwargs,
            kwargs,
            warn_key=f"{type(self).__name__}-bare-model-kwargs",
        )
        self._check_condition(x, mk)
        mk = self._apply_cfg_dropout(mk, generator)
        terms = self.training_losses(
            x, model_kwargs=mk, x0=x0, generator=generator
        )
        loss = terms["loss"]
        weights = terms.get("weights")
        if weights is not None:
            return (weights * loss).sum() / weights.sum().clamp_min(1e-12)
        return loss.mean()

    def _check_interval(self) -> Tuple[float, float]:
        r"""Get training time interval respecting epsilon."""
        eps = self.train_eps
        return eps, 1.0 - eps

    def _sample_t(
        self, batch: int, generator: Optional[torch.Generator]
    ) -> torch.Tensor:
        r"""Draw training times for the configured `t_sampler`.

        'lognormal' is the EDM timestep skew: \(\sigma = e^{z p_{std} + p_{mean}}\)
        with \(z \sim \mathcal{N}(0, 1)\), \(t = 1/(1+\sigma)\), clamped into the
        training interval with a 1e-4 floor. A callable receives
        ``(batch, device=, dtype=, generator=)`` and must return shape (batch,).
        """
        if callable(self.t_sampler):
            return self.t_sampler(
                batch, device=self.device, dtype=self.dtype, generator=generator
            )
        t0, t1 = self._check_interval()
        if self.t_sampler == "lognormal":
            z = torch.randn(
                batch, device=self.device, dtype=self.dtype, generator=generator
            )
            sigma = torch.exp(z * self.t_p_std + self.t_p_mean)
            return (1.0 / (1.0 + sigma)).clamp(min=max(1.0e-4, t0), max=t1)
        return (
            torch.rand(
                batch, device=self.device, dtype=self.dtype, generator=generator
            )
            * (t1 - t0)
            + t0
        )


class BaseContrastiveDivergence(BaseLoss):
    r"""
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

    Distributed runs keep the replay buffer rank-local by design: each rank
    holds independent persistent chains, so the world size multiplies chain
    diversity, and no default path issues a collective. Chains are exchanged
    only through an explicit `mix_buffer_across_ranks` call.

    Checkpointing: `replay_buffer` and `buffer_ptr` are registered buffers and
    enter `state_dict()` once the buffer exists. The buffer is registered
    lazily as None, so a fresh instance must call `initialize_buffer` (same
    data shape) before `load_state_dict` on a checkpoint that contains a
    buffer. Rank-local buffers hold different chains on every rank: save one
    file per rank, for example
    `torch.save(loss.state_dict(), f"loss_rank{rank}.pt")`, and restore with
    the same world size. Do not put this state into a
    `torch.distributed.checkpoint` state dict; its planner deduplicates
    non-sharded tensors as replicated, which silently keeps rank 0's chains
    only. When the loss's `model` is itself sharded, the full
    `loss.state_dict()` contains its DTensor parameters: save those through
    `torch.distributed.checkpoint` on the model instead, and keep only the
    buffer entries (`replay_buffer`, `buffer_ptr`) in the rank-local file,
    restoring them with `load_state_dict(..., strict=False)` after
    `initialize_buffer`. Re-initializing from noise instead of restoring is
    an acceptable fallback; chains re-warm within a few hundred steps.
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

    def mix_buffer_across_ranks(
        self,
        process_group: Optional["torch.distributed.ProcessGroup"] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        r"""Redistribute the replay-buffer chains uniformly across ranks.

        Gathers every rank's buffer, applies one permutation shared by all
        ranks, and keeps this rank's shard: the pooled chains are re-dealt
        with no chain duplicated or lost. The permutation is drawn on rank 0
        (from `generator` there, the global CPU RNG otherwise) and broadcast,
        so per-rank generators cannot desynchronize the shuffle.

        This is a collective: every rank in the group must call it together,
        outside `forward`, at whatever cadence suits the run. It costs one
        all_gather of the full buffer (transiently the world size times the
        buffer memory) plus a small broadcast, and requires equal
        `buffer_size` on every rank. The FIFO pointer is left unchanged;
        after a uniform shuffle every overwrite position is equally valid.
        No-op in single-process runs.

        Args:
            process_group: Group to mix over; the default group when None.
            generator: CPU generator for the shared permutation; significant
                on rank 0 only.

        Raises:
            RuntimeError: If the loss is not persistent or the buffer is not
                initialized.
        """
        if not self.persistent:
            raise RuntimeError(
                "mix_buffer_across_ranks requires a persistent loss "
                "(persistent=True)."
            )
        if not self.buffer_initialized:
            raise RuntimeError(
                "The replay buffer is not initialized; run one training step "
                "or call initialize_buffer() first."
            )
        from torchebm.utils.distributed import (
            all_gather_cat,
            broadcast_tensor,
            get_rank,
            get_world_size,
        )

        if get_world_size(process_group) == 1:
            return
        gathered = all_gather_cat(self.replay_buffer, group=process_group)
        perm = torch.randperm(gathered.shape[0], generator=generator)
        perm = broadcast_tensor(perm, src=0, group=process_group)
        start = get_rank(process_group) * self.buffer_size
        self.replay_buffer.copy_(gathered[perm[start : start + self.buffer_size]])

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
        if not use_autograd:
            self._register_functional_grad_hooks()

    def _register_functional_grad_hooks(self) -> None:
        r"""Restore parameter-gradient placements after functional backwards.

        The functional path bypasses FSDP's hooks, so the backward leaves
        DTensor parameter gradients with `Partial` placement (unreduced
        per-rank contributions); optimizers require gradients that match the
        parameter's placement. This hook redistributes each gradient to the
        parameter's placements at accumulation time, performing the same
        reduce-scatter the hook path would have run. No-op for plain tensors
        and already-matching placements, so it is safe when the same model is
        also trained through hook-path losses.
        """
        dtensor = _dtensor_type()
        if dtensor is None:
            return

        def _restore_placement(param: torch.Tensor) -> None:
            grad = param.grad
            if isinstance(grad, dtensor) and tuple(grad.placements) != tuple(
                param.placements
            ):
                param.grad = grad.redistribute(
                    param.device_mesh, param.placements
                )

        for p in self.model.parameters():
            p.register_post_accumulate_grad_hook(_restore_placement)

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
        if _has_dtensor_params(self.model):
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
