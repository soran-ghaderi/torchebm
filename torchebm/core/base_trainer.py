import logging
import warnings
from contextlib import nullcontext
import torch
from typing import Dict, Optional, Tuple, Union, Any, List, Callable
from torch.utils.data import DataLoader

from .base_model import BaseModel
from .base_sampler import BaseSampler
from .base_loss import BaseLoss

logger = logging.getLogger(__name__)

# TF32 for fp32 matmuls on Ampere+ (no-op on CPU / pre-Ampere).
torch.set_float32_matmul_precision("high")


class BaseTrainer:
    """
    Base class for training energy-based models.

    This class provides a generic interface for training EBMs, supporting various
    training methods and mixed precision training.

    Args:
        model: Energy function to train
        optimizer: PyTorch optimizer to use
        loss_fn: Loss function for training
        device: Device to run training on
        dtype: Data type for computations
        use_mixed_precision: Whether to use mixed precision training
        callbacks: List of callback functions for training events

    Methods:
        train_step: Perform a single training step
        train_epoch: Train for a full epoch
        train: Train for multiple epochs
        validate: Validate the model
        save_checkpoint: Save model checkpoint
        load_checkpoint: Load model from checkpoint
    """

    def __init__(
        self,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        loss_fn: BaseLoss,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        use_mixed_precision: bool = False,
        grad_accum_steps: int = 1,
        callbacks: Optional[List[Callable]] = None,
    ):
        if grad_accum_steps < 1:
            raise ValueError(
                f"grad_accum_steps must be >= 1, got {grad_accum_steps}"
            )
        self.grad_accum_steps = int(grad_accum_steps)
        self._accum_step_count = 0

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        # Set up device
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Set up dtype and mixed precision
        self.dtype = dtype
        self.use_mixed_precision = use_mixed_precision

        # Initialize callbacks
        self.callbacks = callbacks or []

        # Configure mixed precision
        if self.use_mixed_precision:
            try:
                from torch.cuda.amp import GradScaler
                self.autocast_available = self.device.type.startswith("cuda")
                if self.autocast_available:
                    self.grad_scaler = GradScaler()
                else:
                    warnings.warn(
                        f"Mixed precision requested but device is {self.device}. Mixed precision requires CUDA. Falling back to full precision.",
                        UserWarning,
                    )
                    self.use_mixed_precision = False
                    self.autocast_available = False
            except ImportError:
                warnings.warn(
                    "Mixed precision requested but torch.cuda.amp not available. Falling back to full precision. Requires PyTorch 1.6+.",
                    UserWarning,
                )
                self.use_mixed_precision = False
                self.autocast_available = False

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        if hasattr(self.loss_fn, "to"):
            self.loss_fn = self.loss_fn.to(device=self.device, dtype=self.dtype)

        # Create metrics dictionary for tracking
        self.metrics: Dict[str, Any] = {"loss": []}

    def autocast_context(self):
        """Return autocast context if enabled, else no-op."""
        if self.use_mixed_precision and self.autocast_available:
            from torch.cuda.amp import autocast
            return autocast()
        return nullcontext()

    def _split_batch(self, batch: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Split a dataloader batch into ``(data, model_kwargs)``.

        Accepts a bare ``Tensor`` (unconditional), a ``(data, cond)`` pair, or a
        ``dict`` (``x``/``data`` key holds the samples; the rest is conditioning).
        A bare conditioning tensor is exposed to the loss as ``model_kwargs={"y":
        cond}`` (matching the label convention of the conditional models). Data is
        moved to the trainer device+dtype; conditioning tensors are moved
        device-only so integer labels stay integral.
        """
        to_dev = lambda v: v.to(self.device) if torch.is_tensor(v) else v
        if isinstance(batch, torch.Tensor):
            return batch.to(device=self.device, dtype=self.dtype), {}
        if isinstance(batch, dict):
            rest = dict(batch)
            data = rest.pop("x", None)
            if data is None:
                data = rest.pop("data")
            data = data.to(device=self.device, dtype=self.dtype)
            return data, {k: to_dev(v) for k, v in rest.items()}
        if isinstance(batch, (tuple, list)):
            data = batch[0].to(device=self.device, dtype=self.dtype)
            cond = batch[1] if len(batch) > 1 else None
            if cond is None:
                model_kwargs: Dict[str, Any] = {}
            elif isinstance(cond, dict):
                model_kwargs = {k: to_dev(v) for k, v in cond.items()}
            else:
                model_kwargs = {"y": to_dev(cond)}
            return data, model_kwargs
        raise TypeError(
            f"Unsupported batch type {type(batch).__name__}; expected Tensor, "
            "(data, cond) pair, or dict."
        )

    def train_step(
        self, batch: Any, model_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform a single training step.

        When ``grad_accum_steps > 1``, gradients accumulate across calls and the
        optimizer only steps on the accumulation boundary. The loss is scaled by
        ``1 / grad_accum_steps`` so the accumulated gradient matches a single
        step at the larger effective batch size.

        Args:
            batch: Batch of training data. A bare ``Tensor``, a ``(data, cond)``
                pair, or a ``dict`` (see `_split_batch`) for conditional training.
            model_kwargs: Optional conditioning merged over anything unpacked
                from ``batch`` (explicit keys win). Forwarded to the loss.

        Returns:
            Dictionary containing metrics from this step
        """
        batch, batch_kwargs = self._split_batch(batch)
        if model_kwargs:
            batch_kwargs = {**batch_kwargs, **model_kwargs}

        if self._accum_step_count == 0:
            self.optimizer.zero_grad(set_to_none=True)

        if self.use_mixed_precision and self.autocast_available:
            with self.autocast_context():
                loss = self.loss_fn(batch, model_kwargs=batch_kwargs)
            scaled = loss / self.grad_accum_steps if self.grad_accum_steps > 1 else loss
            self.grad_scaler.scale(scaled).backward()
        else:
            loss = self.loss_fn(batch, model_kwargs=batch_kwargs)
            scaled = loss / self.grad_accum_steps if self.grad_accum_steps > 1 else loss
            scaled.backward()

        self._accum_step_count += 1
        if self._accum_step_count >= self.grad_accum_steps:
            if self.use_mixed_precision and self.autocast_available:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            self._accum_step_count = 0

        return {"loss": loss.item()}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader containing training data

        Returns:
            Dictionary with average metrics for the epoch
        """
        # Set model to training mode
        self.model.train()

        # Initialize metrics for this epoch
        epoch_metrics: Dict[str, List[float]] = {"loss": []}

        # Iterate through batches
        for batch in dataloader:
            # Call any batch start callbacks
            for callback in self.callbacks:
                if hasattr(callback, "on_batch_start"):
                    callback.on_batch_start(self, batch)

            # Perform training step
            step_metrics = self.train_step(batch)

            # Update epoch metrics
            for key, value in step_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)

            # Call any batch end callbacks
            for callback in self.callbacks:
                if hasattr(callback, "on_batch_end"):
                    callback.on_batch_end(self, batch, step_metrics)

        # Calculate average metrics
        avg_metrics = {
            key: sum(values) / len(values) for key, values in epoch_metrics.items()
        }

        return avg_metrics

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        validate_fn: Optional[Callable] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            dataloader: DataLoader containing training data
            num_epochs: Number of epochs to train for
            validate_fn: Optional function for validation after each epoch

        Returns:
            Dictionary with metrics over all epochs
        """
        # Initialize training history
        history: Dict[str, List[float]] = {"loss": []}

        # Call any training start callbacks
        for callback in self.callbacks:
            if hasattr(callback, "on_train_start"):
                callback.on_train_start(self)

        # Train for specified number of epochs
        for epoch in range(num_epochs):
            # Call any epoch start callbacks
            for callback in self.callbacks:
                if hasattr(callback, "on_epoch_start"):
                    callback.on_epoch_start(self, epoch)

            # Train for one epoch
            epoch_metrics = self.train_epoch(dataloader)

            # Update training history
            for key, value in epoch_metrics.items():
                if key not in history:
                    history[key] = []
                history[key].append(value)

            # Print progress
            logger.info("Epoch %d/%d, Loss: %.6f", epoch + 1, num_epochs, epoch_metrics["loss"])

            # Validate if function provided
            if validate_fn is not None:
                val_metrics = validate_fn(self.model)
                logger.info("Validation: %s", val_metrics)

                # Update validation metrics in history
                for key, value in val_metrics.items():
                    val_key = f"val_{key}"
                    if val_key not in history:
                        history[val_key] = []
                    history[val_key].append(value)

            # Call any epoch end callbacks
            for callback in self.callbacks:
                if hasattr(callback, "on_epoch_end"):
                    callback.on_epoch_end(self, epoch, epoch_metrics)

        # Call any training end callbacks
        for callback in self.callbacks:
            if hasattr(callback, "on_train_end"):
                callback.on_train_end(self, history)

        return history

    def save_checkpoint(self, path: str) -> None:
        """
        Save a checkpoint of the current training state.

        Args:
            path: Path to save the checkpoint to
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.metrics,
        }

        if self.use_mixed_precision and hasattr(self, "grad_scaler"):
            checkpoint["grad_scaler_state_dict"] = self.grad_scaler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """
        Load a checkpoint to resume training.

        Args:
            path: Path to the checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "metrics" in checkpoint:
            self.metrics = checkpoint["metrics"]

        if (
            self.use_mixed_precision
            and "grad_scaler_state_dict" in checkpoint
            and hasattr(self, "grad_scaler")
        ):
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])


class ContrastiveDivergenceTrainer(BaseTrainer):
    """
    Specialized trainer for contrastive divergence training of EBMs.

    Args:
        model: Energy function to train
        sampler: MCMC sampler for generating negative samples
        optimizer: PyTorch optimizer
        learning_rate: Learning rate (if optimizer not provided)
        k_steps: Number of MCMC steps for generating samples
        persistent: Whether to use persistent contrastive divergence (PCD)
        buffer_size: Replay buffer size for PCD
        device: Device to run training on
        dtype: Data type for computations
        use_mixed_precision: Whether to use mixed precision training
    """

    def __init__(
        self,
        model: BaseModel,
        sampler: BaseSampler,
        optimizer: Optional[torch.optim.Optimizer] = None,
        learning_rate: float = 0.01,
        k_steps: int = 10,
        persistent: bool = False,
        buffer_size: int = 1000,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        use_mixed_precision: bool = False,
        grad_accum_steps: int = 1,
    ):
        # Create optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Import here to avoid circular import
        from torchebm.losses.contrastive_divergence import ContrastiveDivergence

        # Create loss function
        loss_fn = ContrastiveDivergence(
            model=model,
            sampler=sampler,
            k_steps=k_steps,
            persistent=persistent,
            buffer_size=buffer_size,
            dtype=dtype,
            device=device,
            use_mixed_precision=use_mixed_precision,
        )

        # Initialize base trainer
        super().__init__(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            dtype=dtype,
            use_mixed_precision=use_mixed_precision,
            grad_accum_steps=grad_accum_steps,
        )

        self.sampler = sampler

    def train_step(
        self, batch: Any, model_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform a single contrastive divergence training step.

        Honors ``grad_accum_steps`` in the same way as ``BaseTrainer.train_step``.

        Args:
            batch: Batch of real data samples (bare ``Tensor``, ``(data, cond)``
                pair, or ``dict``; see `BaseTrainer._split_batch`).
            model_kwargs: Optional conditioning merged over anything unpacked from
                ``batch`` (explicit keys win). Forwarded to the CD loss so the
                negatives share the positives' conditional energy.

        Returns:
            Dictionary containing metrics from this step
        """
        batch, batch_kwargs = self._split_batch(batch)
        if model_kwargs:
            batch_kwargs = {**batch_kwargs, **model_kwargs}

        if self._accum_step_count == 0:
            self.optimizer.zero_grad(set_to_none=True)

        if self.use_mixed_precision and self.autocast_available:
            with self.autocast_context():
                loss, neg_samples = self.loss_fn(batch, model_kwargs=batch_kwargs)
            scaled = loss / self.grad_accum_steps if self.grad_accum_steps > 1 else loss
            self.grad_scaler.scale(scaled).backward()
        else:
            loss, neg_samples = self.loss_fn(batch, model_kwargs=batch_kwargs)
            scaled = loss / self.grad_accum_steps if self.grad_accum_steps > 1 else loss
            scaled.backward()

        self._accum_step_count += 1
        if self._accum_step_count >= self.grad_accum_steps:
            if self.use_mixed_precision and self.autocast_available:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            self._accum_step_count = 0

        return {
            "loss": loss.item(),
            "pos_energy": self.model(batch, **batch_kwargs).mean().item(),
            "neg_energy": self.model(neg_samples, **batch_kwargs).mean().item(),
        }
