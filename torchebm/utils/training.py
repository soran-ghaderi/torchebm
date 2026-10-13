r"""General training utilities for TorchEBM."""

import math
from collections import OrderedDict
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    r"""Draw training times in \([0, 1)\) under a timestep density.

    The `diffusers.training_utils` function of the same name (Esser et al.,
    2024, "Scaling Rectified Flow Transformers"):

    - ``"uniform"``: \(u \sim \mathcal{U}(0, 1)\).
    - ``"logit_normal"``: \(u = \sigma(z)\), \(z \sim \mathcal{N}(\mu, s^2)\),
      with ``logit_mean`` \(\mu\) and ``logit_std`` \(s\). The EDM lognormal
      skew with location \(p_{mean}\) and scale \(p_{std}\) is the same
      distribution with ``logit_mean=-p_mean, logit_std=p_std``.
    - ``"mode"``: \(u \leftarrow 1 - u - s\left(\cos^2(\tfrac{\pi u}{2}) - 1
      + u\right)\) for \(u \sim \mathcal{U}(0, 1)\) and ``mode_scale`` \(s\).

    Args:
        weighting_scheme: ``"uniform"``, ``"logit_normal"``, or ``"mode"``.
        batch_size: Number of times to draw.
        logit_mean: Mean \(\mu\) of the logit-normal scheme.
        logit_std: Standard deviation \(s\) of the logit-normal scheme.
        mode_scale: Scale \(s\) of the mode scheme.
        device: Device of the result.
        dtype: Data type of the result.
        generator: RNG for the draw; the global RNG when ``None``.

    Returns:
        Tensor of shape (batch_size,).

    Raises:
        ValueError: If ``weighting_scheme`` is unknown.
    """
    if weighting_scheme == "logit_normal":
        z = torch.normal(
            logit_mean,
            logit_std,
            size=(batch_size,),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        return torch.sigmoid(z)
    if weighting_scheme not in ("uniform", "mode"):
        raise ValueError(
            "weighting_scheme must be 'uniform', 'logit_normal', or 'mode', "
            f"got {weighting_scheme!r}"
        )
    u = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    if weighting_scheme == "mode":
        return 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    return u


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999) -> None:
    r"""Update EMA model parameters.

    Supports sharded (DTensor) parameters when `ema_model` and `model` are
    sharded identically (same device mesh and same mixed-precision policy);
    the in-place `mul_`/`add_` then update each local shard with no
    collective. Mixing a sharded model with an unsharded EMA (or different
    meshes) is not supported.

    Args:
        ema_model: Exponential moving average model.
        model: Current model.
        decay: EMA decay rate.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name in ema_params:
            ema_param = ema_params[name]
            if ema_param.device != param.device:
                ema_param.data = ema_param.data.to(param.device)
            ema_param.mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model: nn.Module, flag: bool = True) -> None:
    r"""Set requires_grad flag for all model parameters.

    Args:
        model: Model to modify.
        flag: Whether parameters require gradients.
    """
    for p in model.parameters():
        p.requires_grad = flag


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: str,
    ema_model: Optional[nn.Module] = None,
    args: Optional[Dict[str, Any]] = None,
) -> str:
    r"""Save training checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        step: Current training step.
        checkpoint_dir: Directory for checkpoints.
        ema_model: EMA model (optional).
        args: Additional arguments to save.

    Returns:
        Path to saved checkpoint.
    """
    checkpoint = {
        "model": (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        ),
        "opt": optimizer.state_dict(),
        "step": step,
    }

    if ema_model is not None:
        checkpoint["ema"] = ema_model.state_dict()

    if args is not None:
        checkpoint["args"] = args

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/{step:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    ema_model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    r"""Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load weights into.
        ema_model: EMA model to load (optional).
        optimizer: Optimizer to load state (optional).
        device: Device to map tensors to.

    Returns:
        Dictionary with checkpoint contents.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model" in checkpoint:
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])

    if ema_model is not None and "ema" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema"])

    if optimizer is not None and "opt" in checkpoint:
        optimizer.load_state_dict(checkpoint["opt"])

    return checkpoint


__all__ = [
    "compute_density_for_timestep_sampling",
    "update_ema",
    "requires_grad",
    "save_checkpoint",
    "load_checkpoint",
]
