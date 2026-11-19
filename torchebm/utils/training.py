r"""General training utilities for TorchEBM."""

import math
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode


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


def sample_time_pairs(
    batch_size: int,
    weighting_scheme: str = "logit_normal",
    logit_mean: float = 0.4,
    logit_std: float = 1.0,
    flow_matching_ratio: float = 0.75,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Draw time pairs \(s \le t\) for training two-time models.

    The `sample_tr` of MeanFlow (Geng et al., 2025) in torchebm time, with
    noise at 0 and data at 1: two times are drawn independently from
    `compute_density_for_timestep_sampling`, the smaller is the time \(s\)
    of the input and the larger the target time \(t\), and the first
    ``int(batch_size * flow_matching_ratio)`` pairs get \(t = s\), where the
    MeanFlow loss reduces to flow matching. The defaults are the MeanFlow
    ImageNet settings, a logit-normal with \(P_{mean} = -0.4\),
    \(P_{std} = 1\) and 25% of pairs with \(s \ne t\); flipping time negates
    the logit mean.

    Args:
        batch_size: Number of pairs.
        weighting_scheme: Density of each draw: ``"uniform"``,
            ``"logit_normal"`` (default), or ``"mode"``.
        logit_mean: Mean of the logit-normal density, in torchebm time.
        logit_std: Standard deviation of the logit-normal density.
        flow_matching_ratio: Share of pairs with \(t = s\).
        device: Device of the result.
        dtype: Data type of the result.
        generator: RNG for the draws; the global RNG when ``None``.

    Returns:
        Tuple ``(s, t)`` of tensors of shape (batch_size,) with ``s <= t``.

    Example:
        ```python
        from torchebm.utils import sample_time_pairs

        s, t = sample_time_pairs(len(x1))
        ```
    """
    draws = compute_density_for_timestep_sampling(
        weighting_scheme,
        2 * batch_size,
        logit_mean,
        logit_std,
        device=device,
        dtype=dtype,
        generator=generator,
    ).view(2, batch_size)
    s, t = draws.aminmax(dim=0)
    equal = torch.arange(batch_size, device=s.device) < int(batch_size * flow_matching_ratio)
    return s, torch.where(equal, s, t)


def sample_guidance(
    batch_size: int,
    low: float = 1.0,
    high: float = 8.0,
    power: float = 1.0,
    flow_matching_ratio: float = 0.75,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Draw guidance scales and guidance intervals for models that take them as inputs.

    The scale follows \(p(\omega) \propto \omega^{-\beta}\) on
    \([\omega_{low}, \omega_{high}]\), drawn by inverting its distribution
    function; the start of the interval is uniform on \([0, 1/2]\) and its
    end uniform on \([1/2, 1]\), the same in torchebm time as in the
    reference code. The first ``int(batch_size * flow_matching_ratio)`` rows,
    the pairs with \(t = s\) of `sample_time_pairs` at the same ratio, get
    the full interval \([0, 1]\).

    The defaults of ``low``, ``high`` and ``power`` are those of Improved
    MeanFlow (Lyy-iiis/imeanflow, ``sample_cfg_scale`` with
    ``s_max = 7`` and ``cfg_beta = 1``, and ``sample_cfg_interval``), which
    uses ``flow_matching_ratio=0.5``. Drifting models (Deng et al., 2026)
    draw the scale on \([1, 4]\) with \(\beta = 1\) and use no interval.

    Args:
        batch_size: Number of draws.
        low: Smallest scale \(\omega_{low}\).
        high: Largest scale \(\omega_{high}\).
        power: Exponent \(\beta\) of the density of the scale.
        flow_matching_ratio: Share of leading rows with the full interval.
        device: Device of the result.
        dtype: Data type of the result.
        generator: RNG for the draws; the global RNG when ``None``.

    Returns:
        Tuple ``(guidance_scale, guidance_interval)`` of shapes
        ``(batch_size,)`` and ``(batch_size, 2)``, the interval as start and
        end times.

    Example:
        The guided velocity of Improved MeanFlow, with guidance only inside
        the interval and the model's own conditional and unconditional
        velocities, in one batched forward pass:

        ```python
        from torchebm.utils import sample_guidance, sample_time_pairs

        s, t = sample_time_pairs(len(x1), flow_matching_ratio=0.5)
        scale, interval = sample_guidance(len(x1), flow_matching_ratio=0.5)
        x_s, v = path(torch.randn_like(x1), x1, s)
        inside = (s >= interval[:, 0]) & (s <= interval[:, 1])
        w = torch.where(inside, scale, 1.0)
        with torch.no_grad():
            v_c, v_u = model(
                x_s.repeat(2, 1, 1, 1),
                s.repeat(2),
                torch.cat([y, torch.full_like(y, 1000)]),
                s=s.repeat(2),
                guidance_scale=torch.cat([w, torch.ones_like(w)]),
                guidance_interval=torch.tensor([0.0, 1.0]).expand(2 * len(x1), 2),
            ).chunk(2)
        v_guided = v + (1 - 1 / w).view(-1, 1, 1, 1) * (v_c - v_u)
        ```
    """
    u = torch.rand(3, batch_size, device=device, dtype=dtype, generator=generator)
    if power == 1.0:
        scale = low * (high / low) ** u[0]
    else:
        p = 1.0 - power
        scale = (low**p + u[0] * (high**p - low**p)) ** (1.0 / p)
    full = torch.arange(batch_size, device=u.device) < int(batch_size * flow_matching_ratio)
    start = torch.where(full, 0.0, 0.5 * u[1])
    end = torch.where(full, 1.0, 0.5 + 0.5 * u[2])
    return scale, torch.stack([start, end], dim=-1)


class _LinearTangentMode(TorchFunctionMode):
    r"""Forward-mode tangent of `F.linear` with a tangent on the input only.

    The tangent is \(\dot y = \dot x W^\top\), one matrix product computed
    without gradient. PyTorch's forward-mode rule also multiplies the input
    by the zero tangent of the weight, a second matrix product per layer
    under `torch.compile`, and records the tangent for the backward pass.
    Calls whose weight or bias has a tangent run unchanged.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.linear:
            return self._linear(*args, **kwargs)
        return func(*args, **kwargs)

    @staticmethod
    def _linear(input, weight, bias=None):
        primal, tangent = fwAD.unpack_dual(input, level=0)
        if tangent is None or any(
            fwAD.unpack_dual(p, level=0).tangent is not None for p in (weight, bias) if p is not None
        ):
            return F.linear(input, weight, bias)
        with torch.no_grad():
            output_t = F.linear(tangent, weight)
        return fwAD.make_dual(F.linear(primal, weight, bias), output_t, level=0)


def average_velocity_jvp(
    model: nn.Module,
    x_s: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    tangent: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    flow_matching_ratio: float = 0.75,
    model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    aux_head: bool = False,
) -> Tuple[torch.Tensor, ...]:
    r"""Average velocity of a two-time model and its JVP, the inputs of `MeanFlowLoss`.

    Returns \(u = u_\theta(x_s, s, t)\) and its total derivative along the
    path at fixed \(t\),

    \[
    \frac{d u}{d s} = \partial_x u \; \dot x + \partial_s u,
    \]

    the JVP of the model along ``(tangent, 1)`` in \((x_s, s)\). The tangent
    \(\dot x\) is the path velocity in MeanFlow (Geng et al., 2025) and the
    model velocity \(u_\theta(x_s, s, s)\) in Improved MeanFlow (Geng et al.,
    2025b, Algorithm 1).

    The first ``int(B * flow_matching_ratio)`` pairs must have \(t = s\), as
    `sample_time_pairs` returns them with the same ratio. There the MeanFlow
    target \(v + (t - s) \frac{du}{ds}\) is \(v\), so these rows run the
    forward pass only, their derivative is zero, and `torch.func.jvp` runs
    on the remaining rows. At the MeanFlow ratio of 75% this skips three
    quarters of the forward-mode work of a full-batch JVP; a ratio of 0 runs
    the JVP on the full batch. The JVP rows are copied out of the batch
    first, as `torch.compile` does not trace forward-mode AD through views.

    The derivative carries no gradient, as MeanFlow and Improved MeanFlow use
    it under stop-gradient: the tangents of linear layers take one matrix
    product each and are not recorded, so the backward pass keeps no tangent
    activations and, under `torch.compile`, no product with the zero tangent
    of the weights is formed.

    With ``aux_head=True`` the model is called with ``head="both"`` and the
    output of its auxiliary velocity head (Improved MeanFlow, App. A) comes
    from the same forward pass, with gradient and without JVP. In eager mode
    the tangents of the auxiliary blocks are still computed and discarded;
    `torch.compile` removes them.

    Args:
        model: Two-time model, called as
            ``model(x_s, t, y, s=s, **model_kwargs)``.
        x_s: Inputs at time \(s\), of shape ``(B, ...)``.
        s: Times of ``x_s``, of shape ``(B,)``.
        t: Target times, of shape ``(B,)``.
        tangent: Tangent \(\dot x\) of ``x_s``, of the shape of ``x_s``.
        y: Class labels of shape ``(B,)``, or ``None``.
        flow_matching_ratio: Share of leading pairs with \(t = s\).
        model_kwargs: Further model inputs with the batch on the first
            dimension, such as ``guidance_scale`` and ``guidance_interval``,
            split with the rows.
        aux_head: Also return the output of the auxiliary velocity head, for
            models built with ``aux_head_depth``.

    Returns:
        Tuple ``(u, du_ds)`` of tensors of the shape of ``x_s``, or
        ``(u, du_ds, v)`` with the auxiliary output ``v`` for
        ``aux_head=True``; ``du_ds`` is detached.

    Example:
        ```python
        from torchebm.utils import average_velocity_jvp, sample_time_pairs

        s, t = sample_time_pairs(len(x1))
        x_s, v = path(torch.randn_like(x1), x1, s)
        u, du_ds = average_velocity_jvp(model, x_s, s, t, v, y)
        loss = loss_fn(u, du_ds, v, s, t)
        ```
    """
    n = int(len(x_s) * flow_matching_ratio)
    kwargs = model_kwargs or {}
    rows = (x_s, s, t, tangent, y)
    jvp_kwargs = kwargs
    if n > 0:
        rows = tuple(None if a is None else a[n:].clone() for a in rows)
        jvp_kwargs = {k: v[n:].clone() for k, v in kwargs.items()}
    x_jvp, s_jvp, t_jvp, tangent_jvp, y_jvp = rows
    head = {"head": "both"} if aux_head else {}
    with _LinearTangentMode():
        u, du_ds, *aux = torch.func.jvp(
            lambda x_s, s: model(x_s, t_jvp, y_jvp, s=s, **jvp_kwargs, **head),
            (x_jvp, s_jvp),
            (tangent_jvp, torch.ones_like(s_jvp)),
            has_aux=aux_head,
        )
    du_ds = du_ds.detach()
    if n > 0:
        fm_kwargs = {k: v[:n] for k, v in kwargs.items()}
        fm = model(x_s[:n], t[:n], None if y is None else y[:n], s=s[:n], **fm_kwargs, **head)
        u_fm, *aux_fm = fm if aux_head else (fm,)
        u, du_ds = torch.cat([u_fm, u]), torch.cat([torch.zeros_like(u_fm), du_ds])
        aux = [torch.cat(pair) for pair in zip(aux_fm, aux)]
    return (u, du_ds, *aux)


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
