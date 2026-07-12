import importlib
from typing import Callable, Optional, Type, Union

import torch

from torchebm.core import BaseIntegrator

_INTEGRATOR_NAMES = {
    "euler": "EulerMaruyamaIntegrator",  # historical FlowSampler string
    "euler_maruyama": "EulerMaruyamaIntegrator",
    "backward_euler_maruyama": "BackwardEulerMaruyamaIntegrator",
    "heun": "HeunIntegrator",
    "adaptive_heun": "AdaptiveHeunIntegrator",
    "bosh3": "Bosh3Integrator",
    "dopri5": "Dopri5Integrator",
    "dopri8": "Dopri8Integrator",
    "rk4": "RK4Integrator",
    "rk438": "RK438Integrator",
    "leapfrog": "LeapfrogIntegrator",
    "generalised_leapfrog": "GeneralisedLeapfrogIntegrator",
    "generalized_leapfrog": "GeneralisedLeapfrogIntegrator",  # US alias
}


def get_integrator(
    name: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> BaseIntegrator:
    r"""Construct an integrator by registry name with class defaults.

    Args:
        name: Registry name, e.g. ``"euler_maruyama"``, ``"leapfrog"``,
            ``"dopri5"``.
        device: Device for the integrator.
        dtype: Data type for the integrator.

    Returns:
        A freshly constructed integrator instance.

    Raises:
        ValueError: If ``name`` is not a known integrator.
    """
    try:
        cls_name = _INTEGRATOR_NAMES[name]
    except (KeyError, TypeError):
        valid = ", ".join(sorted(_INTEGRATOR_NAMES))
        raise ValueError(f"Unknown integrator {name!r}. Valid names: {valid}") from None
    # Resolve through the package's lazy __getattr__ to avoid import cycles.
    cls = getattr(importlib.import_module("torchebm.integrators"), cls_name)
    return cls(device=device, dtype=dtype)


def resolve_integrator(
    integrator: Union[str, BaseIntegrator, None],
    *,
    default: str,
    family: Type[BaseIntegrator],
    owner: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> BaseIntegrator:
    r"""Resolve a sampler's ``integrator`` argument to a validated instance.

    ``None`` and string inputs construct via `get_integrator` with the
    sampler's device/dtype. Instances are used as-is: they are validated
    against ``family`` and their device/dtype must already match the
    sampler's (host-side attribute comparison, no implicit ``.to()``).

    Args:
        integrator: ``None`` (use ``default``), a registry name, or an
            integrator instance.
        default: Registry name used when ``integrator`` is ``None``.
        family: Required base class for instances.
        owner: Sampler class name, used in error messages.
        device: Sampler device.
        dtype: Sampler dtype.

    Returns:
        A validated integrator instance.

    Raises:
        TypeError: If the resolved integrator is not of the required
            ``family`` (validated for names and instances alike).
        ValueError: If an instance's device/dtype mismatches the sampler's.
    """
    if isinstance(integrator, BaseIntegrator):
        if not isinstance(integrator, family):
            raise TypeError(
                f"{owner} requires a {family.__name__}; got "
                f"{type(integrator).__name__}"
            )
        if integrator.device != device or integrator.dtype != dtype:
            raise ValueError(
                f"{owner} device/dtype ({device}, {dtype}) does not match the "
                f"integrator's ({integrator.device}, {integrator.dtype}). "
                f"Construct the integrator with matching device/dtype; no "
                f"implicit .to() is performed."
            )
        return integrator
    resolved = get_integrator(
        integrator if integrator is not None else default,
        device=device,
        dtype=dtype,
    )
    if not isinstance(resolved, family):
        raise TypeError(
            f"{owner} requires a {family.__name__}; got " f"{type(resolved).__name__}"
        )
    return resolved


def _integrate_time_grid(
    x: torch.Tensor,
    t: torch.Tensor,
    step_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    if t.ndim != 1:
        raise ValueError("t must be a 1D tensor")
    if t.numel() < 2:
        raise ValueError("t must have length >= 2")
    for i in range(t.numel() - 1):
        dt = t[i + 1] - t[i]
        t_batch = t[i].expand(x.size(0))
        x = step_fn(x, t_batch, dt)
    return x
