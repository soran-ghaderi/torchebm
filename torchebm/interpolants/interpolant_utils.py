import importlib
from typing import Union

from torchebm.core import BaseInterpolant

_INTERPOLANT_NAMES = {
    "linear": "LinearInterpolant",
    "cosine": "CosineInterpolant",
    "vp": "VariancePreservingInterpolant",
}


def get_interpolant(interpolant_type: str) -> BaseInterpolant:
    r"""Get interpolant instance by name.

    Args:
        interpolant_type: One of 'linear', 'cosine', or 'vp'.

    Returns:
        Interpolant instance.

    Raises:
        ValueError: If interpolant_type is not recognized.
    """
    try:
        cls_name = _INTERPOLANT_NAMES[interpolant_type]
    except (KeyError, TypeError):
        valid = list(_INTERPOLANT_NAMES)
        raise ValueError(
            f"Unknown interpolant: {interpolant_type}. Choose from {valid}"
        ) from None
    # Resolve through the package's lazy __getattr__ to avoid import cycles.
    cls = getattr(importlib.import_module("torchebm.interpolants"), cls_name)
    return cls()


def resolve_interpolant(
    interpolant: Union[str, BaseInterpolant, None],
    *,
    default: str,
    owner: str,
) -> BaseInterpolant:
    r"""Resolve an ``interpolant`` argument to a validated instance.

    ``None`` and string inputs construct via `get_interpolant`. Instances
    are validated against `BaseInterpolant` and used as-is. Interpolants
    are stateless math objects, so no device/dtype handling is needed.

    Args:
        interpolant: ``None`` (use ``default``), a registry name, or an
            interpolant instance.
        default: Registry name used when ``interpolant`` is ``None``.
        owner: Owning class name, used in error messages.

    Returns:
        A validated interpolant instance.

    Raises:
        TypeError: If an instance is not a `BaseInterpolant`.
    """
    if interpolant is None:
        return get_interpolant(default)
    if isinstance(interpolant, str):
        return get_interpolant(interpolant)
    if not isinstance(interpolant, BaseInterpolant):
        raise TypeError(
            f"{owner} requires a BaseInterpolant; got " f"{type(interpolant).__name__}"
        )
    return interpolant
