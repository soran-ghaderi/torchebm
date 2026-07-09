r"""Registry and resolver for minibatch couplings."""

import importlib
from typing import Type, Union

from torchebm.core import BaseCoupling

_COUPLING_NAMES = {
    "independent": "IndependentCoupling",
    "ot": "ExactOTCoupling",  # historical alias (EnergyMatchingLoss default)
    "exact_ot": "ExactOTCoupling",
    "sinkhorn": "SinkhornCoupling",
    "greedy": "GreedyCoupling",
    "unbalanced_sinkhorn": "UnbalancedSinkhornCoupling",
}


def get_coupling(name: str) -> BaseCoupling:
    r"""Construct a coupling by registry name with class defaults.

    Args:
        name: Registry name, e.g. ``"independent"``, ``"ot"`` /
            ``"exact_ot"``, ``"sinkhorn"``.

    Returns:
        A freshly constructed coupling instance.

    Raises:
        ValueError: If ``name`` is not a known coupling.
    """
    try:
        cls_name = _COUPLING_NAMES[name]
    except (KeyError, TypeError):
        valid = ", ".join(sorted(_COUPLING_NAMES))
        raise ValueError(
            f"Unknown coupling {name!r}. Valid names: {valid}"
        ) from None
    # Resolve through the package's lazy __getattr__ to avoid import cycles.
    cls = getattr(importlib.import_module("torchebm.couplings"), cls_name)
    return cls()


def resolve_coupling(
    coupling: Union[str, BaseCoupling, None],
    *,
    default: str,
    owner: str,
    family: Type[BaseCoupling] = BaseCoupling,
) -> BaseCoupling:
    r"""Resolve a ``coupling`` argument to a validated instance.

    ``None`` and string inputs construct via `get_coupling`. Instances are
    used as-is after validation against ``family``. Couplings are stateless
    and compute on the inputs' device, so no device/dtype handling is needed.

    Args:
        coupling: ``None`` (use ``default``), a registry name, or a coupling
            instance.
        default: Registry name used when ``coupling`` is ``None``.
        owner: Caller class name, used in error messages.
        family: Required base class for instances.

    Returns:
        A validated coupling instance.

    Raises:
        TypeError: If an instance is not of the required ``family``.
    """
    if coupling is None:
        return get_coupling(default)
    if isinstance(coupling, str):
        return get_coupling(coupling)
    if not isinstance(coupling, family):
        raise TypeError(
            f"{owner} requires a {family.__name__}; got "
            f"{type(coupling).__name__}"
        )
    return coupling
