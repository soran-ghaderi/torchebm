r"""Minibatch couplings for pairing source and target samples.

A coupling reorders (or resamples) a minibatch of source samples \(x_0\)
against targets \(x_1\) before interpolation, so that transport happens along
efficient pairs (OT-CFM, rectified flow, Energy Matching warm-up).

Couplings are lazy-loaded to avoid importing every submodule at package import
time. Direct imports still work:
``from torchebm.couplings import SinkhornCoupling``.
"""

__all__ = [
    "get_coupling",
    "resolve_coupling",
    "CouplingResult",
    "IndependentCoupling",
    "ExactOTCoupling",
    "SinkhornCoupling",
    "GreedyCoupling",
    "UnbalancedSinkhornCoupling",
    "ReflowCoupling",
]

_LAZY_IMPORTS = {
    "get_coupling": ".coupling_utils",
    "resolve_coupling": ".coupling_utils",
    "CouplingResult": "torchebm.core.base_coupling",
    "IndependentCoupling": ".independent",
    "ExactOTCoupling": ".ot",
    "SinkhornCoupling": ".ot",
    "GreedyCoupling": ".ot",
    "UnbalancedSinkhornCoupling": ".ot",
    "ReflowCoupling": ".model_induced",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
