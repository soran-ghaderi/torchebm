r"""Integrators for solving differential equations in energy-based models.

Integrators are lazy-loaded to avoid importing all 8 submodules at
package import time.  Direct imports still work:
``from torchebm.integrators import LeapfrogIntegrator``.
"""

__all__ = [
    "EulerMaruyamaIntegrator",
    "HeunIntegrator",
    "LeapfrogIntegrator",
    "Dopri5Integrator",
    "Dopri8Integrator",
    "RK4Integrator",
    "AdaptiveHeunIntegrator",
    "Bosh3Integrator",
    "_integrate_time_grid",
]

_LAZY_IMPORTS = {
    "_integrate_time_grid": ".integrator_utils",
    "EulerMaruyamaIntegrator": ".euler_maruyama",
    "HeunIntegrator": ".heun",
    "LeapfrogIntegrator": ".leapfrog",
    "Dopri5Integrator": ".dopri",
    "Dopri8Integrator": ".dopri",
    "RK4Integrator": ".rk4",
    "AdaptiveHeunIntegrator": ".adaptive_heun",
    "Bosh3Integrator": ".bosh3",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
