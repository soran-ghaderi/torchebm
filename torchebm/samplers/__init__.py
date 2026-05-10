r"""
Sampling algorithms for energy-based models and generative models.

Includes:
- MCMC samplers (Langevin dynamics, HMC) for energy-based models
- Gradient-based optimization samplers for energy minimization
- Flow/diffusion samplers for trained generative models
"""

__all__ = [
    # MCMC samplers
    "LangevinDynamics",
    "HamiltonianMonteCarlo",
    # Optimization-based samplers
    "GradientDescentSampler",
    "NesterovSampler",
    # Flow/diffusion samplers
    "FlowSampler",
    "PredictionType",
]

_LAZY_IMPORTS = {
    "LangevinDynamics": ".langevin_dynamics",
    "HamiltonianMonteCarlo": ".hmc",
    "GradientDescentSampler": ".gradient_descent",
    "NesterovSampler": ".gradient_descent",
    "FlowSampler": ".flow",
    "PredictionType": ".flow",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
