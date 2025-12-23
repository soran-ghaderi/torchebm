r"""
Sampling algorithms for energy-based models and generative models.

Includes:
- MCMC samplers (Langevin dynamics, HMC) for energy-based models
- Gradient-based optimization samplers for energy minimization
- Flow/diffusion samplers for trained generative models
"""

from . import hmc, langevin_dynamics, gradient_descent, flow
from .langevin_dynamics import LangevinDynamics
from .hmc import HamiltonianMonteCarlo
from .gradient_descent import GradientDescentSampler, NesterovSampler
from .flow import FlowSampler, PredictionType

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
