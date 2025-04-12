"""
Core functionality for energy-based models, including energy functions, base sampler class, and training utilities.
"""

from .base_energy_function import (
    BaseEnergyFunction,
    DoubleWellEnergy,
    GaussianEnergy,
    HarmonicEnergy,
    RastriginEnergy,
    AckleyEnergy,
    RosenbrockEnergy,
)

from .base_sampler import BaseSampler
from .base_loss import BaseLoss, BaseContrastiveDivergence

# from .trainer import Trainer
from .base_optimizer import Optimizer

__all__ = [
    # Energy functions
    "BaseEnergyFunction",
    "DoubleWellEnergy",
    "GaussianEnergy",
    "HarmonicEnergy",
    "RastriginEnergy",
    "AckleyEnergy",
    "RosenbrockEnergy",
    # Base classes and utilities
    "BaseSampler",
    "BaseLoss",
    "BaseContrastiveDivergence",
    # "Trainer",
    # "get_optimizer",
    # "score_matching_loss",
]

# todo: Add ODE solver package
