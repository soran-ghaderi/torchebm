"""
Core functionality for energy-based models, including energy functions, base sampler class, and training utilities.
"""

from .device_mixin import DeviceMixin, normalize_device

from .base_model import (
    BaseModel,
    DoubleWellModel,
    GaussianModel,
    HarmonicModel,
    RastriginModel,
    AckleyModel,
    RosenbrockModel,
    # deprecated
    BaseEnergyFunction,
    DoubleWellEnergy,
    GaussianEnergy,
    HarmonicEnergy,
    RastriginEnergy,
    AckleyEnergy,
    RosenbrockEnergy,
)

from .base_scheduler import (
    BaseScheduler,
    ConstantScheduler,
    ExponentialDecayScheduler,
    LinearScheduler,
    CosineScheduler,
    MultiStepScheduler,
    WarmupScheduler,
)
from .base_sampler import BaseSampler

from .base_loss import BaseLoss, BaseContrastiveDivergence

from .base_integrator import BaseIntegrator
from .base_interpolant import BaseInterpolant, expand_t_like_x
# from .trainer import Trainer
# from .base_optimizer import Optimizer

__all__ = [
    # Energy functions
    "DeviceMixin",
    "normalize_device",
    "BaseModel",
    "DoubleWellModel",
    "GaussianModel",
    "HarmonicModel",
    "RastriginModel",
    "AckleyModel",
    "RosenbrockModel",
    # deprecated
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
    "BaseIntegrator",
    "BaseInterpolant",
    "expand_t_like_x",
    # "Trainer",
    # "get_optimizer",
    # "score_matching_loss",
    "BaseScheduler",
    "ConstantScheduler",
    "ExponentialDecayScheduler",
    "LinearScheduler",
    "CosineScheduler",
]

# todo: Add ODE solver package
