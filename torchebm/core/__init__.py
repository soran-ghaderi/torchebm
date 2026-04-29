"""
Core functionality for energy-based models, including energy functions, base sampler class, and training utilities.
"""

from .device_mixin import DeviceMixin, normalize_device, safe_to
from .base_module import TorchEBMModule

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
from .schedulable import Schedulable
from .base_sampler import BaseSampler

from .base_loss import BaseLoss, BaseContrastiveDivergence, BaseScoreMatching

from .base_integrator import BaseIntegrator, BaseRungeKuttaIntegrator, BaseSDERungeKuttaIntegrator
from .base_interpolant import BaseInterpolant, expand_t_like_x
# from .trainer import Trainer
# from .base_optimizer import Optimizer

__all__ = [
    # Energy functions
    "DeviceMixin",
    "TorchEBMModule",
    "normalize_device",
    "safe_to",
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
    "BaseScoreMatching",
    "BaseContrastiveDivergence",
    "BaseIntegrator",
    "BaseRungeKuttaIntegrator",
    "BaseSDERungeKuttaIntegrator",
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
    "MultiStepScheduler",
    "WarmupScheduler",
    "Schedulable",
]

# todo: Add ODE solver package
