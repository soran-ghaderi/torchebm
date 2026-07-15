"""
Core functionality for energy-based models, including energy functions, base sampler class, and training utilities.
"""

from .base_module import TorchEBMModule, warn_once

from .base_model import (
    BaseModel,
    DoubleWellModel,
    GaussianModel,
    HarmonicModel,
    RastriginModel,
    AckleyModel,
    RosenbrockModel,
)

from .base_scheduler import (
    BaseScheduler,
    ConstantScheduler,
    ExponentialDecayScheduler,
    LinearScheduler,
    CosineScheduler,
    MultiStepScheduler,
    WarmupScheduler,
    TemperatureScheduler,
)
from .schedulable import Schedulable
from .base_sampler import BaseSampler
from .base_coupling import (
    BaseCoupling,
    BaseCostCoupling,
    BaseModelCoupling,
    CouplingResult,
)

from .base_loss import BaseLoss, BaseContrastiveDivergence, BaseScoreMatching

from .base_integrator import (
    BaseIntegrator,
    BaseRungeKuttaIntegrator,
    BaseSDERungeKuttaIntegrator,
    BaseSymplecticIntegrator,
)
from .base_interpolant import BaseInterpolant, expand_t_like_x
# from .trainer import Trainer
# from .base_optimizer import Optimizer

__all__ = [
    # Energy functions
    "TorchEBMModule",
    "warn_once",
    "BaseModel",
    "DoubleWellModel",
    "GaussianModel",
    "HarmonicModel",
    "RastriginModel",
    "AckleyModel",
    "RosenbrockModel",
    # Base classes and utilities
    "BaseSampler",
    "BaseLoss",
    "BaseScoreMatching",
    "BaseContrastiveDivergence",
    "BaseIntegrator",
    "BaseRungeKuttaIntegrator",
    "BaseSDERungeKuttaIntegrator",
    "BaseSymplecticIntegrator",
    "BaseInterpolant",
    "BaseCoupling",
    "BaseCostCoupling",
    "BaseModelCoupling",
    "CouplingResult",
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
    "TemperatureScheduler",
    "Schedulable",
]

# todo: Add ODE solver package
