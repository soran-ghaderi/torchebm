r"""
Stochastic interpolants for generative modeling.

Interpolants define conditional probability paths between source (noise)
and target (data) distributions, parameterized by schedules α(t) and σ(t).
"""

from torchebm.core.base_interpolant import BaseInterpolant, expand_t_like_x
from torchebm.interpolants.linear import LinearInterpolant
from torchebm.interpolants.cosine import CosineInterpolant
from torchebm.interpolants.variance_preserving import VariancePreservingInterpolant

__all__ = [
    "BaseInterpolant",
    "LinearInterpolant",
    "CosineInterpolant",
    "VariancePreservingInterpolant",
    "expand_t_like_x",
]
