r"""
Stochastic interpolants for generative modeling.

Interpolants define conditional probability paths between source (noise)
and target (data) distributions, parameterized by schedules α(t) and σ(t).
"""

from torchebm.interpolants.linear import LinearInterpolant
from torchebm.interpolants.cosine import CosineInterpolant
from torchebm.interpolants.variance_preserving import VariancePreservingInterpolant

__all__ = [
    "LinearInterpolant",
    "CosineInterpolant",
    "VariancePreservingInterpolant",
]
