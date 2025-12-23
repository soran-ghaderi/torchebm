"""
Loss functions for training energy-based models and generative models.
"""

from .loss_utils import (
    mean_flat,
    get_interpolant,
    compute_eqm_ct,
    dispersive_loss,
)

from .contrastive_divergence import (
    ContrastiveDivergence,
    PersistentContrastiveDivergence,
    ParallelTemperingCD,
)
from .score_matching import (
    ScoreMatching,
    DenoisingScoreMatching,
    SlicedScoreMatching,
)
from .equilibrium_matching import EquilibriumMatchingLoss

__all__ = [
    # Contrastive Divergence
    "ContrastiveDivergence",
    "PersistentContrastiveDivergence",
    "ParallelTemperingCD",
    # Score Matching
    "ScoreMatching",
    "DenoisingScoreMatching",
    "SlicedScoreMatching",
    # Equilibrium Matching
    "EquilibriumMatchingLoss",
    # Utilities
    "mean_flat",
    "get_interpolant",
    "compute_eqm_ct",
    "dispersive_loss",
]
