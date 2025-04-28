"""
BaseLoss functions for training energy-based models, including contrastive divergence variants.
"""

from .contrastive_divergence import (
    ContrastiveDivergence,
    PersistentContrastiveDivergence,
    ParallelTemperingCD,
)
from torchebm.losses.score_matching import (
    ScoreMatching,
    DenosingScoreMatching,
    SlicedScoreMatching,
)

__all__ = [
    "ContrastiveDivergence",
    "PersistentContrastiveDivergence",
    "ParallelTemperingCD",
    "ScoreMatching",
    "DenosingScoreMatching",
    "SlicedScoreMatching",
]
