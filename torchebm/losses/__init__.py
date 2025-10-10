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
    DenoisingScoreMatching,
    SlicedScoreMatching,
)

__all__ = [
    "ContrastiveDivergence",
    "PersistentContrastiveDivergence",
    "ParallelTemperingCD",
    "ScoreMatching",
    "DenoisingScoreMatching",
    "SlicedScoreMatching",
]
