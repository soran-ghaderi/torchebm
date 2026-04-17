"""
Loss functions for training energy-based models and generative models.
"""

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

_LAZY_IMPORTS = {
    "ContrastiveDivergence": ".contrastive_divergence",
    "PersistentContrastiveDivergence": ".contrastive_divergence",
    "ParallelTemperingCD": ".contrastive_divergence",
    "ScoreMatching": ".score_matching",
    "DenoisingScoreMatching": ".score_matching",
    "SlicedScoreMatching": ".score_matching",
    "EquilibriumMatchingLoss": ".equilibrium_matching",
    "mean_flat": ".loss_utils",
    "get_interpolant": ".loss_utils",
    "compute_eqm_ct": ".loss_utils",
    "dispersive_loss": ".loss_utils",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
