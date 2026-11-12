"""
Loss functions for training energy-based models and generative models.
"""

__all__ = [
    # Contrastive Divergence
    "ContrastiveDivergenceLoss",
    "ContrastiveDivergence",
    "PersistentContrastiveDivergence",
    # Score Matching
    "ScoreMatchingLoss",
    "DenoisingScoreMatchingLoss",
    "SlicedScoreMatchingLoss",
    "ScoreMatching",
    "DenoisingScoreMatching",
    "SlicedScoreMatching",
    # Equilibrium Matching
    "EquilibriumMatchingLoss",
    # Flow Matching
    "FlowMatchingLoss",
    # Energy Matching
    "EnergyMatchingLoss",
    # Noise-Contrastive Estimation
    "NoiseContrastiveEstimation",
    "GaussianNoise",
    "GaussianMixtureNoise",
    # Utilities
    "mean_flat",
    "get_interpolant",
    "trimmed_mean",
    "compute_flow_weight",
    "compute_eqm_ct",
    "dispersive_loss",
]

_LAZY_IMPORTS = {
    "ContrastiveDivergenceLoss": ".contrastive_divergence",
    "ContrastiveDivergence": ".contrastive_divergence",
    "PersistentContrastiveDivergence": ".contrastive_divergence",
    "ScoreMatchingLoss": ".score_matching",
    "DenoisingScoreMatchingLoss": ".score_matching",
    "SlicedScoreMatchingLoss": ".score_matching",
    "ScoreMatching": ".score_matching",
    "DenoisingScoreMatching": ".score_matching",
    "SlicedScoreMatching": ".score_matching",
    "EquilibriumMatchingLoss": ".equilibrium_matching",
    "FlowMatchingLoss": ".flow_matching",
    "EnergyMatchingLoss": ".energy_matching",
    "NoiseContrastiveEstimation": ".noise_contrastive_estimation",
    "GaussianNoise": ".noise_contrastive_estimation",
    "GaussianMixtureNoise": ".noise_contrastive_estimation",
    "mean_flat": ".loss_utils",
    "get_interpolant": ".loss_utils",
    "trimmed_mean": ".loss_utils",
    "compute_flow_weight": ".loss_utils",
    "compute_eqm_ct": ".loss_utils",
    "dispersive_loss": ".loss_utils",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __package__)
        return getattr(module, name)
    if name == "functional":
        import importlib

        return importlib.import_module(".functional", __package__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
