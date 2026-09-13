import inspect

import torch

import torchebm.losses as losses
from torchebm.core import GaussianModel
from torchebm.samplers import LangevinDynamics


def test_all_public_loss_classes_can_be_constructed():
    model = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))
    sampler = LangevinDynamics(model=model)
    constructors = {
        "ContrastiveDivergence": lambda: losses.ContrastiveDivergence(model, sampler),
        "PersistentContrastiveDivergence": lambda: (
            losses.PersistentContrastiveDivergence(model, sampler)
        ),
        "ScoreMatching": lambda: losses.ScoreMatching(model),
        "DenoisingScoreMatching": lambda: losses.DenoisingScoreMatching(model),
        "SlicedScoreMatching": lambda: losses.SlicedScoreMatching(model),
        "EquilibriumMatchingLoss": lambda: losses.EquilibriumMatchingLoss(model),
        "FlowMatchingLoss": lambda: losses.FlowMatchingLoss(model),
        "EnergyMatchingLoss": lambda: losses.EnergyMatchingLoss(model),
    }
    public_classes = {
        name for name in losses.__all__ if inspect.isclass(getattr(losses, name))
    }

    assert constructors.keys() == public_classes
    for name, constructor in constructors.items():
        assert type(constructor()).__name__ == name
