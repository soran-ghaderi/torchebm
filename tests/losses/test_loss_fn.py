"""Loss classes as criteria on model outputs, and the deprecated model-bound mode."""

import pytest
import torch
from torch import nn

import torchebm.losses.functional as F
from torchebm._deprecation import _WARNED_ONCE, TorchEBMDeprecationWarning
from torchebm.core import BaseModel
from torchebm.losses import (
    ContrastiveDivergence,
    ContrastiveDivergenceLoss,
    DenoisingScoreMatching,
    DenoisingScoreMatchingLoss,
    EnergyMatchingLoss,
    EquilibriumMatchingLoss,
    FlowMatchingLoss,
    PersistentContrastiveDivergence,
    ScoreMatching,
    ScoreMatchingLoss,
    SlicedScoreMatching,
    SlicedScoreMatchingLoss,
    compute_eqm_ct,
)
from torchebm.samplers import LangevinDynamics


class Quadratic(BaseModel):
    def forward(self, x):
        return 0.5 * x.square().sum(dim=-1)


class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x, t, **kwargs):
        return self.linear(x)


def test_contrastive_divergence_criterion():
    positive, negative = torch.randn(8), torch.randn(8)
    criterion = ContrastiveDivergenceLoss(energy_reg_weight=0.1, trim_fraction=0.25)
    expected = F.contrastive_divergence_loss(
        positive, negative, energy_reg_weight=0.1, trim_fraction=0.25
    )
    assert torch.equal(criterion(positive, negative), expected)


def test_contrastive_divergence_clamp_floors_the_loss():
    criterion = ContrastiveDivergenceLoss(clamp=0.5)
    assert criterion(torch.zeros(4), torch.full((4,), 10.0)) == -0.5


@pytest.mark.parametrize(
    "kwargs", [{"energy_reg_weight": -1.0}, {"trim_fraction": 1.0}, {"clamp": -1.0}]
)
def test_contrastive_divergence_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        ContrastiveDivergenceLoss(**kwargs)


def test_model_bound_contrastive_divergence_is_deprecated():
    model = Quadratic()
    sampler = LangevinDynamics(model, step_size=0.1)
    with pytest.warns(TorchEBMDeprecationWarning, match="ContrastiveDivergenceLoss"):
        ContrastiveDivergence(model, sampler)
    with pytest.warns(TorchEBMDeprecationWarning):
        PersistentContrastiveDivergence(model, sampler)


def test_flow_matching_criterion():
    input, target, weight = torch.randn(4, 2), torch.randn(4, 2), torch.rand(4)
    per_sample = FlowMatchingLoss(reduction="none")(input, target, weight)
    assert torch.equal(per_sample, F.flow_matching_loss(input, target, weight, "none"))
    assert torch.equal(
        FlowMatchingLoss()(input, target), F.flow_matching_loss(input, target)
    )


def test_flow_matching_criterion_requires_target():
    with pytest.raises(TypeError, match="target"):
        FlowMatchingLoss()(torch.randn(4, 2))


def test_equilibrium_matching_criterion_scales_target_by_ct():
    input, target, t = torch.randn(4, 2), torch.randn(4, 2), torch.rand(4)
    criterion = EquilibriumMatchingLoss(ct_threshold=0.5, ct_multiplier=2.0)
    ct = compute_eqm_ct(t, threshold=0.5, multiplier=2.0)
    expected = F.equilibrium_matching_loss(input, target, ct)
    assert torch.equal(criterion(input, target, t), expected)
    constant = EquilibriumMatchingLoss(ct="constant")(input, target, t)
    assert torch.equal(
        constant, F.equilibrium_matching_loss(input, target, torch.full((4,), 4.0))
    )


def test_equilibrium_matching_criterion_requires_target_and_t():
    with pytest.raises(TypeError, match="target"):
        EquilibriumMatchingLoss()(torch.randn(4, 2), torch.randn(4, 2))


def test_energy_matching_criterion_warm_up_and_joint_phase():
    input, target, t = torch.randn(4, 2), torch.randn(4, 2), torch.rand(4)
    positive, negative = torch.randn(4), torch.randn(6)
    criterion = EnergyMatchingLoss(lambda_cd=2.0, cd_trim_fraction=0.0, cd_clamp=None)
    flow = F.energy_matching_loss(input, target, t)
    assert torch.equal(criterion(input, target, t), flow)
    cd = 2.0 * F.contrastive_divergence_loss(positive, negative)
    assert torch.equal(criterion(input, target, t, positive, negative), flow + cd)


def test_energy_matching_criterion_clamps_the_contrastive_term():
    zeros = torch.zeros(4, 2)
    criterion = EnergyMatchingLoss(lambda_cd=1.0, cd_clamp=0.02)
    loss = criterion(
        zeros, zeros, torch.zeros(4), torch.zeros(4), torch.full((4,), 10.0)
    )
    assert loss == -0.02


@pytest.mark.parametrize(
    "build",
    [
        lambda: FlowMatchingLoss(model=Field()),
        lambda: EquilibriumMatchingLoss(model=Field()),
        lambda: EnergyMatchingLoss(model=Quadratic(), lambda_cd=0.0),
    ],
    ids=["flow", "equilibrium", "energy"],
)
def test_model_bound_mode_is_deprecated_but_still_runs(build):
    _WARNED_ONCE.clear()
    with pytest.warns(TorchEBMDeprecationWarning, match="deprecated"):
        loss = build()
    assert loss(torch.randn(8, 2)).dim() == 0


def test_score_matching_criterion():
    score, divergence = torch.randn(4, 3), torch.randn(4)
    assert torch.equal(
        ScoreMatchingLoss(reduction="sum")(score, divergence),
        F.score_matching_loss(score, divergence, "sum"),
    )


def test_denoising_score_matching_criterion():
    input, target, weight = torch.randn(4, 3), torch.randn(4, 3), torch.rand(4)
    assert torch.equal(
        DenoisingScoreMatchingLoss()(input, target, weight),
        F.denoising_score_matching_loss(input, target, weight),
    )


def test_sliced_score_matching_criterion_draws_projections_from_generator():
    x = torch.randn(6, 2, requires_grad=True)
    score = -x
    criterion = SlicedScoreMatchingLoss(n_projections=3)
    first = criterion(score, x, generator=torch.Generator().manual_seed(0))
    second = criterion(score, x, generator=torch.Generator().manual_seed(0))
    assert torch.equal(first, second)
    projections = torch.randn(6, 2, generator=torch.Generator().manual_seed(0)).sign()
    expected = F.sliced_score_matching_loss(score, x, projections, n_projections=3)
    assert torch.equal(first, expected)


def test_sliced_score_matching_criterion_validates():
    with pytest.raises(ValueError, match="projection_type"):
        SlicedScoreMatchingLoss(projection_type="bogus")
    with pytest.raises(ValueError, match="n_projections"):
        SlicedScoreMatchingLoss(n_projections=4)(
            torch.zeros(6, 2), torch.zeros(6, 2, requires_grad=True)
        )


@pytest.mark.parametrize(
    "legacy", [ScoreMatching, DenoisingScoreMatching, SlicedScoreMatching]
)
def test_model_bound_score_matching_is_deprecated(legacy):
    with pytest.warns(TorchEBMDeprecationWarning, match="Loss"):
        legacy(Quadratic())


def test_reduction_is_validated():
    with pytest.raises(ValueError, match="reduction"):
        FlowMatchingLoss(reduction="avg")
