"""torchebm.losses.functional against closed forms and the loss classes built on it."""

import pytest
import torch
from torch import nn

import torchebm.losses.functional as F
from torchebm.core import BaseModel
from torchebm.interpolants import LinearInterpolant
from torchebm.losses import (
    ContrastiveDivergence,
    DenoisingScoreMatching,
    EnergyMatchingLoss,
    EquilibriumMatchingLoss,
    FlowMatchingLoss,
    ScoreMatching,
    SlicedScoreMatching,
    compute_eqm_ct,
    compute_flow_weight,
)
from torchebm.samplers import LangevinDynamics


class Quadratic(BaseModel):
    """E(x) = ||x||^2 / 2: the score is -x and its divergence is -dim."""

    def forward(self, x):
        return 0.5 * x.square().sum(dim=-1)


class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x, t, **kwargs):
        return self.linear(x)


def fixed_t(values):
    return lambda batch, *, device, dtype, generator: values.to(device, dtype)


def test_contrastive_divergence_loss_formula():
    positive, negative = torch.tensor([1.0, 3.0]), torch.tensor([2.0, 6.0])
    assert F.contrastive_divergence_loss(positive, negative) == -2.0
    regularized = F.contrastive_divergence_loss(
        positive, negative, energy_reg_weight=0.1
    )
    assert torch.isclose(regularized, torch.tensor(-2.0 + 0.1 * (5.0 + 20.0)))


def test_contrastive_divergence_loss_trims_largest_negative_energies():
    positive, negative = torch.zeros(4), torch.tensor([1.0, 2.0, 3.0, 100.0])
    assert F.contrastive_divergence_loss(positive, negative, trim_fraction=0.25) == -2.0


def test_contrastive_divergence_loss_matches_class():
    torch.manual_seed(0)
    model = Quadratic()
    cd = ContrastiveDivergence(
        model, LangevinDynamics(model, step_size=0.1), energy_reg_weight=1e-3
    )
    x, x_neg = torch.randn(8, 2), torch.randn(8, 2)
    expected = F.contrastive_divergence_loss(
        model(x), model(x_neg), energy_reg_weight=1e-3
    )
    assert torch.equal(cd.compute_loss(x, x_neg), expected)


def test_flow_matching_loss_is_mse_over_features():
    input, target = torch.randn(4, 3, 5), torch.randn(4, 3, 5)
    per_sample = F.flow_matching_loss(input, target, reduction="none")
    assert per_sample.shape == (4,)
    assert torch.allclose(per_sample, (input - target).square().flatten(1).mean(dim=1))
    assert torch.allclose(
        F.flow_matching_loss(input, target), nn.functional.mse_loss(input, target)
    )
    assert torch.allclose(
        F.flow_matching_loss(input, target, reduction="sum"), per_sample.sum()
    )


def test_flow_matching_loss_weight_is_per_sample():
    input, target = torch.randn(4, 3), torch.randn(4, 3)
    weight = torch.tensor([1.0, 2.0, 0.0, 0.5])
    weighted = F.flow_matching_loss(input, target, weight=weight, reduction="none")
    assert torch.allclose(
        weighted, F.flow_matching_loss(input, target, reduction="none") * weight
    )


def test_flow_matching_loss_matches_class():
    torch.manual_seed(0)
    field, path = Field(), LinearInterpolant()
    x0, x1, t = torch.randn(8, 2), torch.randn(8, 2), torch.rand(8)
    loss = FlowMatchingLoss(field, t_sampler=fixed_t(t), loss_weight_fn=lambda t: 1 + t)
    x_t, u_t = path.interpolate(x0, x1, t)
    expected = F.flow_matching_loss(field(x_t, t), u_t, weight=1 + t, reduction="none")
    assert torch.equal(loss.training_losses(x1, x0=x0)["loss"], expected)


def test_equilibrium_matching_loss_scales_target_by_ct():
    input, target = torch.randn(4, 3), torch.randn(4, 3)
    ct = torch.tensor([4.0, 2.0, 1.0, 0.5])
    expected = (input - target * ct[:, None]).square().mean(dim=1)
    assert torch.allclose(
        F.equilibrium_matching_loss(input, target, ct, reduction="none"), expected
    )


def test_equilibrium_matching_loss_matches_class():
    torch.manual_seed(0)
    field, path = Field(), LinearInterpolant()
    x0, x1, t = torch.randn(8, 2), torch.randn(8, 2), torch.rand(8)
    loss = EquilibriumMatchingLoss(field, energy_type="none", t_sampler=fixed_t(t))
    x_t, u_t = path.interpolate(x0, x1, t)
    expected = F.equilibrium_matching_loss(
        field(x_t, torch.zeros_like(t)), -u_t, compute_eqm_ct(t), reduction="none"
    )
    assert torch.equal(loss.training_losses(x1, x0=x0)["loss"], expected)


def test_energy_matching_loss_gates_the_flow_term():
    input, target = torch.randn(4, 2), torch.randn(4, 2)
    t = torch.tensor([0.0, 0.8, 0.9, 1.0])
    gate = torch.tensor([1.0, 1.0, 0.5, 0.0])
    assert torch.allclose(compute_flow_weight(t), gate)
    expected = (input - target).square().mean(dim=1) * gate
    assert torch.allclose(
        F.energy_matching_loss(input, target, t, reduction="none"), expected
    )
    ungated = F.energy_matching_loss(
        input, target, t, flow_weight_cutoff=1.0, reduction="none"
    )
    assert torch.equal(ungated, F.flow_matching_loss(input, target, reduction="none"))


def test_energy_matching_loss_matches_class_flow_term():
    torch.manual_seed(0)
    model, path = Quadratic(), LinearInterpolant()
    x0, x1, t = torch.randn(8, 2), torch.randn(8, 2), torch.rand(8)
    loss = EnergyMatchingLoss(
        model, coupling="independent", lambda_cd=0.0, sigma=0.0, t_sampler=fixed_t(t)
    )
    x_t, u_t = path.interpolate(x0, x1, t)
    expected = F.energy_matching_loss(-x_t, u_t, t)
    assert torch.allclose(loss.training_losses(x1, x0=x0)["flow_loss"], expected)


def test_score_matching_loss_formula():
    score, divergence = torch.tensor([[3.0, 4.0]]), torch.tensor([-2.0])
    assert F.score_matching_loss(score, divergence) == 0.5 * 25.0 - 2.0


def test_score_matching_loss_matches_exact_class():
    torch.manual_seed(0)
    model = Quadratic()
    x = torch.randn(8, 2)
    expected = F.score_matching_loss(-x, torch.full((8,), -2.0))
    assert torch.allclose(ScoreMatching(model)(x), expected)


def test_denoising_score_matching_loss_formula():
    input, target = torch.tensor([[1.0, 2.0]]), torch.zeros(1, 2)
    assert F.denoising_score_matching_loss(input, target) == 2.5
    weighted = F.denoising_score_matching_loss(
        input, target, weight=torch.tensor([4.0])
    )
    assert weighted == 10.0


def test_denoising_score_matching_loss_matches_class():
    torch.manual_seed(0)
    model, sigma = Quadratic(), 0.1
    x = torch.randn(8, 2)
    loss = DenoisingScoreMatching(model, noise_scale=sigma)
    x_perturbed, noise = loss.perturb_data(
        x, generator=torch.Generator().manual_seed(1)
    )
    expected = F.denoising_score_matching_loss(
        loss.compute_score(x_perturbed), noise / sigma**2
    )
    assert torch.equal(loss(x, generator=torch.Generator().manual_seed(1)), expected)


def test_sliced_score_matching_loss_formula():
    """For E = ||x||^2 / 2: v.s = -v.x and v.grad(v.s) = -||v||^2."""
    x = torch.randn(6, 2, requires_grad=True)
    projections = torch.randn(6, 2)
    loss = F.sliced_score_matching_loss(
        -x, x, projections, n_projections=3, reduction="none"
    )
    projected = (projections * x).sum(dim=1)
    per_row = -projections.square().sum(dim=1) + 0.5 * projected.square()
    assert torch.allclose(loss.detach(), per_row.view(3, 2).mean(dim=0).detach())


def test_sliced_score_matching_loss_matches_class():
    torch.manual_seed(0)
    model = Quadratic()
    x = torch.randn(8, 2)
    loss = SlicedScoreMatching(model, n_projections=3)
    tiled = x.unsqueeze(0).expand(3, *x.shape).contiguous().view(-1, 2)
    tiled.requires_grad_(True)
    projections = loss._get_random_projections(
        tiled, generator=torch.Generator().manual_seed(1)
    )
    score = torch.autograd.grad((-model(tiled)).sum(), tiled, create_graph=True)[0]
    expected = F.sliced_score_matching_loss(score, tiled, projections, n_projections=3)
    assert torch.equal(loss(x, generator=torch.Generator().manual_seed(1)), expected)


def test_invalid_reduction_raises():
    with pytest.raises(ValueError, match="reduction"):
        F.flow_matching_loss(torch.zeros(2, 2), torch.zeros(2, 2), reduction="avg")
