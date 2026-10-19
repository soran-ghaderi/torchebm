"""Legacy API against the target API on identical inputs and seeds.

Each step of the standard-API migration (.progress/STANDARD_API.md) adds its
pair here: the legacy call and the target call, one seeded generator for
both, compared with torch.equal wherever the operations are the same. A pair
stays until the deprecation ledger retires its legacy path and is deleted
with it.
"""

import pytest
import torch
from torch import nn
from torch.func import grad, jacrev, vmap

from torchebm.core import BaseModel, HarmonicModel
from torchebm.interpolants import (
    CosineInterpolant,
    LinearInterpolant,
    VarianceExplodingInterpolant,
    VariancePreservingInterpolant,
)
from torchebm.losses import (
    ContrastiveDivergence,
    ContrastiveDivergenceLoss,
    DenoisingScoreMatching,
    DenoisingScoreMatchingLoss,
    EnergyMatchingLoss,
    EquilibriumMatchingLoss,
    FlowMatchingLoss,
    ScoreMatching,
    ScoreMatchingLoss,
    SlicedScoreMatching,
    SlicedScoreMatchingLoss,
)
from torchebm.samplers import (
    FlowSampler,
    GradientDescentSampler,
    HamiltonianMonteCarlo,
    LangevinDynamics,
    NesterovSampler,
    RiemannianManifoldHMC,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::torchebm._deprecation.TorchEBMDeprecationWarning"
)

INTERPOLANTS = [
    LinearInterpolant,
    CosineInterpolant,
    VariancePreservingInterpolant,
    VarianceExplodingInterpolant,
]


class Quadratic(BaseModel):
    def forward(self, x):
        return 0.5 * x.square().sum(dim=-1)


class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x, t, **kwargs):
        return self.linear(x)


def seeded():
    return torch.Generator().manual_seed(1234)


def legacy_path_draws(x1, generator):
    """The draws of the legacy interpolant losses, in order: the source, then t."""
    x0 = torch.randn_like(x1, generator=generator)
    t = torch.rand(x1.shape[0], generator=generator)
    return x0, t


def test_score_prediction_matches_legacy_gradient():
    x = torch.randn(8, 3)
    legacy = -HarmonicModel(k=2.0).gradient(x)
    with torch.no_grad():
        target = HarmonicModel(k=2.0, prediction_type="score")(x)
    assert torch.equal(target, legacy)


@pytest.mark.parametrize("interpolant", INTERPOLANTS)
def test_interpolant_call_matches_legacy_interpolate(interpolant):
    torch.manual_seed(0)
    x0, x1, t = torch.randn(8, 3), torch.randn(8, 3), torch.rand(8)
    legacy = interpolant().interpolate(x0, x1, t)
    target = interpolant(prediction_type="velocity")(x0, x1, t)
    assert all(torch.equal(a, b) for a, b in zip(target, legacy))
    assert interpolant(prediction_type="epsilon")(x0, x1, t)[1] is x0
    assert interpolant(prediction_type="sample")(x0, x1, t)[1] is x1


def test_contrastive_divergence_matches_legacy():
    torch.manual_seed(0)
    model, x = Quadratic(), torch.randn(8, 2)
    sampler = LangevinDynamics(model, step_size=0.1)
    legacy = ContrastiveDivergence(model, sampler, k_steps=3, energy_reg_weight=1e-3)
    legacy_loss, legacy_negatives = legacy(x, generator=seeded())

    x_neg = sampler.sample(x=x.clone(), n_steps=3, generator=seeded())
    loss = ContrastiveDivergenceLoss(energy_reg_weight=1e-3)(model(x), model(x_neg))
    assert torch.equal(x_neg, legacy_negatives)
    assert torch.equal(loss, legacy_loss)


def test_flow_matching_matches_legacy():
    torch.manual_seed(0)
    field, x = Field(), torch.randn(8, 2)
    legacy = FlowMatchingLoss(model=field)(x, generator=seeded())

    x0, t = legacy_path_draws(x, seeded())
    x_t, u_t = LinearInterpolant()(x0, x, t)
    assert torch.equal(FlowMatchingLoss()(field(x_t, t), u_t), legacy)


def test_equilibrium_matching_matches_legacy_up_to_the_field_sign():
    """Legacy regresses onto -c(t) u_t (data to noise); the criterion onto +c(t) u_t."""
    torch.manual_seed(0)
    field, x = Field(), torch.randn(8, 2)
    legacy = EquilibriumMatchingLoss(model=field, energy_type="none")(
        x, generator=seeded()
    )

    x0, t = legacy_path_draws(x, seeded())
    x_t, u_t = LinearInterpolant()(x0, x, t)
    score = -field(x_t, torch.zeros_like(t))
    assert torch.equal(EquilibriumMatchingLoss()(score, u_t, t), legacy)


def test_energy_matching_flow_term_matches_legacy():
    torch.manual_seed(0)
    x = torch.randn(8, 2)
    legacy = EnergyMatchingLoss(
        model=Quadratic(), coupling="independent", lambda_cd=0.0, sigma=0.0
    )(x, generator=seeded())

    x0, t = legacy_path_draws(x, seeded())
    x_t, u_t = LinearInterpolant()(x0, x, t)
    score = Quadratic(prediction_type="score")(x_t)
    assert torch.equal(EnergyMatchingLoss()(score, u_t, t), legacy)


def test_score_matching_matches_legacy():
    torch.manual_seed(0)
    model, x = Quadratic(), torch.randn(8, 2)
    legacy = ScoreMatching(model)(x)

    score_fn = grad(lambda x_i: -model(x_i.unsqueeze(0)).squeeze())
    x_flat = x.view(8, -1).detach().requires_grad_(True)
    score = vmap(score_fn)(x_flat)
    divergence = vmap(lambda x_i: jacrev(score_fn)(x_i).diagonal().sum())(x_flat)
    assert torch.equal(ScoreMatchingLoss()(score, divergence), legacy)


def test_denoising_score_matching_matches_legacy():
    torch.manual_seed(0)
    x, sigma = torch.randn(8, 2), 0.1
    legacy = DenoisingScoreMatching(Quadratic(), noise_scale=sigma)(
        x, generator=seeded()
    )

    noise = torch.randn_like(x, generator=seeded()) * sigma
    score = Quadratic(prediction_type="score")(x + noise)
    assert torch.equal(DenoisingScoreMatchingLoss()(score, -noise / sigma**2), legacy)


def test_sliced_score_matching_matches_legacy():
    torch.manual_seed(0)
    x = torch.randn(8, 2)
    legacy = SlicedScoreMatching(Quadratic(), n_projections=3)(x, generator=seeded())

    x_tiled = x.repeat(3, 1).requires_grad_(True)
    score = Quadratic(prediction_type="score")(x_tiled)
    criterion = SlicedScoreMatchingLoss(n_projections=3)
    assert torch.equal(criterion(score, x_tiled, generator=seeded()), legacy)


class TimeField(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x, t, **kwargs):
        return self.linear(x) * (1.0 + t.view(-1, 1))


FLOW_PREDICTIONS = [("velocity", "velocity"), ("score", "score"), ("noise", "epsilon")]


def identity_metric(x):
    eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    return eye.expand(x.shape[0], *eye.shape).contiguous()


@pytest.mark.parametrize(
    "integrator", ["euler_maruyama", "heun", "backward_euler_maruyama"]
)
@pytest.mark.parametrize("trajectory", [False, True])
def test_langevin_call_matches_legacy_sample(integrator, trajectory):
    x = torch.randn(8, 2, generator=seeded())
    kwargs = dict(
        step_size=0.1, noise_scale=0.5, clamp=(-3.0, 3.0), integrator=integrator
    )
    legacy = LangevinDynamics(Quadratic(), **kwargs).sample(
        x=x, n_steps=5, generator=seeded(), return_trajectory=trajectory
    )
    for model in (Quadratic(), Quadratic(prediction_type="score")):
        target = LangevinDynamics(**kwargs)(
            model,
            x,
            n_steps=5,
            generator=seeded(),
            return_trajectory=trajectory,
        )
        assert torch.equal(target, legacy)


@pytest.mark.parametrize(
    "sampler_cls, kwargs",
    [(GradientDescentSampler, {}), (NesterovSampler, {"momentum": 0.9})],
)
@pytest.mark.parametrize("trajectory", [False, True])
def test_optimization_call_matches_legacy_sample(sampler_cls, kwargs, trajectory):
    x = torch.randn(8, 2, generator=seeded())
    legacy = sampler_cls(Quadratic(), step_size=0.1, **kwargs).sample(
        x=x, n_steps=7, return_trajectory=trajectory
    )
    target = sampler_cls(step_size=0.1, **kwargs)(
        Quadratic(), x, n_steps=7, return_trajectory=trajectory
    )
    assert torch.equal(target, legacy)


@pytest.mark.parametrize(
    "sampler_cls, kwargs",
    [
        (HamiltonianMonteCarlo, {"n_leapfrog_steps": 3}),
        (RiemannianManifoldHMC, {"n_leapfrog_steps": 2, "metric_fn": identity_metric}),
    ],
)
@pytest.mark.parametrize("trajectory", [False, True])
def test_hmc_call_matches_legacy_sample(sampler_cls, kwargs, trajectory):
    x = torch.randn(8, 2, generator=seeded())
    legacy = sampler_cls(Quadratic(), step_size=0.1, **kwargs).sample(
        x=x, n_steps=5, generator=seeded(), return_trajectory=trajectory
    )
    for model in (Quadratic(), Quadratic(prediction_type=("energy", "score"))):
        target = sampler_cls(step_size=0.1, **kwargs)(
            model,
            x,
            n_steps=5,
            generator=seeded(),
            return_trajectory=trajectory,
        )
        assert torch.equal(target, legacy)


@pytest.mark.parametrize(
    "integrator", ["euler", "midpoint", "rk4", "backward_euler_maruyama"]
)
@pytest.mark.parametrize("prediction, prediction_type", FLOW_PREDICTIONS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("interpolant", ["linear", "vp"])
def test_flow_ode_call_matches_legacy_sample(
    integrator, prediction, prediction_type, reverse, interpolant
):
    torch.manual_seed(0)
    model = TimeField()
    x = torch.randn(6, 2, generator=seeded())
    kwargs = dict(
        interpolant=interpolant, sample_eps=0.01, reverse=reverse, integrator=integrator
    )
    legacy = FlowSampler(model, prediction=prediction, **kwargs).sample(
        x=x, n_steps=4, return_trajectory=True
    )
    target = FlowSampler(prediction_type=prediction_type, **kwargs)(
        model, x, n_steps=4, return_trajectory=True
    )
    assert torch.equal(target, legacy)


@pytest.mark.parametrize("prediction, prediction_type", FLOW_PREDICTIONS)
@pytest.mark.parametrize("reverse", [False, True])
def test_flow_adaptive_call_matches_legacy_sample(prediction, prediction_type, reverse):
    torch.manual_seed(0)
    model = TimeField()
    x = torch.randn(6, 2, generator=seeded())
    kwargs = dict(sample_eps=0.01, reverse=reverse, integrator="dopri5")
    legacy = FlowSampler(model, prediction=prediction, **kwargs).sample(x=x, n_steps=4)
    target = FlowSampler(prediction_type=prediction_type, **kwargs)(model, x, n_steps=4)
    assert torch.equal(target, legacy)


@pytest.mark.parametrize("integrator", ["euler_maruyama", "heun"])
@pytest.mark.parametrize("prediction, prediction_type", FLOW_PREDICTIONS)
@pytest.mark.parametrize("last_step", ["Mean", "Euler", "Tweedie", None])
@pytest.mark.parametrize("diffusion_form", ["SBDM", "sigma"])
def test_flow_sde_call_matches_legacy_sample(
    integrator, prediction, prediction_type, last_step, diffusion_form
):
    torch.manual_seed(0)
    model = TimeField()
    x = torch.randn(6, 2, generator=seeded())
    kwargs = dict(
        mode="sde",
        sample_eps=0.05,
        diffusion_form=diffusion_form,
        last_step=last_step,
        integrator=integrator,
    )
    legacy = FlowSampler(model, prediction=prediction, **kwargs).sample(
        x=x, n_steps=4, generator=seeded(), return_trajectory=True
    )
    target = FlowSampler(prediction_type=prediction_type, **kwargs)(
        model, x, n_steps=4, generator=seeded(), return_trajectory=True
    )
    assert torch.equal(target, legacy)


@pytest.mark.parametrize("reverse", [False, True])
def test_flow_custom_grid_matches_legacy_uniform_grid(reverse):
    torch.manual_seed(0)
    model = TimeField()
    x = torch.randn(6, 2, generator=seeded())
    kwargs = dict(reverse=reverse, integrator="midpoint")
    legacy = FlowSampler(model, prediction="velocity", **kwargs).sample(
        x=x, n_steps=4, return_trajectory=True
    )
    target = FlowSampler(**kwargs)(
        model, x, t=torch.linspace(0.0, 1.0, 5), return_trajectory=True
    )
    assert torch.equal(target, legacy)
