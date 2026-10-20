import pytest
import torch
from torch import nn

from torchebm._deprecation import TorchEBMDeprecationWarning
from torchebm.core import BaseModel, TemperatureScheduler
from torchebm.models.wrappers import (
    ClassifierFreeGuidance,
    EqMEnergy,
    InteractionModel,
)
from torchebm.samplers import FlowSampler, LangevinDynamics


class _DummyBase(nn.Module):
    def __init__(self, null_id: int, out_channels: int = 4):
        super().__init__()
        self.null_id = null_id
        self.out_channels = out_channels

    def forward(self, x, t, y=None, **kwargs):
        b, _, h, w = x.shape
        base = torch.zeros(b, self.out_channels, h, w)
        cond_marker = (y != self.null_id).float().view(b, 1, 1, 1)
        return base + cond_marker


def test_cfg_scale_one_matches_base_forward():
    base = _DummyBase(null_id=10)
    wrapped = ClassifierFreeGuidance(base, guidance_scale=1.0, null_condition=10)
    x = torch.randn(2, 3, 4, 4)
    t = torch.rand(2)
    y = torch.tensor([1, 2])
    out = wrapped(x, t, y=y)
    assert torch.allclose(out, base(x, t, y=y))


def test_cfg_applies_guidance_to_first_channels_only():
    base = _DummyBase(null_id=10, out_channels=4)
    wrapped = ClassifierFreeGuidance(
        base, guidance_scale=2.0, null_condition=10, guide_channels=3
    )
    x = torch.randn(1, 3, 4, 4)
    t = torch.rand(1)
    y = torch.tensor([1])
    out = wrapped(x, t, y=y)
    assert out.shape == (1, 4, 4, 4)
    # First 3 channels: uncond(0) + 2.0 * (cond(1) - uncond(0)) = 2.0
    assert torch.allclose(out[:, :3], torch.full_like(out[:, :3], 2.0))
    # Last channel: uncond == 0
    assert torch.allclose(out[:, 3:], torch.zeros_like(out[:, 3:]))


def test_cfg_guide_channels_exceeds_output_clamps():
    base = _DummyBase(null_id=5, out_channels=2)
    wrapped = ClassifierFreeGuidance(
        base, guidance_scale=3.0, null_condition=5, guide_channels=10
    )
    x = torch.randn(1, 3, 2, 2)
    t = torch.rand(1)
    y = torch.tensor([0])
    out = wrapped(x, t, y=y)
    assert out.shape == (1, 2, 2, 2)
    assert torch.allclose(out, torch.full_like(out, 3.0))


# InteractionModel (Energy Matching pairwise repulsion)
# =====================================================


class _Quadratic(BaseModel):
    def forward(self, x):
        return 0.5 * x.flatten(1).square().sum(dim=1)


def test_interaction_model_energy_two_points():
    """E_i = V(x_i) - 0.5 * (s / sigma_w^2) * sum_j ||x_i - x_j||^2."""
    model = InteractionModel(_Quadratic(), sigma_w=1.0, strength=1.0)
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    # V = [0, 1]; ||x_0 - x_1||^2 = 2 -> W = [1, 1]; E = V - W = [-1, 0]
    energy = model(x)
    assert torch.allclose(energy, torch.tensor([-1.0, 0.0]), atol=1e-6)


def test_interaction_model_zero_strength_matches_base():
    base = _Quadratic()
    model = InteractionModel(base, sigma_w=1.0, strength=0.0)
    x = torch.randn(8, 2)
    assert torch.allclose(model(x), base(x), atol=1e-6)


def test_interaction_model_gradient_analytic():
    """Batch-coupled drift: grad E_i = x_i - 2 * (s / sigma_w^2) * (x_i - mean-term).

    For B = 2, s = sigma_w = 1: d(sum E)/dx_0 = x_0 - 2 * (x_0 - x_1).
    """
    model = InteractionModel(_Quadratic(), sigma_w=1.0, strength=1.0)
    x = torch.tensor([[0.5, -0.5], [1.5, 2.0]])
    grad = model.gradient(x)
    x0, x1 = x[0], x[1]
    expected = torch.stack([x0 - 2 * (x0 - x1), x1 - 2 * (x1 - x0)])
    assert torch.allclose(grad, expected, atol=1e-5)


def test_interaction_model_scheduler_advances_with_sampler():
    """The strength scheduler lives in the sampler's subtree and steps with it."""
    strength = TemperatureScheduler(
        epsilon_max=0.1, tau_star=0.5, n_steps=10, sqrt=False
    )
    model = InteractionModel(_Quadratic(), sigma_w=1.0, strength=strength)
    sampler = LangevinDynamics(model=model, step_size=0.01, noise_scale=1.0)
    sampler.sample(x=torch.randn(4, 2), n_steps=7)
    assert strength.step_count == 7


def test_interaction_model_repulsion_increases_spread():
    """Repulsive sampling spreads a clustered batch further apart."""

    def spread(strength):
        torch.manual_seed(0)
        x_init = torch.randn(16, 2) * 0.01
        model = InteractionModel(_Quadratic(), sigma_w=1.0, strength=strength)
        sampler = LangevinDynamics(model=model, step_size=0.01, noise_scale=0.01)
        samples = sampler.sample(x=x_init, n_steps=50)
        return torch.cdist(samples, samples).mean()

    assert spread(strength=0.4) > spread(strength=0.0) * 2


def test_interaction_model_invalid_sigma_w():
    with pytest.raises(ValueError, match="sigma_w"):
        InteractionModel(_Quadratic(), sigma_w=0.0)


# The sampler call: sampler(model, x, ...)
# ========================================


class _ConditionalQuadratic(BaseModel):
    def forward(self, x, y=None):
        energy = 0.5 * x.flatten(1).square().sum(dim=1)
        return energy if y is None else energy + y


class _Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x, t, y=None, **kwargs):
        out = self.linear(x)
        return out if y is None else out + y.view(-1, 1)


class _VelocityField(_Field):
    prediction_type = "velocity"


def _seeded():
    return torch.Generator().manual_seed(3)


@pytest.mark.parametrize(
    "model, expected",
    [
        (_Field(), None),
        (_VelocityField(), "velocity"),
        (_ConditionalQuadratic(), "energy"),
        (_ConditionalQuadratic(prediction_type="score"), "score"),
        (_ConditionalQuadratic(prediction_type=("energy", "score")), "energy"),
    ],
)
def test_cfg_reports_the_wrapped_prediction_type(model, expected):
    guided = ClassifierFreeGuidance(model, guidance_scale=2.0, null_condition=0.0)
    assert guided.prediction_type == expected


def test_cfg_call_time_flow_sampling_matches_the_null_labels():
    torch.manual_seed(0)
    field = _Field()
    guided = ClassifierFreeGuidance(field, guidance_scale=0.0, null_condition=0.0)
    x = torch.randn(4, 2, generator=_seeded())
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    sampler = FlowSampler(integrator="euler")
    out = sampler(guided, x, n_steps=5, model_kwargs={"y": y})
    plain = sampler(field, x, n_steps=5, model_kwargs={"y": torch.zeros(4)})
    assert torch.equal(out, plain)


def test_cfg_call_time_langevin_sampling_matches_the_null_labels():
    energy = _ConditionalQuadratic()
    guided = ClassifierFreeGuidance(energy, guidance_scale=0.0, null_condition=0.0)
    x = torch.randn(4, 2, generator=_seeded())
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    sampler = LangevinDynamics(step_size=0.01)
    out = sampler(guided, x, n_steps=10, model_kwargs={"y": y}, generator=_seeded())
    plain = sampler(
        energy, x, n_steps=10, model_kwargs={"y": torch.zeros(4)}, generator=_seeded()
    )
    assert torch.equal(out, plain)


def test_cfg_over_a_score_model_matches_the_guided_energy():
    kwargs = dict(guidance_scale=2.5, null_condition=0.0)
    energy = ClassifierFreeGuidance(_ConditionalQuadratic(), **kwargs)
    score = ClassifierFreeGuidance(
        _ConditionalQuadratic(prediction_type="score"), **kwargs
    )
    x = torch.randn(4, 2, generator=_seeded())
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    sampler = LangevinDynamics(step_size=0.01)
    from_energy = sampler(
        energy, x, n_steps=10, model_kwargs={"y": y}, generator=_seeded()
    )
    from_score = sampler(
        score, x, n_steps=10, model_kwargs={"y": y}, generator=_seeded()
    )
    assert torch.allclose(from_energy, from_score, atol=1e-6)


def test_interaction_model_rejects_a_score_potential():
    with pytest.raises(ValueError, match="prediction_type"):
        InteractionModel(_Quadratic(prediction_type="score"), sigma_w=1.0)


def test_interaction_model_score_output_is_the_negative_gradient():
    model = InteractionModel(
        _Quadratic(), sigma_w=1.0, strength=1.0, prediction_type="score"
    )
    x = torch.tensor([[0.5, -0.5], [1.5, 2.0]])
    assert torch.equal(model(x), -model.gradient(x))


def test_interaction_model_schedule_advances_with_the_call():
    strength = TemperatureScheduler(
        epsilon_max=0.1, tau_star=0.5, n_steps=10, sqrt=False
    )
    model = InteractionModel(_Quadratic(), sigma_w=1.0, strength=strength)
    sampler = LangevinDynamics(step_size=0.01, noise_scale=1.0)
    sampler(model, torch.randn(4, 2), n_steps=7)
    assert strength.step_count == 7
    sampler(model, torch.randn(4, 2), n_steps=3)
    assert strength.step_count == 3


def test_eqm_energy_is_deprecated():
    with pytest.warns(TorchEBMDeprecationWarning, match=r"sampler\(model"):
        EqMEnergy(_Field(), energy_type="implicit")
