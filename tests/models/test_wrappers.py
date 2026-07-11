import pytest
import torch
from torch import nn

from torchebm.core import BaseModel, TemperatureScheduler
from torchebm.models.wrappers import InteractionModel, LabelClassifierFreeGuidance
from torchebm.samplers import LangevinDynamics


class _DummyBase(nn.Module):
    def __init__(self, null_id: int, out_channels: int = 4):
        super().__init__()
        self.null_id = null_id
        self.out_channels = out_channels

    def forward(self, x, t, *, y):
        b, _, h, w = x.shape
        base = torch.zeros(b, self.out_channels, h, w)
        cond_marker = (y != self.null_id).float().view(b, 1, 1, 1)
        return base + cond_marker


def test_cfg_scale_one_returns_base_forward():
    base = _DummyBase(null_id=10)
    wrapped = LabelClassifierFreeGuidance(
        base, null_label_id=10, cfg_scale=1.0, guide_channels=3
    )
    x = torch.randn(2, 3, 4, 4)
    t = torch.rand(2)
    y = torch.tensor([1, 2])
    out = wrapped(x, t, y=y)
    assert torch.allclose(out, base(x, t, y=y))


def test_cfg_applies_guidance_to_first_channels_only():
    base = _DummyBase(null_id=10, out_channels=4)
    wrapped = LabelClassifierFreeGuidance(
        base, null_label_id=10, cfg_scale=2.0, guide_channels=3
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
    wrapped = LabelClassifierFreeGuidance(
        base, null_label_id=5, cfg_scale=3.0, guide_channels=10
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
