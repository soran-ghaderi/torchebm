"""Tests for VarianceExplodingInterpolant (NCSN / VE-SDE schedule)."""

import math

import pytest
import torch

from torchebm.interpolants import VarianceExplodingInterpolant, get_interpolant


def test_default_initialization():
    interpolant = VarianceExplodingInterpolant()
    assert interpolant.sigma_min == 0.01
    assert interpolant.sigma_max == 50.0
    assert interpolant.prediction_type == "velocity"


@pytest.mark.parametrize(
    "sigma_min,sigma_max", [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (-1.0, 1.0)]
)
def test_rejects_invalid_noise_levels(sigma_min, sigma_max):
    with pytest.raises(ValueError, match="sigma_min"):
        VarianceExplodingInterpolant(sigma_min=sigma_min, sigma_max=sigma_max)


def test_registry_name():
    assert isinstance(get_interpolant("ve"), VarianceExplodingInterpolant)


def test_data_is_never_scaled():
    alpha, d_alpha = VarianceExplodingInterpolant().compute_alpha_t(torch.rand(8))
    assert torch.equal(alpha, torch.ones(8))
    assert torch.equal(d_alpha, torch.zeros(8))


def test_sigma_is_geometric_from_sigma_max_to_sigma_min():
    interpolant = VarianceExplodingInterpolant(sigma_min=0.01, sigma_max=50.0)
    sigma, _ = interpolant.compute_sigma_t(torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(sigma, torch.tensor([50.0, math.sqrt(0.01 * 50.0), 0.01]))


def test_sigma_matches_reference_schedule():
    """score_sde ``VESDE.marginal_prob``: std = sigma_min (sigma_max / sigma_min)^s
    with s = 1 - t, since the reference puts data at s = 0."""
    interpolant = VarianceExplodingInterpolant(sigma_min=0.01, sigma_max=50.0)
    t = torch.linspace(0, 1, 11, dtype=torch.float64)
    sigma, _ = interpolant.compute_sigma_t(t)
    assert torch.allclose(sigma, 0.01 * (50.0 / 0.01) ** (1 - t))


def test_sigma_derivative_matches_autograd():
    interpolant = VarianceExplodingInterpolant()
    t = torch.rand(16, dtype=torch.float64)
    _, d_sigma = interpolant.compute_sigma_t(t)
    _, expected = torch.func.jvp(
        lambda t: interpolant.compute_sigma_t(t)[0], (t,), (torch.ones_like(t),)
    )
    assert torch.allclose(d_sigma, expected)


def test_interpolate_is_the_perturbation_kernel():
    interpolant = VarianceExplodingInterpolant()
    x0, x1, t = torch.randn(16, 4), torch.randn(16, 4), torch.rand(16)
    sigma, d_sigma = interpolant.compute_sigma_t(t.view(16, 1))
    x_t, u_t = interpolant.interpolate(x0, x1, t)
    assert torch.allclose(x_t, x1 + sigma * x0)
    assert torch.allclose(u_t, d_sigma * x0)


def test_velocity_to_score_is_the_kernel_score():
    """The score of N(x_1, sigma^2 I) at x_t is -(x_t - x_1) / sigma^2."""
    interpolant = VarianceExplodingInterpolant()
    x0, x1, t = torch.randn(16, 4), torch.randn(16, 4), torch.rand(16)
    sigma, _ = interpolant.compute_sigma_t(t.view(16, 1))
    x_t, u_t = interpolant.interpolate(x0, x1, t)
    score = interpolant.velocity_to_score(u_t, x_t, t)
    assert torch.allclose(score, -(x_t - x1) / sigma.square(), atol=1e-5)


def test_velocity_score_round_trip():
    interpolant = VarianceExplodingInterpolant()
    x, t, score = torch.randn(16, 4), torch.rand(16), torch.randn(16, 4)
    velocity = interpolant.score_to_velocity(score, x, t)
    assert torch.allclose(interpolant.velocity_to_score(velocity, x, t), score, atol=1e-5)


def test_velocity_to_noise_recovers_x0():
    interpolant = VarianceExplodingInterpolant()
    x0, x1, t = torch.randn(16, 4), torch.randn(16, 4), torch.rand(16)
    x_t, u_t = interpolant.interpolate(x0, x1, t)
    assert torch.allclose(interpolant.velocity_to_noise(u_t, x_t, t), x0, atol=1e-5)


def test_drift_matches_ve_sde():
    """Zero drift and drift_var = g(t)^2 / 2 with g(t)^2 = 2 sigma(t)^2 log(sigma_max / sigma_min),
    the same half-g^2 convention VariancePreservingInterpolant uses (beta / 2)."""
    interpolant = VarianceExplodingInterpolant(sigma_min=0.01, sigma_max=50.0)
    x, t = torch.randn(16, 4), torch.rand(16)
    drift_mean, drift_var = interpolant.compute_drift(x, t)
    sigma, _ = interpolant.compute_sigma_t(t.view(16, 1))
    assert torch.allclose(drift_mean, torch.zeros_like(x))
    assert torch.allclose(drift_var, sigma.square() * math.log(50.0 / 0.01))


def test_epsilon_target_is_the_denoising_score_matching_noise():
    interpolant = VarianceExplodingInterpolant(prediction_type="epsilon")
    x0, x1, t = torch.randn(16, 4), torch.randn(16, 4), torch.rand(16)
    x_t, target = interpolant(x0, x1, t)
    assert target is x0
    assert torch.allclose(x_t, interpolant.interpolate(x0, x1, t)[0])
