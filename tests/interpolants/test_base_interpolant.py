"""Tests for BaseInterpolant and interpolant utilities."""

import pytest
import torch

from torchebm.core.base_interpolant import BaseInterpolant, expand_t_like_x
from torchebm.interpolants import LinearInterpolant, CosineInterpolant, VariancePreservingInterpolant
from tests.conftest import requires_cuda


################################ expand_t_like_x Tests ######################################


def test_expand_t_like_x_1d():
    """Test expansion for 1D spatial."""
    t = torch.tensor([0.1, 0.2, 0.3, 0.4])  # (batch_size,)
    x = torch.randn(4, 8)  # (batch_size, features)
    
    t_expanded = expand_t_like_x(t, x)
    
    assert t_expanded.shape == (4, 1)
    assert torch.allclose(t_expanded.squeeze(-1), t)


def test_expand_t_like_x_2d():
    """Test expansion for 2D spatial (images)."""
    t = torch.tensor([0.1, 0.2, 0.3, 0.4])  # (batch_size,)
    x = torch.randn(4, 3, 16, 16)  # (batch_size, channels, H, W)
    
    t_expanded = expand_t_like_x(t, x)
    
    assert t_expanded.shape == (4, 1, 1, 1)
    # Can broadcast with x
    result = t_expanded * x
    assert result.shape == x.shape


def test_expand_t_like_x_3d():
    """Test expansion for 3D spatial (video/volumetric)."""
    t = torch.tensor([0.1, 0.2])
    x = torch.randn(2, 3, 8, 16, 16)  # (batch, channels, D, H, W)
    
    t_expanded = expand_t_like_x(t, x)
    
    assert t_expanded.shape == (2, 1, 1, 1, 1)
    result = t_expanded * x
    assert result.shape == x.shape


def test_expand_t_preserves_values():
    """Verify expansion preserves original time values."""
    t = torch.tensor([0.25, 0.5, 0.75])
    x = torch.randn(3, 4, 8, 8)
    
    t_expanded = expand_t_like_x(t, x)
    
    # Expanded values match originals
    assert torch.allclose(t_expanded[:, 0, 0, 0], t)


def test_expand_t_device_dtype():
    """Test device and dtype preservation."""
    t = torch.tensor([0.1, 0.2], dtype=torch.float64)
    x = torch.randn(2, 3, dtype=torch.float64)
    
    t_expanded = expand_t_like_x(t, x)
    
    assert t_expanded.dtype == t.dtype


@requires_cuda
def test_expand_t_cuda():
    """Test expansion on CUDA."""
    t = torch.tensor([0.1, 0.2], device="cuda")
    x = torch.randn(2, 3, 8, 8, device="cuda")
    
    t_expanded = expand_t_like_x(t, x)
    
    assert t_expanded.device.type == "cuda"
    assert t_expanded.shape == (2, 1, 1, 1)


################################ BaseInterpolant Interface Tests ############################


class ConcreteInterpolant(BaseInterpolant):
    """Minimal concrete implementation for testing abstract base class."""
    
    def compute_alpha_t(self, t):
        return t, torch.ones_like(t)
    
    def compute_sigma_t(self, t):
        return 1 - t, -torch.ones_like(t)


def test_base_interpolant_abstract():
    """Test that BaseInterpolant cannot be instantiated directly."""
    # BaseInterpolant is abstract, but we can test the interface through a concrete impl
    interpolant = ConcreteInterpolant()
    assert isinstance(interpolant, BaseInterpolant)


def test_default_d_alpha_alpha_ratio():
    """Test default implementation of d_alpha_alpha_ratio."""
    interpolant = ConcreteInterpolant()
    t = torch.tensor([0.2, 0.4, 0.6, 0.8])
    
    ratio = interpolant.compute_d_alpha_alpha_ratio_t(t)
    
    # Default: d_alpha / alpha = 1 / t (for this concrete implementation)
    expected = 1.0 / torch.clamp(t, min=1e-8)
    assert torch.allclose(ratio, expected, atol=1e-6)


################################ Cross-Interpolant Consistency Tests ########################


@pytest.fixture(params=[
    LinearInterpolant,
    CosineInterpolant,
    VariancePreservingInterpolant,
])
def any_interpolant(request):
    """Fixture providing all interpolant types."""
    return request.param()


def test_all_interpolants_boundary_t0(any_interpolant):
    """All interpolants should satisfy: at t=0, x_t ≈ x0."""
    torch.manual_seed(42)
    x0 = torch.randn(16, 4)
    x1 = torch.randn(16, 4)
    t = torch.zeros(16)
    
    xt, _ = any_interpolant.interpolate(x0, x1, t)
    
    # At t=0, calculate exact expected value using coefficients
    alpha, _ = any_interpolant.compute_alpha_t(t[0:1])
    sigma, _ = any_interpolant.compute_sigma_t(t[0:1])
    expected = alpha * x1 + sigma * x0
    
    # Verify consistency between interpolate() and compute_*_t() coefficients
    assert torch.allclose(xt, expected, atol=1e-5)


def test_all_interpolants_boundary_t1(any_interpolant):
    """All interpolants should satisfy: at t=1, x_t ≈ x1."""
    torch.manual_seed(42)
    x0 = torch.randn(16, 4)
    x1 = torch.randn(16, 4)
    t = torch.ones(16)
    
    xt, _ = any_interpolant.interpolate(x0, x1, t)
    
    # At t=1, calculate exact expected value using coefficients
    alpha, _ = any_interpolant.compute_alpha_t(t[0:1])
    sigma, _ = any_interpolant.compute_sigma_t(t[0:1])
    expected = alpha * x1 + sigma * x0
    
    # Verify consistency
    assert torch.allclose(xt, expected, atol=1e-5)


def test_all_interpolants_compute_drift(any_interpolant):
    """All interpolants should implement compute_drift."""
    torch.manual_seed(42)
    x = torch.randn(16, 4)
    t = torch.rand(16) * 0.8 + 0.1
    
    drift_mean, drift_var = any_interpolant.compute_drift(x, t)
    
    assert drift_mean.shape == x.shape
    # drift_var is broadcast-compatible (batch_size, 1, ...)
    assert drift_var.shape[0] == x.shape[0]
    _ = drift_var * x
    assert torch.all(torch.isfinite(drift_mean))
    assert torch.all(torch.isfinite(drift_var))


def test_all_interpolants_compute_diffusion(any_interpolant):
    """All interpolants should implement compute_diffusion."""
    torch.manual_seed(42)
    x = torch.randn(16, 4)
    t = torch.rand(16) * 0.8 + 0.1
    
    for form in ["constant", "SBDM", "sigma", "linear"]:
        diffusion = any_interpolant.compute_diffusion(x, t, form=form)
        # diffusion is broadcast-compatible (batch_size, 1, ...)
        assert diffusion.shape[0] == x.shape[0]
        _ = diffusion * x
        assert torch.all(torch.isfinite(diffusion))


def test_all_interpolants_velocity_score_roundtrip(any_interpolant):
    """All interpolants should satisfy velocity <-> score roundtrip."""
    torch.manual_seed(42)
    x = torch.randn(16, 4)
    t = torch.rand(16) * 0.8 + 0.1
    
    velocity = torch.randn_like(x)
    
    score = any_interpolant.velocity_to_score(velocity, x, t)
    velocity_recovered = any_interpolant.score_to_velocity(score, x, t)
    
    assert torch.allclose(velocity, velocity_recovered, atol=1e-4)


def test_all_interpolants_noise_recovery(any_interpolant):
    """All interpolants should allow recovery of original noise from exact velocity."""
    torch.manual_seed(42)
    x0 = torch.randn(16, 4)  # noise
    x1 = torch.randn(16, 4)  # data
    t = torch.rand(16) * 0.8 + 0.1
    
    xt, ut = any_interpolant.interpolate(x0, x1, t)
    
    noise_pred = any_interpolant.velocity_to_noise(ut, xt, t)
    
    assert torch.allclose(noise_pred, x0, atol=1e-3)


################################ Shape Consistency Tests ###################################


@pytest.mark.parametrize("shape", [
    (8, 4),
    (16, 3, 8, 8),
    (4, 1, 16, 16, 16),
])
def test_all_interpolants_multi_dim_shapes(any_interpolant, shape):
    """All interpolants should handle various input shapes."""
    torch.manual_seed(42)
    x0 = torch.randn(*shape)
    x1 = torch.randn(*shape)
    t = torch.rand(shape[0]) * 0.98 + 0.01
    
    xt, ut = any_interpolant.interpolate(x0, x1, t)
    
    assert xt.shape == shape
    assert ut.shape == shape


################################ Gradient Flow Tests #######################################


def test_interpolate_gradient_flow(any_interpolant):
    """Verify gradients flow through interpolation."""
    torch.manual_seed(42)
    x0 = torch.randn(8, 4, requires_grad=True)
    x1 = torch.randn(8, 4, requires_grad=True)
    t = torch.rand(8)
    
    xt, ut = any_interpolant.interpolate(x0, x1, t)
    
    # Gradient w.r.t. x0
    loss = xt.sum()
    loss.backward()
    
    assert x0.grad is not None
    assert x1.grad is not None
    assert torch.any(x0.grad != 0)
    assert torch.any(x1.grad != 0)


def test_velocity_to_score_gradient_flow(any_interpolant):
    """Verify gradients flow through velocity_to_score."""
    torch.manual_seed(42)
    velocity = torch.randn(8, 4, requires_grad=True)
    x = torch.randn(8, 4)
    t = torch.rand(8) * 0.8 + 0.1
    
    score = any_interpolant.velocity_to_score(velocity, x, t)
    loss = score.sum()
    loss.backward()
    
    assert velocity.grad is not None


def test_score_to_velocity_gradient_flow(any_interpolant):
    """Verify gradients flow through score_to_velocity."""
    torch.manual_seed(42)
    score = torch.randn(8, 4, requires_grad=True)
    x = torch.randn(8, 4)
    t = torch.rand(8) * 0.8 + 0.1
    
    velocity = any_interpolant.score_to_velocity(score, x, t)
    loss = velocity.sum()
    loss.backward()
    
    assert score.grad is not None


################################ Numerical Precision Comparison #############################


def test_float32_vs_float64_precision():
    """Compare precision between float32 and float64."""
    interpolant = LinearInterpolant()
    torch.manual_seed(42)
    
    x0_32 = torch.randn(16, 4, dtype=torch.float32)
    x1_32 = torch.randn(16, 4, dtype=torch.float32)
    t_32 = torch.rand(16, dtype=torch.float32)
    
    x0_64 = x0_32.double()
    x1_64 = x1_32.double()
    t_64 = t_32.double()
    
    xt_32, ut_32 = interpolant.interpolate(x0_32, x1_32, t_32)
    xt_64, ut_64 = interpolant.interpolate(x0_64, x1_64, t_64)
    
    # Results should be close (within float32 precision)
    assert torch.allclose(xt_32.double(), xt_64, atol=1e-6)
    assert torch.allclose(ut_32.double(), ut_64, atol=1e-6)
