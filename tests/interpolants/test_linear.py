"""Tests for LinearInterpolant (optimal transport interpolant)."""

import math

import pytest
import torch

from torchebm.interpolants import LinearInterpolant
from tests.conftest import requires_cuda


@pytest.fixture
def interpolant():
    """Create a LinearInterpolant instance."""
    return LinearInterpolant()


@pytest.fixture(
    params=[
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ]
)
def device(request):
    return request.param


################################ Alpha and Sigma Tests ######################################


def test_compute_alpha_t_values():
    """Test that alpha(t) = t."""
    interpolant = LinearInterpolant()
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    
    alpha, d_alpha = interpolant.compute_alpha_t(t)
    
    # alpha(t) = t
    expected_alpha = t
    assert torch.allclose(alpha, expected_alpha, atol=1e-7)
    
    # d_alpha(t) = 1
    expected_d_alpha = torch.ones_like(t)
    assert torch.allclose(d_alpha, expected_d_alpha, atol=1e-7)


def test_compute_sigma_t_values():
    """Test that sigma(t) = 1 - t."""
    interpolant = LinearInterpolant()
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    
    sigma, d_sigma = interpolant.compute_sigma_t(t)
    
    # sigma(t) = 1 - t
    expected_sigma = 1 - t
    assert torch.allclose(sigma, expected_sigma, atol=1e-7)
    
    # d_sigma(t) = -1
    expected_d_sigma = -torch.ones_like(t)
    assert torch.allclose(d_sigma, expected_d_sigma, atol=1e-7)


def test_boundary_conditions():
    """Verify boundary conditions: x_0 = x_0, x_1 = x_1."""
    interpolant = LinearInterpolant()
    
    # At t=0: alpha=0, sigma=1, so x_t = x0
    t0 = torch.tensor([0.0])
    alpha_0, _ = interpolant.compute_alpha_t(t0)
    sigma_0, _ = interpolant.compute_sigma_t(t0)
    
    assert torch.allclose(alpha_0, torch.tensor([0.0]), atol=1e-7)
    assert torch.allclose(sigma_0, torch.tensor([1.0]), atol=1e-7)
    
    # At t=1: alpha=1, sigma=0, so x_t = x1
    t1 = torch.tensor([1.0])
    alpha_1, _ = interpolant.compute_alpha_t(t1)
    sigma_1, _ = interpolant.compute_sigma_t(t1)
    
    assert torch.allclose(alpha_1, torch.tensor([1.0]), atol=1e-7)
    assert torch.allclose(sigma_1, torch.tensor([0.0]), atol=1e-7)


def test_d_alpha_alpha_ratio():
    """Test d_alpha/alpha ratio = 1/t."""
    interpolant = LinearInterpolant()
    t = torch.tensor([0.1, 0.25, 0.5, 0.75, 1.0])
    
    ratio = interpolant.compute_d_alpha_alpha_ratio_t(t)
    
    # Expected: 1/t
    expected = 1.0 / t
    assert torch.allclose(ratio, expected, atol=1e-6)


def test_d_alpha_alpha_ratio_near_zero():
    """Test numerical stability of d_alpha/alpha near t=0."""
    interpolant = LinearInterpolant()
    t = torch.tensor([1e-10, 1e-8, 1e-6])
    
    ratio = interpolant.compute_d_alpha_alpha_ratio_t(t)
    
    # Should be clamped, not infinite
    assert torch.all(torch.isfinite(ratio))


################################ Interpolate Method Tests ###################################


def test_interpolate_basic(interpolant, device):
    """Test basic interpolation between noise and data."""
    torch.manual_seed(42)
    batch_size = 32
    dim = 4
    
    x0 = torch.randn(batch_size, dim, device=device)  # noise
    x1 = torch.randn(batch_size, dim, device=device)  # data
    t = torch.rand(batch_size, device=device)
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    assert xt.shape == (batch_size, dim)
    assert ut.shape == (batch_size, dim)
    assert torch.all(torch.isfinite(xt))
    assert torch.all(torch.isfinite(ut))


def test_interpolate_manual_verification():
    """Manually verify interpolation formula: x_t = t*x1 + (1-t)*x0."""
    interpolant = LinearInterpolant()
    
    # Simple case: x0 = 0, x1 = 1
    x0 = torch.zeros(4, 2)
    x1 = torch.ones(4, 2)
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    # Expected x_t = t*1 + (1-t)*0 = t
    expected_xt = t.unsqueeze(-1).expand_as(x0)
    assert torch.allclose(xt, expected_xt, atol=1e-6)
    
    # Expected u_t = d_alpha*x1 + d_sigma*x0 = 1*1 + (-1)*0 = 1
    expected_ut = torch.ones_like(x0)
    assert torch.allclose(ut, expected_ut, atol=1e-6)


def test_interpolate_boundary_t0():
    """At t=0, x_t should equal x0."""
    interpolant = LinearInterpolant()
    torch.manual_seed(42)
    
    x0 = torch.randn(16, 3)
    x1 = torch.randn(16, 3)
    t = torch.zeros(16)
    
    xt, _ = interpolant.interpolate(x0, x1, t)
    
    assert torch.allclose(xt, x0, atol=1e-6)


def test_interpolate_boundary_t1():
    """At t=1, x_t should equal x1."""
    interpolant = LinearInterpolant()
    torch.manual_seed(42)
    
    x0 = torch.randn(16, 3)
    x1 = torch.randn(16, 3)
    t = torch.ones(16)
    
    xt, _ = interpolant.interpolate(x0, x1, t)
    
    assert torch.allclose(xt, x1, atol=1e-6)


def test_interpolate_midpoint():
    """At t=0.5, x_t should be midpoint between x0 and x1."""
    interpolant = LinearInterpolant()
    torch.manual_seed(42)
    
    x0 = torch.randn(16, 3)
    x1 = torch.randn(16, 3)
    t = torch.full((16,), 0.5)
    
    xt, _ = interpolant.interpolate(x0, x1, t)
    
    expected = 0.5 * (x0 + x1)
    assert torch.allclose(xt, expected, atol=1e-6)


def test_interpolate_velocity_is_constant():
    """For linear interpolant, velocity should be constant: u_t = x1 - x0."""
    interpolant = LinearInterpolant()
    torch.manual_seed(42)
    
    x0 = torch.randn(16, 4)
    x1 = torch.randn(16, 4)
    
    # Velocity at different times should be the same
    t1 = torch.full((16,), 0.2)
    t2 = torch.full((16,), 0.5)
    t3 = torch.full((16,), 0.8)
    
    _, ut1 = interpolant.interpolate(x0, x1, t1)
    _, ut2 = interpolant.interpolate(x0, x1, t2)
    _, ut3 = interpolant.interpolate(x0, x1, t3)
    
    # u_t = d_alpha*x1 + d_sigma*x0 = 1*x1 + (-1)*x0 = x1 - x0
    expected_velocity = x1 - x0
    
    assert torch.allclose(ut1, expected_velocity, atol=1e-6)
    assert torch.allclose(ut2, expected_velocity, atol=1e-6)
    assert torch.allclose(ut3, expected_velocity, atol=1e-6)


################################ Compute Drift Tests ########################################


def test_compute_drift_basic(interpolant, device):
    """Test drift computation returns valid tensors."""
    torch.manual_seed(42)
    x = torch.randn(32, 4, device=device)
    t = torch.rand(32, device=device)
    
    drift_mean, drift_var = interpolant.compute_drift(x, t)
    
    assert drift_mean.shape == x.shape
    # drift_var is broadcast-compatible (batch_size, 1, ...)
    assert drift_var.shape[0] == x.shape[0]
    # Verify broadcast works
    _ = drift_var * x
    assert torch.all(torch.isfinite(drift_mean))
    assert torch.all(torch.isfinite(drift_var))


def test_compute_drift_manual():
    """Manually verify drift computation for linear interpolant.
    
    For linear interpolant:
    - alpha = t, d_alpha = 1, so d_alpha/alpha = 1/t
    - sigma = 1-t, d_sigma = -1
    - drift_mean = (d_alpha/alpha) * x = x/t
    - drift_var = (d_alpha/alpha)*sigma² - sigma*d_sigma = (1/t)*(1-t)² + (1-t)
    """
    interpolant = LinearInterpolant()
    
    x = torch.ones(4, 2)
    t = torch.tensor([0.2, 0.4, 0.6, 0.8])
    
    drift_mean, drift_var = interpolant.compute_drift(x, t)
    
    # drift_mean = -alpha_ratio * x (note the negative sign in compute_drift)
    alpha_ratio = 1.0 / t
    expected_drift_mean = -alpha_ratio.unsqueeze(-1) * x
    
    # drift_var = alpha_ratio * sigma² - sigma * d_sigma
    sigma = 1 - t
    d_sigma = -torch.ones_like(t)
    expected_drift_var = alpha_ratio * (sigma ** 2) - sigma * d_sigma
    expected_drift_var = expected_drift_var.unsqueeze(-1).expand_as(x)
    
    assert torch.allclose(drift_mean, expected_drift_mean, atol=1e-5)
    assert torch.allclose(drift_var, expected_drift_var, atol=1e-5)


################################ Compute Diffusion Tests ###################################


def test_compute_diffusion_forms(interpolant, device):
    """Test all diffusion form options."""
    torch.manual_seed(42)
    x = torch.randn(16, 4, device=device)
    t = torch.rand(16, device=device) * 0.8 + 0.1  # Avoid boundaries
    
    for form in ["constant", "SBDM", "sigma", "linear"]:
        diffusion = interpolant.compute_diffusion(x, t, form=form)
        # diffusion is broadcast-compatible (batch_size, 1, ...)
        assert diffusion.shape[0] == x.shape[0]
        # Verify broadcast works
        _ = diffusion * x
        assert torch.all(torch.isfinite(diffusion))


def test_compute_diffusion_invalid_form():
    """Test that invalid diffusion form raises error."""
    interpolant = LinearInterpolant()
    x = torch.randn(4, 2)
    t = torch.rand(4)
    
    with pytest.raises(ValueError, match="Unknown diffusion form"):
        interpolant.compute_diffusion(x, t, form="invalid")


def test_compute_diffusion_constant():
    """Test constant diffusion form."""
    interpolant = LinearInterpolant()
    x = torch.randn(8, 3)
    t = torch.rand(8)
    norm = 2.0
    
    diffusion = interpolant.compute_diffusion(x, t, form="constant", norm=norm)
    
    expected = torch.full_like(x, norm)
    assert torch.allclose(diffusion, expected, atol=1e-6)


################################ Velocity/Score Conversion Tests ###########################


def test_velocity_to_score_and_back(interpolant, device):
    """Test velocity <-> score conversion roundtrip."""
    torch.manual_seed(42)
    x = torch.randn(32, 4, device=device)
    t = torch.rand(32, device=device) * 0.8 + 0.1  # Avoid boundaries
    
    # Create a mock velocity
    velocity = torch.randn_like(x)
    
    # Convert velocity -> score -> velocity
    score = interpolant.velocity_to_score(velocity, x, t)
    velocity_recovered = interpolant.score_to_velocity(score, x, t)
    
    assert torch.allclose(velocity, velocity_recovered, atol=1e-4)


def test_velocity_to_noise_consistency(interpolant, device):
    """Test velocity to noise prediction conversion."""
    torch.manual_seed(42)
    x0 = torch.randn(16, 4, device=device)  # noise
    x1 = torch.randn(16, 4, device=device)  # data
    t = torch.rand(16, device=device) * 0.8 + 0.1  # Avoid boundaries
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    # The true velocity is x1 - x0 
    # If we know the true velocity, we should recover the noise x0
    noise_pred = interpolant.velocity_to_noise(ut, xt, t)
    
    # For linear interpolant with exact velocity, should recover original noise
    assert torch.allclose(noise_pred, x0, atol=1e-4)


################################ Multi-dimensional Tests ###################################


@pytest.mark.parametrize("shape", [(32, 3, 8, 8), (16, 2, 4, 5, 3)])
def test_multi_dimensional_inputs(shape):
    """Test interpolation with multi-dimensional inputs (e.g., images)."""
    interpolant = LinearInterpolant()
    torch.manual_seed(42)
    
    x0 = torch.randn(*shape)
    x1 = torch.randn(*shape)
    t = torch.rand(shape[0])
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    assert xt.shape == shape
    assert ut.shape == shape
    assert torch.all(torch.isfinite(xt))
    assert torch.all(torch.isfinite(ut))


################################ Batch Independence Tests ##################################


def test_batch_independence():
    """Verify each sample in batch is processed independently."""
    interpolant = LinearInterpolant()
    torch.manual_seed(42)
    
    batch_size = 8
    dim = 4
    
    x0 = torch.randn(batch_size, dim)
    x1 = torch.randn(batch_size, dim)
    t = torch.rand(batch_size)
    
    # Full batch
    xt_batch, ut_batch = interpolant.interpolate(x0, x1, t)
    
    # Process individually
    for i in range(batch_size):
        xi0 = x0[i:i+1]
        xi1 = x1[i:i+1]
        ti = t[i:i+1]
        
        xt_i, ut_i = interpolant.interpolate(xi0, xi1, ti)
        
        assert torch.allclose(xt_batch[i], xt_i.squeeze(0), atol=1e-6)
        assert torch.allclose(ut_batch[i], ut_i.squeeze(0), atol=1e-6)


################################ Device and Dtype Tests ####################################


@requires_cuda
def test_cuda_computation():
    """Test computation on CUDA device."""
    interpolant = LinearInterpolant()
    
    x0 = torch.randn(16, 4, device="cuda")
    x1 = torch.randn(16, 4, device="cuda")
    t = torch.rand(16, device="cuda")
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    assert xt.device.type == "cuda"
    assert ut.device.type == "cuda"


def test_float64_precision():
    """Test computation with float64 precision."""
    interpolant = LinearInterpolant()
    
    x0 = torch.randn(16, 4, dtype=torch.float64)
    x1 = torch.randn(16, 4, dtype=torch.float64)
    t = torch.rand(16, dtype=torch.float64)
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    assert xt.dtype == torch.float64
    assert ut.dtype == torch.float64


################################ Numerical Stability Tests #################################


def test_numerical_stability_large_values():
    """Test stability with large input values."""
    interpolant = LinearInterpolant()
    
    x0 = torch.randn(16, 4) * 1000
    x1 = torch.randn(16, 4) * 1000
    t = torch.rand(16)
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    assert torch.all(torch.isfinite(xt))
    assert torch.all(torch.isfinite(ut))


def test_numerical_stability_small_values():
    """Test stability with small input values."""
    interpolant = LinearInterpolant()
    
    x0 = torch.randn(16, 4) * 1e-6
    x1 = torch.randn(16, 4) * 1e-6
    t = torch.rand(16)
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    assert torch.all(torch.isfinite(xt))
    assert torch.all(torch.isfinite(ut))


################################ ODE Integration Test ######################################


def test_ode_integration_linear_path():
    """Verify that integrating the velocity field recovers the interpolation.
    
    For linear interpolant: dx/dt = u_t = x1 - x0 (constant)
    Integrating from t=0 to t=1 should give:
    x(1) = x(0) + integral(x1 - x0, 0, 1) = x0 + (x1 - x0) = x1
    """
    interpolant = LinearInterpolant()
    torch.manual_seed(42)
    
    x0 = torch.randn(32, 4)
    x1 = torch.randn(32, 4)
    
    # Euler integration of dx/dt = x1 - x0
    n_steps = 100
    dt = 1.0 / n_steps
    x = x0.clone()
    
    for step in range(n_steps):
        t = torch.full((32,), step * dt)
        _, ut = interpolant.interpolate(x0, x1, t)
        x = x + ut * dt
    
    # After integration, x should equal x1
    assert torch.allclose(x, x1, atol=1e-3)
