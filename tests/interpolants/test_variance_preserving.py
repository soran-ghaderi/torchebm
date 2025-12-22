"""Tests for VariancePreservingInterpolant (DDPM-style schedule)."""

import math

import pytest
import torch

from torchebm.interpolants import VariancePreservingInterpolant
from tests.conftest import requires_cuda


@pytest.fixture
def interpolant():
    """Create a VariancePreservingInterpolant with default parameters."""
    return VariancePreservingInterpolant()


@pytest.fixture
def interpolant_custom():
    """Create a VariancePreservingInterpolant with custom parameters."""
    return VariancePreservingInterpolant(sigma_min=0.01, sigma_max=50.0)


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


################################ Initialization Tests #######################################


def test_default_initialization():
    """Test default initialization parameters."""
    interpolant = VariancePreservingInterpolant()
    assert interpolant.sigma_min == 0.1
    assert interpolant.sigma_max == 20.0


def test_custom_initialization():
    """Test custom initialization parameters."""
    interpolant = VariancePreservingInterpolant(sigma_min=0.05, sigma_max=50.0)
    assert interpolant.sigma_min == 0.05
    assert interpolant.sigma_max == 50.0


################################ Alpha and Sigma Tests ######################################


def test_log_mean_coeff():
    """Test the log mean coefficient computation.
    
    log_mean_coeff(t) = -0.25 * (1-t)² * (sigma_max - sigma_min) - 0.5 * (1-t) * sigma_min
    """
    interpolant = VariancePreservingInterpolant(sigma_min=0.1, sigma_max=20.0)
    t = torch.tensor([0.0, 0.5, 1.0])
    
    lmc = interpolant._log_mean_coeff(t)
    
    # At t=0: lmc = -0.25 * 1 * 19.9 - 0.5 * 1 * 0.1 = -4.975 - 0.05 = -5.025
    # At t=1: lmc = -0.25 * 0 - 0.5 * 0 = 0
    expected_t0 = -0.25 * 1.0 * (20.0 - 0.1) - 0.5 * 1.0 * 0.1
    expected_t1 = 0.0
    
    assert torch.isclose(lmc[0], torch.tensor(expected_t0), atol=1e-5)
    assert torch.isclose(lmc[2], torch.tensor(expected_t1), atol=1e-5)


def test_d_log_mean_coeff():
    """Test the derivative of log mean coefficient.
    
    d_log_mean_coeff(t) = 0.5 * (1-t) * (sigma_max - sigma_min) + 0.5 * sigma_min
    """
    interpolant = VariancePreservingInterpolant(sigma_min=0.1, sigma_max=20.0)
    t = torch.tensor([0.0, 0.5, 1.0])
    
    d_lmc = interpolant._d_log_mean_coeff(t)
    
    # At t=0: d_lmc = 0.5 * 1 * 19.9 + 0.5 * 0.1 = 9.95 + 0.05 = 10.0
    # At t=1: d_lmc = 0.5 * 0 * 19.9 + 0.5 * 0.1 = 0 + 0.05 = 0.05
    expected_t0 = 0.5 * 1.0 * (20.0 - 0.1) + 0.5 * 0.1
    expected_t1 = 0.5 * 0.1
    
    assert torch.isclose(d_lmc[0], torch.tensor(expected_t0), atol=1e-5)
    assert torch.isclose(d_lmc[2], torch.tensor(expected_t1), atol=1e-5)


def test_compute_alpha_t_values():
    """Test alpha(t) = exp(log_mean_coeff(t))."""
    interpolant = VariancePreservingInterpolant()
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    
    alpha, d_alpha = interpolant.compute_alpha_t(t)
    
    # Manual computation
    lmc = interpolant._log_mean_coeff(t)
    expected_alpha = torch.exp(lmc)
    expected_d_alpha = expected_alpha * interpolant._d_log_mean_coeff(t)
    
    assert torch.allclose(alpha, expected_alpha, atol=1e-6)
    assert torch.allclose(d_alpha, expected_d_alpha, atol=1e-5)


def test_compute_sigma_t_values():
    """Test sigma(t) = sqrt(1 - alpha(t)²)."""
    interpolant = VariancePreservingInterpolant()
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    
    sigma, d_sigma = interpolant.compute_sigma_t(t)
    
    # Verify sigma² + alpha² ≈ 1 (VP property)
    alpha, _ = interpolant.compute_alpha_t(t)
    variance_sum = alpha ** 2 + sigma ** 2
    
    assert torch.allclose(variance_sum, torch.ones_like(t), atol=1e-5)


def test_variance_preserving_property():
    """Verify that alpha(t)² + sigma(t)² = 1."""
    interpolant = VariancePreservingInterpolant()
    t = torch.linspace(0.01, 1.0, 100)  # Avoid t=0 for numerical stability
    
    alpha, _ = interpolant.compute_alpha_t(t)
    sigma, _ = interpolant.compute_sigma_t(t)
    
    variance_sum = alpha ** 2 + sigma ** 2
    expected = torch.ones_like(t)
    
    assert torch.allclose(variance_sum, expected, atol=1e-5)


def test_boundary_conditions():
    """Verify boundary conditions at t=0 and t=1."""
    interpolant = VariancePreservingInterpolant()
    
    # At t=0: high noise (alpha near 0, sigma near 1)
    t0 = torch.tensor([0.0])
    alpha_0, _ = interpolant.compute_alpha_t(t0)
    sigma_0, _ = interpolant.compute_sigma_t(t0)
    
    # At t=0, with default params, alpha should be very small
    assert alpha_0.item() < 0.01
    # sigma should be near 1
    assert sigma_0.item() > 0.99
    
    # At t=1: pure signal (alpha = 1, sigma = 0)
    t1 = torch.tensor([1.0])
    alpha_1, _ = interpolant.compute_alpha_t(t1)
    sigma_1, _ = interpolant.compute_sigma_t(t1)
    
    assert torch.isclose(alpha_1, torch.tensor([1.0]), atol=1e-5)
    assert sigma_1.item() < 0.01


def test_d_alpha_alpha_ratio():
    """Test d_alpha/alpha ratio equals d_log_mean_coeff."""
    interpolant = VariancePreservingInterpolant()
    t = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    
    ratio = interpolant.compute_d_alpha_alpha_ratio_t(t)
    
    # For VP: d(log(alpha))/dt = d_log_mean_coeff
    expected = interpolant._d_log_mean_coeff(t)
    
    assert torch.allclose(ratio, expected, atol=1e-6)


################################ Interpolate Method Tests ###################################


def test_interpolate_basic(interpolant, device):
    """Test basic interpolation between noise and data."""
    torch.manual_seed(42)
    batch_size = 32
    dim = 4
    
    x0 = torch.randn(batch_size, dim, device=device)
    x1 = torch.randn(batch_size, dim, device=device)
    t = torch.rand(batch_size, device=device) * 0.99 + 0.01  # Avoid boundaries
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    assert xt.shape == (batch_size, dim)
    assert ut.shape == (batch_size, dim)
    assert torch.all(torch.isfinite(xt))
    assert torch.all(torch.isfinite(ut))


def test_interpolate_manual_verification():
    """Manually verify interpolation formula: x_t = alpha(t)*x1 + sigma(t)*x0."""
    interpolant = VariancePreservingInterpolant()
    
    x0 = torch.zeros(4, 2)
    x1 = torch.ones(4, 2)
    t = torch.tensor([0.1, 0.4, 0.7, 1.0])
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    # Manual computation
    alpha, d_alpha = interpolant.compute_alpha_t(t)
    sigma, d_sigma = interpolant.compute_sigma_t(t)
    
    expected_xt = alpha.unsqueeze(-1) * x1 + sigma.unsqueeze(-1) * x0
    expected_ut = d_alpha.unsqueeze(-1) * x1 + d_sigma.unsqueeze(-1) * x0
    
    assert torch.allclose(xt, expected_xt, atol=1e-6)
    assert torch.allclose(ut, expected_ut, atol=1e-5)


def test_interpolate_boundary_t1():
    """At t=1, x_t should be very close to x1."""
    interpolant = VariancePreservingInterpolant()
    torch.manual_seed(42)
    
    x0 = torch.randn(16, 3)
    x1 = torch.randn(16, 3)
    t = torch.ones(16)
    
    xt, _ = interpolant.interpolate(x0, x1, t)
    
    # At t=1, alpha=1, sigma≈0, so x_t ≈ x1
    assert torch.allclose(xt, x1, atol=1e-4)


def test_interpolate_near_t0():
    """Near t=0, x_t should be dominated by x0 (noise)."""
    interpolant = VariancePreservingInterpolant()
    torch.manual_seed(42)
    
    x0 = torch.randn(16, 3)
    x1 = torch.randn(16, 3)
    t = torch.full((16,), 0.001)
    
    xt, _ = interpolant.interpolate(x0, x1, t)
    
    # Check that x0 dominates
    sigma, _ = interpolant.compute_sigma_t(t[0:1])
    alpha, _ = interpolant.compute_alpha_t(t[0:1])
    
    # sigma should be much larger than alpha
    assert sigma.item() > alpha.item() * 10


################################ Compute Drift Tests (VP-specific) ##########################


def test_compute_drift_vp_specific():
    """Test VP-specific drift computation using beta parameterization.
    
    drift_mean = -0.5 * beta(t) * x
    drift_var = beta(t) / 2
    where beta(t) = sigma_min + (1-t) * (sigma_max - sigma_min)
    """
    interpolant = VariancePreservingInterpolant(sigma_min=0.1, sigma_max=20.0)
    
    x = torch.ones(4, 2)
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    
    drift_mean, drift_var = interpolant.compute_drift(x, t)
    
    # VP-specific override computes:
    # beta_t = sigma_min + (1-t) * (sigma_max - sigma_min)
    # drift_mean = -0.5 * beta_t * x
    # drift_var = beta_t / 2
    beta_t = 0.1 + (1 - t) * (20.0 - 0.1)
    expected_drift_mean = -0.5 * beta_t.unsqueeze(-1) * x
    
    # drift_var is (batch, 1) broadcast-compatible
    expected_drift_var = (beta_t / 2).unsqueeze(-1)
    
    assert torch.allclose(drift_mean, expected_drift_mean, atol=1e-5)
    # Check values match after broadcasting or squeeze comparison
    assert torch.allclose(drift_var.squeeze(-1), expected_drift_var.squeeze(-1), atol=1e-5)


def test_compute_drift_beta_schedule():
    """Verify beta schedule is linear from sigma_max to sigma_min."""
    interpolant = VariancePreservingInterpolant(sigma_min=0.1, sigma_max=20.0)
    
    x = torch.ones(2, 3)
    t0 = torch.tensor([0.0, 0.0])
    t1 = torch.tensor([1.0, 1.0])
    
    _, drift_var_t0 = interpolant.compute_drift(x, t0)
    _, drift_var_t1 = interpolant.compute_drift(x, t1)
    
    # At t=0: beta = sigma_max = 20.0, drift_var = 10.0
    # At t=1: beta = sigma_min = 0.1, drift_var = 0.05
    assert torch.allclose(drift_var_t0, torch.full_like(x, 10.0), atol=1e-5)
    assert torch.allclose(drift_var_t1, torch.full_like(x, 0.05), atol=1e-5)


################################ Compute Diffusion Tests ###################################


def test_compute_diffusion_forms(interpolant, device):
    """Test all diffusion form options."""
    torch.manual_seed(42)
    x = torch.randn(16, 4, device=device)
    t = torch.rand(16, device=device) * 0.8 + 0.1
    
    for form in ["constant", "SBDM", "sigma", "linear"]:
        diffusion = interpolant.compute_diffusion(x, t, form=form)
        # diffusion is broadcast-compatible (batch_size, 1, ...)
        assert diffusion.shape[0] == x.shape[0]
        # Verify broadcast works
        _ = diffusion * x
        assert torch.all(torch.isfinite(diffusion))


################################ Velocity/Score Conversion Tests ###########################


def test_velocity_to_score_and_back(interpolant, device):
    """Test velocity <-> score conversion roundtrip."""
    torch.manual_seed(42)
    x = torch.randn(32, 4, device=device)
    t = torch.rand(32, device=device) * 0.8 + 0.1
    
    velocity = torch.randn_like(x)
    
    score = interpolant.velocity_to_score(velocity, x, t)
    velocity_recovered = interpolant.score_to_velocity(score, x, t)
    
    assert torch.allclose(velocity, velocity_recovered, atol=1e-4)


def test_velocity_to_noise_consistency(interpolant, device):
    """Test velocity to noise prediction conversion."""
    torch.manual_seed(42)
    x0 = torch.randn(16, 4, device=device)
    x1 = torch.randn(16, 4, device=device)
    t = torch.rand(16, device=device) * 0.8 + 0.1
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    noise_pred = interpolant.velocity_to_noise(ut, xt, t)
    
    assert torch.allclose(noise_pred, x0, atol=1e-3)


################################ DDPM Consistency Tests ####################################


def test_ddpm_forward_process():
    """Verify forward process matches DDPM formulation.
    
    In DDPM: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    
    Our formulation at time t with x0=noise, x1=data:
    x_t = alpha(t) * x1 + sigma(t) * x0
    
    At t=1 (clean): x_t = x1
    At t=0 (noisy): x_t ≈ x0
    """
    interpolant = VariancePreservingInterpolant()
    torch.manual_seed(42)
    
    noise = torch.randn(32, 4)  # x0
    data = torch.randn(32, 4)   # x1
    
    # Forward process at various noise levels
    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for t_val in t_values:
        t = torch.full((32,), t_val)
        xt, _ = interpolant.interpolate(noise, data, t)
        
        alpha, _ = interpolant.compute_alpha_t(t[:1])
        sigma, _ = interpolant.compute_sigma_t(t[:1])
        
        # Verify x_t = alpha * data + sigma * noise
        expected = alpha * data + sigma * noise
        assert torch.allclose(xt, expected, atol=1e-5)


def test_snr_monotonic():
    """Verify signal-to-noise ratio is monotonically increasing with t.
    
    SNR(t) = alpha(t)² / sigma(t)²
    """
    interpolant = VariancePreservingInterpolant()
    t = torch.linspace(0.01, 0.99, 100)
    
    alpha, _ = interpolant.compute_alpha_t(t)
    sigma, _ = interpolant.compute_sigma_t(t)
    
    snr = (alpha ** 2) / (sigma ** 2 + 1e-8)
    
    # SNR should be increasing
    snr_diffs = snr[1:] - snr[:-1]
    assert torch.all(snr_diffs > 0), "SNR should be monotonically increasing"


################################ Different Sigma Parameters Tests ##########################


@pytest.mark.parametrize("sigma_min,sigma_max", [
    (0.01, 10.0),
    (0.1, 20.0),
    (0.001, 100.0),
    (1.0, 5.0),
])
def test_various_sigma_parameters(sigma_min, sigma_max):
    """Test interpolant with various sigma parameters."""
    interpolant = VariancePreservingInterpolant(sigma_min=sigma_min, sigma_max=sigma_max)
    torch.manual_seed(42)
    
    x0 = torch.randn(16, 4)
    x1 = torch.randn(16, 4)
    t = torch.rand(16) * 0.98 + 0.01
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    assert torch.all(torch.isfinite(xt))
    assert torch.all(torch.isfinite(ut))
    
    # VP property should still hold
    alpha, _ = interpolant.compute_alpha_t(t)
    sigma, _ = interpolant.compute_sigma_t(t)
    variance_sum = alpha ** 2 + sigma ** 2
    assert torch.allclose(variance_sum, torch.ones_like(t), atol=1e-4)


################################ Multi-dimensional Tests ###################################


@pytest.mark.parametrize("shape", [(32, 3, 8, 8), (16, 2, 4, 5, 3)])
def test_multi_dimensional_inputs(shape):
    """Test interpolation with multi-dimensional inputs."""
    interpolant = VariancePreservingInterpolant()
    torch.manual_seed(42)
    
    x0 = torch.randn(*shape)
    x1 = torch.randn(*shape)
    t = torch.rand(shape[0]) * 0.98 + 0.01
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    assert xt.shape == shape
    assert ut.shape == shape
    assert torch.all(torch.isfinite(xt))
    assert torch.all(torch.isfinite(ut))


################################ Batch Independence Tests ##################################


def test_batch_independence():
    """Verify each sample in batch is processed independently."""
    interpolant = VariancePreservingInterpolant()
    torch.manual_seed(42)
    
    batch_size = 8
    dim = 4
    
    x0 = torch.randn(batch_size, dim)
    x1 = torch.randn(batch_size, dim)
    t = torch.rand(batch_size) * 0.98 + 0.01
    
    xt_batch, ut_batch = interpolant.interpolate(x0, x1, t)
    
    for i in range(batch_size):
        xi0 = x0[i:i+1]
        xi1 = x1[i:i+1]
        ti = t[i:i+1]
        
        xt_i, ut_i = interpolant.interpolate(xi0, xi1, ti)
        
        assert torch.allclose(xt_batch[i], xt_i.squeeze(0), atol=1e-6)
        assert torch.allclose(ut_batch[i], ut_i.squeeze(0), atol=1e-5)


################################ Device and Dtype Tests ####################################


@requires_cuda
def test_cuda_computation():
    """Test computation on CUDA device."""
    interpolant = VariancePreservingInterpolant()
    
    x0 = torch.randn(16, 4, device="cuda")
    x1 = torch.randn(16, 4, device="cuda")
    t = torch.rand(16, device="cuda") * 0.98 + 0.01
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    assert xt.device.type == "cuda"
    assert ut.device.type == "cuda"


def test_float64_precision():
    """Test computation with float64 precision."""
    interpolant = VariancePreservingInterpolant()
    
    x0 = torch.randn(16, 4, dtype=torch.float64)
    x1 = torch.randn(16, 4, dtype=torch.float64)
    t = torch.rand(16, dtype=torch.float64) * 0.98 + 0.01
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    assert xt.dtype == torch.float64
    assert ut.dtype == torch.float64


################################ Numerical Stability Tests #################################


def test_numerical_stability_large_values():
    """Test stability with large input values."""
    interpolant = VariancePreservingInterpolant()
    
    x0 = torch.randn(16, 4) * 1000
    x1 = torch.randn(16, 4) * 1000
    t = torch.rand(16) * 0.98 + 0.01
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    assert torch.all(torch.isfinite(xt))
    assert torch.all(torch.isfinite(ut))


def test_numerical_stability_near_t0():
    """Test numerical stability near t=0 (high noise regime)."""
    interpolant = VariancePreservingInterpolant()
    torch.manual_seed(42)
    
    x0 = torch.randn(16, 4)
    x1 = torch.randn(16, 4)
    t = torch.full((16,), 1e-4)
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    assert torch.all(torch.isfinite(xt))
    assert torch.all(torch.isfinite(ut))
    
    # Drift should also be stable
    drift_mean, drift_var = interpolant.compute_drift(xt, t)
    assert torch.all(torch.isfinite(drift_mean))
    assert torch.all(torch.isfinite(drift_var))


def test_numerical_stability_sigma_near_zero():
    """Test stability when sigma approaches zero (near t=1)."""
    interpolant = VariancePreservingInterpolant()
    torch.manual_seed(42)
    
    x = torch.randn(16, 4)
    t = torch.full((16,), 0.9999)
    
    # Diffusion with sigma form should handle small sigma
    diffusion = interpolant.compute_diffusion(x, t, form="sigma")
    assert torch.all(torch.isfinite(diffusion))


################################ Derivative Verification ###################################


def test_alpha_derivative_numerical():
    """Verify alpha derivative numerically."""
    interpolant = VariancePreservingInterpolant()
    
    eps = 1e-4
    t = torch.tensor([0.3, 0.5, 0.7])
    
    alpha, d_alpha = interpolant.compute_alpha_t(t)
    
    # Numerical derivative
    alpha_plus, _ = interpolant.compute_alpha_t(t + eps)
    alpha_minus, _ = interpolant.compute_alpha_t(t - eps)
    numerical_d_alpha = (alpha_plus - alpha_minus) / (2 * eps)
    
    # Loosened tolerance for numerical derivative check
    assert torch.allclose(d_alpha, numerical_d_alpha, atol=1e-3)


def test_sigma_derivative_numerical():
    """Verify sigma derivative numerically."""
    interpolant = VariancePreservingInterpolant()
    
    eps = 1e-5
    t = torch.tensor([0.3, 0.5, 0.7])
    
    sigma, d_sigma = interpolant.compute_sigma_t(t)
    
    # Numerical derivative
    sigma_plus, _ = interpolant.compute_sigma_t(t + eps)
    sigma_minus, _ = interpolant.compute_sigma_t(t - eps)
    numerical_d_sigma = (sigma_plus - sigma_minus) / (2 * eps)
    
    # Loosened tolerance for numerical derivative check
    assert torch.allclose(d_sigma, numerical_d_sigma, atol=5e-3)


################################ ODE Integration Test ######################################


def test_ode_integration_recovers_data():
    """Verify that integrating the velocity field recovers the data."""
    interpolant = VariancePreservingInterpolant()
    torch.manual_seed(42)
    
    x0 = torch.randn(32, 4)
    x1 = torch.randn(32, 4)
    
    # Euler integration
    n_steps = 500
    dt = 1.0 / n_steps
    x = x0.clone()
    
    for step in range(n_steps):
        t = torch.full((32,), step * dt)
        _, ut = interpolant.interpolate(x0, x1, t)
        x = x + ut * dt
    
    # After integration, x should be close to x1
    assert torch.allclose(x, x1, atol=5e-2)
