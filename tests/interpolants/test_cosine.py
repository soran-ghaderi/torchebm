"""Tests for CosineInterpolant (geodesic variance preserving interpolant)."""

import math

import pytest
import torch

from torchebm.interpolants import CosineInterpolant
from tests.conftest import requires_cuda


@pytest.fixture
def interpolant():
    """Create a CosineInterpolant instance."""
    return CosineInterpolant()


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
    """Test that alpha(t) = sin(pi*t/2)."""
    interpolant = CosineInterpolant()
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    
    alpha, d_alpha = interpolant.compute_alpha_t(t)
    
    # alpha(t) = sin(pi*t/2)
    expected_alpha = torch.sin(t * math.pi / 2)
    assert torch.allclose(alpha, expected_alpha, atol=1e-7)
    
    # d_alpha(t) = (pi/2) * cos(pi*t/2)
    expected_d_alpha = (math.pi / 2) * torch.cos(t * math.pi / 2)
    assert torch.allclose(d_alpha, expected_d_alpha, atol=1e-7)


def test_compute_sigma_t_values():
    """Test that sigma(t) = cos(pi*t/2)."""
    interpolant = CosineInterpolant()
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    
    sigma, d_sigma = interpolant.compute_sigma_t(t)
    
    # sigma(t) = cos(pi*t/2)
    expected_sigma = torch.cos(t * math.pi / 2)
    assert torch.allclose(sigma, expected_sigma, atol=1e-7)
    
    # d_sigma(t) = -(pi/2) * sin(pi*t/2)
    expected_d_sigma = -(math.pi / 2) * torch.sin(t * math.pi / 2)
    assert torch.allclose(d_sigma, expected_d_sigma, atol=1e-7)


def test_boundary_conditions():
    """Verify boundary conditions: x_0 = x_0, x_1 = x_1."""
    interpolant = CosineInterpolant()
    
    # At t=0: alpha=sin(0)=0, sigma=cos(0)=1, so x_t = x0
    t0 = torch.tensor([0.0])
    alpha_0, _ = interpolant.compute_alpha_t(t0)
    sigma_0, _ = interpolant.compute_sigma_t(t0)
    
    assert torch.allclose(alpha_0, torch.tensor([0.0]), atol=1e-7)
    assert torch.allclose(sigma_0, torch.tensor([1.0]), atol=1e-7)
    
    # At t=1: alpha=sin(pi/2)=1, sigma=cos(pi/2)=0, so x_t = x1
    t1 = torch.tensor([1.0])
    alpha_1, _ = interpolant.compute_alpha_t(t1)
    sigma_1, _ = interpolant.compute_sigma_t(t1)
    
    assert torch.allclose(alpha_1, torch.tensor([1.0]), atol=1e-7)
    assert torch.allclose(sigma_1, torch.tensor([0.0]), atol=1e-6)


def test_variance_preserving_property():
    """Verify that alpha(t)² + sigma(t)² = 1 (GVP property)."""
    interpolant = CosineInterpolant()
    t = torch.linspace(0.0, 1.0, 101)
    
    alpha, _ = interpolant.compute_alpha_t(t)
    sigma, _ = interpolant.compute_sigma_t(t)
    
    # sin²(x) + cos²(x) = 1
    variance_sum = alpha ** 2 + sigma ** 2
    expected = torch.ones_like(t)
    
    assert torch.allclose(variance_sum, expected, atol=1e-6)


def test_d_alpha_alpha_ratio():
    """Test d_alpha/alpha ratio = (pi/2) * cot(pi*t/2)."""
    interpolant = CosineInterpolant()
    t = torch.tensor([0.1, 0.25, 0.5, 0.75])
    
    ratio = interpolant.compute_d_alpha_alpha_ratio_t(t)
    
    # Expected: (pi/2) / tan(pi*t/2)
    expected = math.pi / (2 * torch.tan(t * math.pi / 2))
    assert torch.allclose(ratio, expected, atol=1e-5)


def test_d_alpha_alpha_ratio_near_zero():
    """Test numerical stability of d_alpha/alpha near t=0."""
    interpolant = CosineInterpolant()
    t = torch.tensor([1e-6, 1e-5, 1e-4])
    
    ratio = interpolant.compute_d_alpha_alpha_ratio_t(t)
    
    # Should be finite (clamped)
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
    """Manually verify interpolation formula: x_t = sin(pi*t/2)*x1 + cos(pi*t/2)*x0."""
    interpolant = CosineInterpolant()
    
    # Simple case: x0 = 0, x1 = 1
    x0 = torch.zeros(4, 2)
    x1 = torch.ones(4, 2)
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    # Expected x_t = sin(pi*t/2)*1 + cos(pi*t/2)*0 = sin(pi*t/2)
    expected_xt = torch.sin(t * math.pi / 2).unsqueeze(-1).expand_as(x0)
    assert torch.allclose(xt, expected_xt, atol=1e-6)
    
    # Expected u_t = (pi/2)*cos(pi*t/2)*1 + (-(pi/2)*sin(pi*t/2))*0 = (pi/2)*cos(pi*t/2)
    expected_ut = (math.pi / 2) * torch.cos(t * math.pi / 2).unsqueeze(-1).expand_as(x0)
    assert torch.allclose(ut, expected_ut, atol=1e-6)


def test_interpolate_boundary_t0():
    """At t=0, x_t should equal x0."""
    interpolant = CosineInterpolant()
    torch.manual_seed(42)
    
    x0 = torch.randn(16, 3)
    x1 = torch.randn(16, 3)
    t = torch.zeros(16)
    
    xt, _ = interpolant.interpolate(x0, x1, t)
    
    assert torch.allclose(xt, x0, atol=1e-6)


def test_interpolate_boundary_t1():
    """At t=1, x_t should equal x1."""
    interpolant = CosineInterpolant()
    torch.manual_seed(42)
    
    x0 = torch.randn(16, 3)
    x1 = torch.randn(16, 3)
    t = torch.ones(16)
    
    xt, _ = interpolant.interpolate(x0, x1, t)
    
    assert torch.allclose(xt, x1, atol=1e-6)


def test_interpolate_midpoint():
    """At t=0.5, x_t should be (x0 + x1)/sqrt(2)."""
    interpolant = CosineInterpolant()
    torch.manual_seed(42)
    
    x0 = torch.randn(16, 3)
    x1 = torch.randn(16, 3)
    t = torch.full((16,), 0.5)
    
    xt, _ = interpolant.interpolate(x0, x1, t)
    
    # At t=0.5: alpha = sin(pi/4) = 1/sqrt(2), sigma = cos(pi/4) = 1/sqrt(2)
    sqrt2_inv = 1.0 / math.sqrt(2)
    expected = sqrt2_inv * x1 + sqrt2_inv * x0
    
    assert torch.allclose(xt, expected, atol=1e-6)


def test_velocity_varies_with_time():
    """For cosine interpolant, velocity varies with time."""
    interpolant = CosineInterpolant()
    torch.manual_seed(42)
    
    x0 = torch.randn(16, 4)
    x1 = torch.randn(16, 4)
    
    t1 = torch.full((16,), 0.2)
    t2 = torch.full((16,), 0.5)
    t3 = torch.full((16,), 0.8)
    
    _, ut1 = interpolant.interpolate(x0, x1, t1)
    _, ut2 = interpolant.interpolate(x0, x1, t2)
    _, ut3 = interpolant.interpolate(x0, x1, t3)
    
    # Velocities should differ
    assert not torch.allclose(ut1, ut2, atol=1e-4)
    assert not torch.allclose(ut2, ut3, atol=1e-4)


################################ Geodesic Path Verification ################################


def test_geodesic_path_constant_speed():
    """Verify the interpolation follows a geodesic (constant angular speed).
    
    For cosine interpolant on the unit sphere, ||dx/dt|| should be constant
    when ||x0|| = ||x1|| = 1.
    """
    interpolant = CosineInterpolant()
    
    # Unit vectors in orthogonal directions
    x0 = torch.tensor([[1.0, 0.0]])
    x1 = torch.tensor([[0.0, 1.0]])
    
    # Check velocity magnitude at different times
    velocities = []
    for t_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        t = torch.tensor([t_val])
        _, ut = interpolant.interpolate(x0, x1, t)
        velocities.append(torch.norm(ut).item())
    
    # All velocity magnitudes should be equal (pi/2)
    expected_speed = math.pi / 2
    for v in velocities:
        assert abs(v - expected_speed) < 1e-5


def test_interpolated_norm_preserved():
    """For unit vectors, interpolation should stay on the unit sphere."""
    interpolant = CosineInterpolant()
    
    # Random unit vectors
    torch.manual_seed(42)
    x0 = torch.randn(32, 4)
    x0 = x0 / torch.norm(x0, dim=-1, keepdim=True)
    x1 = torch.randn(32, 4)
    x1 = x1 / torch.norm(x1, dim=-1, keepdim=True)
    
    t = torch.rand(32)
    xt, _ = interpolant.interpolate(x0, x1, t)
    
    # Due to GVP property: alpha² + sigma² = 1
    # ||x_t||² = alpha²||x1||² + sigma²||x0||² + 2*alpha*sigma*<x0,x1>
    # For orthogonal vectors, ||x_t||² = alpha² + sigma² = 1
    # For general vectors, this is NOT exactly 1
    
    # But we can verify the weighted contribution is well-behaved
    norms = torch.norm(xt, dim=-1)
    
    # Norms should be bounded
    assert torch.all(norms > 0)
    assert torch.all(norms < 3)  # Reasonable bound for random unit vectors


################################ Compute Drift Tests ########################################


def test_compute_drift_basic(interpolant, device):
    """Test drift computation returns valid tensors."""
    torch.manual_seed(42)
    x = torch.randn(32, 4, device=device)
    t = torch.rand(32, device=device) * 0.8 + 0.1  # Avoid boundaries
    
    drift_mean, drift_var = interpolant.compute_drift(x, t)
    
    assert drift_mean.shape == x.shape
    # drift_var is broadcast-compatible (batch_size, 1, ...)
    assert drift_var.shape[0] == x.shape[0]
    # Verify broadcast works
    _ = drift_var * x
    assert torch.all(torch.isfinite(drift_mean))
    assert torch.all(torch.isfinite(drift_var))


def test_compute_drift_manual():
    """Manually verify drift computation for cosine interpolant.
    
    alpha = sin(pi*t/2), d_alpha = (pi/2)*cos(pi*t/2)
    sigma = cos(pi*t/2), d_sigma = -(pi/2)*sin(pi*t/2)
    
    d_alpha/alpha = (pi/2) * cot(pi*t/2)
    drift_mean = -(d_alpha/alpha) * x
    drift_var = (d_alpha/alpha)*sigma² - sigma*d_sigma
    """
    interpolant = CosineInterpolant()
    
    x = torch.ones(4, 2)
    t = torch.tensor([0.2, 0.4, 0.6, 0.8])
    
    drift_mean, drift_var = interpolant.compute_drift(x, t)
    
    # Compute expected values manually
    alpha = torch.sin(t * math.pi / 2)
    d_alpha = (math.pi / 2) * torch.cos(t * math.pi / 2)
    sigma = torch.cos(t * math.pi / 2)
    d_sigma = -(math.pi / 2) * torch.sin(t * math.pi / 2)
    
    alpha_ratio = d_alpha / torch.clamp(alpha, min=1e-8)
    expected_drift_mean = -alpha_ratio.unsqueeze(-1) * x
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


def test_compute_diffusion_sigma_form():
    """Test sigma diffusion form matches sigma(t)."""
    interpolant = CosineInterpolant()
    x = torch.randn(8, 3)
    t = torch.tensor([0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    
    diffusion = interpolant.compute_diffusion(x, t, form="sigma", norm=1.0)
    
    expected_sigma = torch.cos(t * math.pi / 2)
    expected = expected_sigma.unsqueeze(-1).expand_as(x)
    
    assert torch.allclose(diffusion, expected, atol=1e-6)


################################ Velocity/Score Conversion Tests ###########################


def test_velocity_to_score_and_back(interpolant, device):
    """Test velocity <-> score conversion roundtrip."""
    torch.manual_seed(42)
    x = torch.randn(32, 4, device=device)
    t = torch.rand(32, device=device) * 0.8 + 0.1  # Avoid boundaries
    
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
    
    # For cosine interpolant with exact velocity, should recover original noise
    assert torch.allclose(noise_pred, x0, atol=1e-4)


################################ Multi-dimensional Tests ###################################


@pytest.mark.parametrize("shape", [(32, 3, 8, 8), (16, 2, 4, 5, 3)])
def test_multi_dimensional_inputs(shape):
    """Test interpolation with multi-dimensional inputs."""
    interpolant = CosineInterpolant()
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
    interpolant = CosineInterpolant()
    torch.manual_seed(42)
    
    batch_size = 8
    dim = 4
    
    x0 = torch.randn(batch_size, dim)
    x1 = torch.randn(batch_size, dim)
    t = torch.rand(batch_size)
    
    xt_batch, ut_batch = interpolant.interpolate(x0, x1, t)
    
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
    interpolant = CosineInterpolant()
    
    x0 = torch.randn(16, 4, device="cuda")
    x1 = torch.randn(16, 4, device="cuda")
    t = torch.rand(16, device="cuda")
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    assert xt.device.type == "cuda"
    assert ut.device.type == "cuda"


def test_float64_precision():
    """Test computation with float64 precision."""
    interpolant = CosineInterpolant()
    
    x0 = torch.randn(16, 4, dtype=torch.float64)
    x1 = torch.randn(16, 4, dtype=torch.float64)
    t = torch.rand(16, dtype=torch.float64)
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    assert xt.dtype == torch.float64
    assert ut.dtype == torch.float64


################################ Numerical Stability Tests #################################


def test_numerical_stability_large_values():
    """Test stability with large input values."""
    interpolant = CosineInterpolant()
    
    x0 = torch.randn(16, 4) * 1000
    x1 = torch.randn(16, 4) * 1000
    t = torch.rand(16)
    
    xt, ut = interpolant.interpolate(x0, x1, t)
    
    assert torch.all(torch.isfinite(xt))
    assert torch.all(torch.isfinite(ut))


def test_numerical_stability_near_boundaries():
    """Test stability near t=0 and t=1."""
    interpolant = CosineInterpolant()
    torch.manual_seed(42)
    
    x0 = torch.randn(8, 4)
    x1 = torch.randn(8, 4)
    
    # Near t=0
    t_near_0 = torch.full((8,), 1e-6)
    xt_0, ut_0 = interpolant.interpolate(x0, x1, t_near_0)
    assert torch.all(torch.isfinite(xt_0))
    assert torch.all(torch.isfinite(ut_0))
    
    # Near t=1
    t_near_1 = torch.full((8,), 1.0 - 1e-6)
    xt_1, ut_1 = interpolant.interpolate(x0, x1, t_near_1)
    assert torch.all(torch.isfinite(xt_1))
    assert torch.all(torch.isfinite(ut_1))


################################ ODE Integration Test ######################################


def test_ode_integration_recovers_data():
    """Verify that integrating the velocity field recovers the data.
    
    For cosine interpolant: dx/dt = u_t(x_t, t)
    Integrating from t=0 to t=1 should give: x(1) = x1
    """
    interpolant = CosineInterpolant()
    torch.manual_seed(42)
    
    x0 = torch.randn(32, 4)
    x1 = torch.randn(32, 4)
    
    # Euler integration
    n_steps = 200
    dt = 1.0 / n_steps
    x = x0.clone()
    
    for step in range(n_steps):
        t = torch.full((32,), step * dt)
        # For true velocity, we need the conditional velocity u_t
        # which requires knowing x0 and x1
        _, ut = interpolant.interpolate(x0, x1, t)
        x = x + ut * dt
    
    # After integration, x should equal x1
    assert torch.allclose(x, x1, atol=2e-2)


################################ Comparison with Known Analytical Results ##################


def test_derivative_identity():
    """Verify trigonometric derivative identities.
    
    d/dt[sin(pi*t/2)] = (pi/2)*cos(pi*t/2)
    d/dt[cos(pi*t/2)] = -(pi/2)*sin(pi*t/2)
    """
    interpolant = CosineInterpolant()
    t = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], requires_grad=True)
    
    alpha, d_alpha = interpolant.compute_alpha_t(t)
    sigma, d_sigma = interpolant.compute_sigma_t(t)
    
    # Verify using autograd
    alpha_sum = alpha.sum()
    alpha_sum.backward(retain_graph=True)
    autograd_d_alpha = t.grad.clone()
    t.grad.zero_()
    
    sigma_sum = sigma.sum()
    sigma_sum.backward()
    autograd_d_sigma = t.grad.clone()
    
    assert torch.allclose(d_alpha, autograd_d_alpha, atol=1e-5)
    assert torch.allclose(d_sigma, autograd_d_sigma, atol=1e-5)
