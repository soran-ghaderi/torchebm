"""Tests for EulerMaruyamaIntegrator and BackwardEulerMaruyamaIntegrator."""

import math

import pytest
import torch
import numpy as np

from torchebm.core import BaseModel, GaussianModel, DoubleWellModel
from torchebm.integrators import EulerMaruyamaIntegrator, BackwardEulerMaruyamaIntegrator
from tests.conftest import requires_cuda


@pytest.fixture
def integrator():
    """Create a default EulerMaruyamaIntegrator."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return EulerMaruyamaIntegrator(device=device, dtype=torch.float32)


@pytest.fixture
def gaussian_model():
    """Create a Gaussian energy model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return GaussianModel(
        mean=torch.zeros(2, device=device), cov=torch.eye(2, device=device)
    ).to(device)


@pytest.fixture
def double_well_model():
    """Create a double-well energy model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return DoubleWellModel(barrier_height=2.0).to(device)


def test_euler_maruyama_initialization():
    """Test basic initialization."""
    integrator = EulerMaruyamaIntegrator()
    assert isinstance(integrator, EulerMaruyamaIntegrator)


def test_euler_maruyama_initialization_with_device():
    """Test initialization with specific device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = EulerMaruyamaIntegrator(device=device, dtype=torch.float32)
    assert integrator.device == torch.device(device)
    assert integrator.dtype == torch.float32


@requires_cuda
def test_euler_maruyama_cuda():
    """Test CUDA initialization."""
    integrator = EulerMaruyamaIntegrator(device="cuda", dtype=torch.float32)
    assert integrator.device == torch.device("cuda")



################################ Step Method Tests ###############################################

def test_step_with_drift_only(integrator):
    """Test step with drift only (ODE case)."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_  # Simple mean-reverting drift
    step_size = 0.01

    result = integrator.step(state, step_size=step_size, drift=drift)

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    # Verify ODE update: x_new = x - x * dt = x * (1 - dt)
    expected = x * (1 - step_size)
    assert torch.allclose(result["x"], expected, atol=1e-5)


def test_step_with_drift_and_noise(integrator):
    """Test step with drift and diffusion (SDE case)."""
    device = integrator.device
    torch.manual_seed(42)
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_
    step_size = 0.01
    noise_scale = 1.0

    result = integrator.step(
        state, step_size=step_size, drift=drift, noise_scale=noise_scale
    )

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    # Should differ from ODE update due to noise
    expected_ode = x * (1 - step_size)
    assert not torch.allclose(result["x"], expected_ode, atol=1e-5)


def test_step_with_model(integrator, gaussian_model):
    """Test step using model gradient as drift."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    step_size = 0.01
    drift = lambda x_, t_: -gaussian_model.gradient(x_)

    result = integrator.step(state, step_size=step_size, drift=drift)

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_step_with_explicit_diffusion(integrator):
    """Test step with explicit diffusion coefficient."""
    device = integrator.device
    torch.manual_seed(42)
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_
    step_size = 0.01
    diffusion = torch.tensor(0.5, device=device)

    result = integrator.step(
        state, step_size=step_size, drift=drift, diffusion=diffusion
    )

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_step_with_time_input(integrator):
    """Test step with explicit time input."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    t = torch.ones(10, device=device) * 0.5
    drift = lambda x_, t_: -x_ * (1 + t_[:, None])  # Time-dependent drift
    step_size = 0.01

    result = integrator.step(state, step_size=step_size, drift=drift, t=t)

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_step_requires_drift_or_model(integrator):
    """Test that step raises error when neither model nor drift is provided."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}

    with pytest.raises(ValueError, match="drift must be provided explicitly"):
        integrator.step(state, step_size=0.01)


def test_step_with_custom_noise(integrator):
    """Test step with pre-sampled noise."""
    device = integrator.device
    torch.manual_seed(42)
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_
    step_size = 0.01
    noise = torch.randn_like(x)

    result = integrator.step(
        state,
        step_size=step_size,
        drift=drift,
        noise=noise,
        noise_scale=1.0,
    )

    assert "x" in result
    assert result["x"].shape == x.shape


def test_noise_scale_diffusion_equivalence(integrator):
    """Verify noise_scale and diffusion produce identical results when diffusion = noise_scale².
    
    The SDE is: dx = f(x,t)dt + sqrt(2*D)*dW
    When using noise_scale σ, D = σ², so sqrt(2*D) = sqrt(2)*σ
    """
    device = integrator.device
    torch.manual_seed(42)
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -x_
    step_size = 0.01
    noise_scale = 1.5
    
    # Use same noise for both calls
    noise = torch.randn_like(x)
    
    # Call with noise_scale
    torch.manual_seed(42)
    x1 = x.clone()
    result1 = integrator.step(
        {"x": x1}, step_size=step_size, drift=drift,
        noise_scale=noise_scale, noise=noise.clone()
    )
    
    # Call with equivalent diffusion = noise_scale²
    torch.manual_seed(42)
    x2 = x.clone()
    diffusion = torch.tensor(noise_scale**2, device=device)
    result2 = integrator.step(
        {"x": x2}, step_size=step_size, drift=drift,
        diffusion=diffusion, noise=noise.clone()
    )
    
    # Results should be identical
    assert torch.allclose(result1["x"], result2["x"], atol=1e-6)


def test_ode_vs_sde_behavior(integrator):
    """Verify ODE produces deterministic results while SDE adds stochasticity."""
    device = integrator.device
    x = torch.ones(10, 2, device=device)
    drift = lambda x_, t_: -x_
    step_size = 0.01
    
    # ODE: no noise_scale or diffusion
    result_ode = integrator.step({"x": x.clone()}, step_size=step_size, drift=drift)
    
    # Two ODE calls should give identical results
    result_ode2 = integrator.step({"x": x.clone()}, step_size=step_size, drift=drift)
    assert torch.allclose(result_ode["x"], result_ode2["x"])
    
    # ODE should match analytical: x_new = x + (-x)*dt = x*(1-dt)
    expected = x * (1 - step_size)
    assert torch.allclose(result_ode["x"], expected, atol=1e-6)
    
    # SDE should produce different results each call (due to noise)
    torch.manual_seed(1)
    result_sde1 = integrator.step({"x": x.clone()}, step_size=step_size, drift=drift, noise_scale=1.0)
    torch.manual_seed(2)
    result_sde2 = integrator.step({"x": x.clone()}, step_size=step_size, drift=drift, noise_scale=1.0)
    
    # Results should differ
    assert not torch.allclose(result_sde1["x"], result_sde2["x"])


################################ Integrate Method Tests ############################################


def test_integrate_ode(integrator):
    """Test integration for ODE (no diffusion)."""
    device = integrator.device
    x = torch.ones(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_  # dx/dt = -x, solution: x(t) = x(0) * exp(-t)
    step_size = 0.01
    n_steps = 100

    result = integrator.integrate(
        state, step_size=step_size, n_steps=n_steps, drift=drift
    )

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    # Verify approximate exponential decay: x(t=1) ≈ x(0) * exp(-1)
    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, rtol=0.1)


def test_integrate_sde(integrator):
    """Test integration for SDE."""
    device = integrator.device
    torch.manual_seed(42)
    x = torch.zeros(100, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_  # Ornstein-Uhlenbeck process
    diffusion = lambda x_, t_: torch.ones_like(x_)
    step_size = 0.01
    n_steps = 100

    result = integrator.integrate(
        state,
        step_size=step_size,
        n_steps=n_steps,
        drift=drift,
        diffusion=diffusion,
    )

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_integrate_with_model(integrator, gaussian_model):
    """Test integration using model gradient."""
    device = integrator.device
    x = torch.randn(10, 2, device=device) * 3
    state = {"x": x}
    step_size = 0.01
    n_steps = 100
    drift = lambda x_, t_: -gaussian_model.gradient(x_)

    result = integrator.integrate(
        state, step_size=step_size, n_steps=n_steps, drift=drift
    )

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    # Samples should move towards mean (0)
    initial_dist = torch.norm(x, dim=-1).mean()
    final_dist = torch.norm(result["x"], dim=-1).mean()
    assert final_dist < initial_dist


def test_integrate_invalid_n_steps(integrator):
    """Test that integrate raises error for invalid n_steps."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_

    with pytest.raises(ValueError, match="n_steps must be positive"):
        integrator.integrate(
            state, step_size=0.01, n_steps=0, drift=drift
        )

    with pytest.raises(ValueError, match="n_steps must be positive"):
        integrator.integrate(
            state, step_size=0.01, n_steps=-5, drift=drift
        )


def test_integrate_with_custom_time_grid(integrator):
    """Test integration with custom time grid."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_
    n_steps = 50
    t = torch.linspace(0, 1, n_steps, device=device)

    result = integrator.integrate(
        state, step_size=0.02, n_steps=n_steps, drift=drift, t=t
    )

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_integrate_time_grid_validation(integrator):
    """Test time grid validation in integrate."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_
    n_steps = 50

    # Wrong shape (should be 1D)
    t_wrong_shape = torch.linspace(0, 1, n_steps, device=device).reshape(5, 10)
    with pytest.raises(ValueError, match="t must be a 1D tensor"):
        integrator.integrate(
            state, step_size=0.02, n_steps=n_steps, drift=drift, t=t_wrong_shape
        )

    # Too short (must have at least 2 points)
    t_too_short = torch.tensor([0.0], device=device)
    with pytest.raises(ValueError, match="t must be a 1D tensor with length >= 2"):
        integrator.integrate(
            state, step_size=0.02, n_steps=1, drift=drift, t=t_too_short
        )


def test_single_step_integration(integrator):
    """Test that 1-step integration works (important for time-invariant models)."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_  # Simple decay

    # Single step with t having exactly 2 points
    t = torch.tensor([0.0, 1.0], device=device)
    result = integrator.integrate(
        state, step_size=1.0, n_steps=1, drift=drift, t=t
    )

    # Expected: x_new = x + drift(x, 0) * dt = x - x * 1.0 = 0
    assert result["x"].shape == x.shape
    assert torch.allclose(result["x"], torch.zeros_like(x), atol=1e-6)


################################ Manual Verification Tests - Compare Against Known Solutions ################################


def test_manual_euler_step():
    """Manual verification of single Euler-Maruyama step."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = EulerMaruyamaIntegrator(device=device, dtype=torch.float32)

    # Simple case: x = [1, 2], drift = -x, no noise
    x = torch.tensor([[1.0, 2.0]], device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_
    step_size = 0.1

    result = integrator.step(state, step_size=step_size, drift=drift)

    # Expected: x_new = x + drift * dt = [1, 2] + [-1, -2] * 0.1 = [0.9, 1.8]
    expected = torch.tensor([[0.9, 1.8]], device=device)
    assert torch.allclose(result["x"], expected, atol=1e-6)


def test_manual_euler_sde_step():
    """Manual verification of SDE step with known noise."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = EulerMaruyamaIntegrator(device=device, dtype=torch.float32)

    x = torch.tensor([[1.0, 2.0]], device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_  # drift = [-1, -2]
    step_size = 0.1
    diffusion = torch.tensor(0.5, device=device)  # D = 0.5
    noise = torch.tensor([[1.0, -1.0]], device=device)

    result = integrator.step(
        state,
        step_size=step_size,
        drift=drift,
        diffusion=diffusion,
        noise=noise,
    )

    # Expected:
    # drift_term = [-1, -2] * 0.1 = [-0.1, -0.2]
    # dW = noise * sqrt(dt) = [1, -1] * sqrt(0.1) = [0.3162.., -0.3162..]
    # stochastic_term = sqrt(2 * D) * dW = sqrt(1.0) * [0.3162.., -0.3162..]
    # x_new = [1, 2] + [-0.1, -0.2] + [0.3162.., -0.3162..]
    dw = noise * math.sqrt(step_size)
    stochastic_term = math.sqrt(2.0 * 0.5) * dw
    expected = x + torch.tensor([[-0.1, -0.2]], device=device) + stochastic_term

    assert torch.allclose(result["x"], expected, atol=1e-5)


def test_exponential_decay_convergence():
    """Test that ODE integration converges to analytical solution."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = EulerMaruyamaIntegrator(device=device, dtype=torch.float64)

    # dx/dt = -x, solution: x(t) = x(0) * exp(-t)
    x0 = torch.tensor([[2.0, 3.0]], device=device, dtype=torch.float64)
    state = {"x": x0}
    drift = lambda x_, t_: -x_

    # Test with different step sizes
    step_sizes = [0.1, 0.01, 0.001]
    errors = []

    for step_size in step_sizes:
        n_steps = int(1.0 / step_size)
        result = integrator.integrate(
            {"x": x0.clone()}, step_size=step_size, n_steps=n_steps, drift=drift
        )
        expected = x0 * math.exp(-1.0)
        error = torch.abs(result["x"] - expected).max().item()
        errors.append(error)

    # Verify error decreases with smaller step size (first-order convergence)
    for i in range(len(errors) - 1):
        ratio = errors[i] / errors[i + 1]
        assert ratio > 5, f"Expected ~10x error reduction, got {ratio}"


def test_ornstein_uhlenbeck_statistics():
    """Test OU process converges to stationary distribution."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = EulerMaruyamaIntegrator(device=device, dtype=torch.float32)

    # OU process: dX = -θX dt + σ dW
    # Stationary distribution: N(0, σ²/(2θ))
    theta = 1.0
    sigma = math.sqrt(2.0)  # So stationary variance = 1.0
    n_samples = 2000
    n_steps = 500
    step_size = 0.02

    torch.manual_seed(42)
    x = torch.randn(n_samples, 1, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -theta * x_
    diffusion = lambda x_, t_: torch.full_like(x_, sigma**2 / 2)

    result = integrator.integrate(
        state,
        step_size=step_size,
        n_steps=n_steps,
        drift=drift,
        diffusion=diffusion,
    )

    # Check stationary statistics
    sample_mean = result["x"].mean().item()
    sample_var = result["x"].var().item()
    expected_var = sigma**2 / (2 * theta)

    assert abs(sample_mean) < 0.15, f"Mean {sample_mean} should be close to 0"
    assert abs(sample_var - expected_var) < 0.25, f"Variance {sample_var} should be close to {expected_var}"


########################### Reproducibility Tests ###############################################


def test_reproducibility(integrator):
    """Test that same seed produces same results."""
    device = integrator.device

    torch.manual_seed(42)
    x1 = torch.randn(10, 2, device=device)
    state1 = {"x": x1}
    result1 = integrator.step(
        state1, step_size=0.01, drift=lambda x_, t_: -x_, noise_scale=1.0
    )

    torch.manual_seed(42)
    x2 = torch.randn(10, 2, device=device)
    state2 = {"x": x2}
    result2 = integrator.step(
        state2, step_size=0.01, drift=lambda x_, t_: -x_, noise_scale=1.0
    )

    assert torch.allclose(result1["x"], result2["x"])


########################### Edge Cases and Numerical Stability ######################################


def test_large_batch_size(integrator):
    """Test with large batch size."""
    device = integrator.device
    x = torch.randn(1000, 10, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_

    result = integrator.step(state, step_size=0.01, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_high_dimension(integrator):
    """Test with high-dimensional input."""
    device = integrator.device
    x = torch.randn(10, 100, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_

    result = integrator.step(state, step_size=0.01, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_small_step_size(integrator):
    """Test with very small step size."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_

    result = integrator.step(state, step_size=1e-8, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_large_values(integrator):
    """Test numerical stability with large values."""
    device = integrator.device
    x = torch.randn(10, 2, device=device) * 1000
    state = {"x": x}
    drift = lambda x_, t_: -x_ * 0.001  # Small drift coefficient to avoid overflow

    result = integrator.step(state, step_size=0.01, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


###################### BackwardEulerMaruyamaIntegrator Tests ####################################


@pytest.fixture
def bwd_integrator():
    """Create a default BackwardEulerMaruyamaIntegrator."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return BackwardEulerMaruyamaIntegrator(device=device, dtype=torch.float32)


################################ Initialization Tests ################################################


def test_backward_euler_initialization():
    """Test basic initialization."""
    bwd = BackwardEulerMaruyamaIntegrator()
    assert isinstance(bwd, BackwardEulerMaruyamaIntegrator)


def test_backward_euler_initialization_with_device():
    """Test initialization with specific device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bwd = BackwardEulerMaruyamaIntegrator(device=device, dtype=torch.float32)
    assert bwd.device == torch.device(device)
    assert bwd.dtype == torch.float32


def test_backward_euler_initialization_with_solver_params():
    """Test initialization with custom Picard solver parameters."""
    bwd = BackwardEulerMaruyamaIntegrator(solver_max_iter=20, solver_tol=1e-8)
    assert bwd.solver_max_iter == 20
    assert bwd.solver_tol == 1e-8


def test_backward_euler_tableau():
    """Verify Butcher tableau is configured for implicit backward Euler."""
    bwd = BackwardEulerMaruyamaIntegrator()
    assert bwd.tableau_a == ((1.0,),)
    assert bwd.tableau_b == (1.0,)
    assert bwd.tableau_c == (1.0,)
    # Stage 0 must be flagged as implicit (a[0,0] = 1.0 != 0)
    assert bwd._stage_implicit[0] is True


@requires_cuda
def test_backward_euler_cuda():
    """Test CUDA initialization."""
    bwd = BackwardEulerMaruyamaIntegrator(device="cuda", dtype=torch.float32)
    assert bwd.device == torch.device("cuda")


################################ Step Method Tests ###############################################


def test_bwd_step_with_drift_only(bwd_integrator):
    """Test step with drift only (ODE case)."""
    device = bwd_integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -x_

    result = bwd_integrator.step({"x": x}, step_size=0.01, drift=drift)

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_bwd_step_with_drift_and_noise(bwd_integrator):
    """Test step with drift and noise (SDE case)."""
    device = bwd_integrator.device
    torch.manual_seed(42)
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -x_

    result = bwd_integrator.step({"x": x}, step_size=0.01, drift=drift, noise_scale=1.0)

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    # Should differ from ODE update due to noise
    ode_result = bwd_integrator.step({"x": x.clone()}, step_size=0.01, drift=drift)
    assert not torch.allclose(result["x"], ode_result["x"], atol=1e-5)


def test_bwd_step_with_model(bwd_integrator, gaussian_model):
    """Test step using model gradient as drift."""
    device = bwd_integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -gaussian_model.gradient(x_)

    result = bwd_integrator.step({"x": x}, step_size=0.01, drift=drift)

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_bwd_step_with_explicit_diffusion(bwd_integrator):
    """Test step with explicit diffusion coefficient."""
    device = bwd_integrator.device
    torch.manual_seed(42)
    x = torch.randn(10, 2, device=device)
    diffusion = torch.tensor(0.5, device=device)

    result = bwd_integrator.step(
        {"x": x}, step_size=0.01, drift=lambda x_, t_: -x_, diffusion=diffusion
    )

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_bwd_step_with_time_input(bwd_integrator):
    """Test step with explicit time input."""
    device = bwd_integrator.device
    x = torch.randn(10, 2, device=device)
    t = torch.ones(10, device=device) * 0.5
    drift = lambda x_, t_: -x_ * (1 + t_[:, None])

    result = bwd_integrator.step({"x": x}, step_size=0.01, drift=drift, t=t)

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_bwd_step_requires_drift(bwd_integrator):
    """Test that step raises error when drift is not provided."""
    device = bwd_integrator.device
    x = torch.randn(10, 2, device=device)

    with pytest.raises(ValueError, match="drift must be provided explicitly"):
        bwd_integrator.step({"x": x}, step_size=0.01)


def test_bwd_step_with_custom_noise(bwd_integrator):
    """Test step with pre-sampled noise tensor."""
    device = bwd_integrator.device
    x = torch.randn(10, 2, device=device)
    noise = torch.randn_like(x)

    result = bwd_integrator.step(
        {"x": x}, step_size=0.01, drift=lambda x_, t_: -x_,
        noise=noise, noise_scale=1.0
    )

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


################################ Implicit Solver Correctness #####################################


def test_manual_backward_euler_step():
    """Verify single backward Euler step against analytical solution.

    For f(x) = -x, backward Euler solves x_new = x + (-x_new)*dt,
    giving the analytical result x_new = x / (1 + dt).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bwd = BackwardEulerMaruyamaIntegrator(
        device=device, dtype=torch.float64, solver_max_iter=50
    )

    x = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    step_size = 0.1

    result = bwd.step({"x": x}, step_size=step_size, drift=lambda x_, t_: -x_)

    expected = x / (1 + step_size)
    assert torch.allclose(result["x"], expected, atol=1e-5)


def test_manual_backward_euler_sde_step():
    """Verify backward Euler SDE step with known noise.

    Deterministic part: x / (1 + dt).  Noise part: sqrt(2*D) * noise * sqrt(dt).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bwd = BackwardEulerMaruyamaIntegrator(
        device=device, dtype=torch.float64, solver_max_iter=50
    )

    x = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    step_size = 0.1
    diffusion = torch.tensor(0.5, device=device, dtype=torch.float64)
    noise = torch.tensor([[1.0, -1.0]], device=device, dtype=torch.float64)

    result = bwd.step(
        {"x": x}, step_size=step_size, drift=lambda x_, t_: -x_,
        diffusion=diffusion, noise=noise
    )

    x_det = x / (1 + step_size)
    dw = noise * math.sqrt(step_size)
    expected = x_det + math.sqrt(2.0 * 0.5) * dw
    assert torch.allclose(result["x"], expected, atol=1e-5)


def test_backward_euler_differs_from_forward_euler():
    """Verify backward and forward Euler produce different deterministic updates."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fwd = EulerMaruyamaIntegrator(device=device, dtype=torch.float64)
    bwd = BackwardEulerMaruyamaIntegrator(
        device=device, dtype=torch.float64, solver_max_iter=50
    )

    x = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_
    step_size = 0.5

    fwd_result = fwd.step({"x": x.clone()}, step_size=step_size, drift=drift)
    bwd_result = bwd.step({"x": x.clone()}, step_size=step_size, drift=drift)

    # Forward Euler: x*(1 - dt)
    assert torch.allclose(fwd_result["x"], x * (1 - step_size), atol=1e-6)
    # Backward Euler: x / (1 + dt)
    assert torch.allclose(bwd_result["x"], x / (1 + step_size), atol=1e-5)
    assert not torch.allclose(fwd_result["x"], bwd_result["x"])


def test_picard_convergence_with_solver_check():
    """Early-exit Picard (solver_check_every > 0) matches fixed-iteration result."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bwd_check = BackwardEulerMaruyamaIntegrator(
        device=device, dtype=torch.float64,
        solver_max_iter=50, solver_tol=1e-10, solver_check_every=5,
    )
    bwd_fixed = BackwardEulerMaruyamaIntegrator(
        device=device, dtype=torch.float64, solver_max_iter=50,
    )

    x = torch.randn(5, 4, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_
    step_size = 0.05

    r_check = bwd_check.step({"x": x.clone()}, step_size=step_size, drift=drift)
    r_fixed = bwd_fixed.step({"x": x.clone()}, step_size=step_size, drift=drift)

    expected = x / (1 + step_size)
    assert torch.allclose(r_check["x"], expected, atol=1e-6)
    assert torch.allclose(r_fixed["x"], expected, atol=1e-6)


################################ Stability and Contraction Tests #################################


def test_bwd_contraction_in_picard_regime():
    """Backward Euler contracts toward zero for mean-reverting drift in Picard regime.

    Picard iteration in this implementation requires ``step_size * L < 1`` to
    converge.  Within that regime the backward Euler update ``x / (1 + dt*L)``
    is always a contraction.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bwd = BackwardEulerMaruyamaIntegrator(
        device=device, dtype=torch.float64, solver_max_iter=50
    )

    # dt * L = 0.5 — well within Picard convergence
    x = torch.randn(10, 4, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_
    result = bwd.step({"x": x}, step_size=0.5, drift=drift)

    assert torch.all(torch.isfinite(result["x"]))
    assert torch.all(result["x"].abs() < x.abs())
    # Expected analytical value: x / (1 + dt)
    assert torch.allclose(result["x"], x / 1.5, atol=1e-3)


def test_bwd_more_accurate_than_forward_for_mild_stiffness():
    """Both schemes are stable for dt*L < 1, but backward Euler tracks
    the exact exponential decay more accurately than forward Euler.

    For dx/dt = -x integrated over t in [0, 1] with N steps:
      - exact: x(1) = x(0) * exp(-1)
      - forward Euler factor: (1 - dt)^N
      - backward Euler factor: 1 / (1 + dt)^N
    For moderate dt the BE factor is closer to exp(-1).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fwd = EulerMaruyamaIntegrator(device=device, dtype=torch.float64)
    bwd = BackwardEulerMaruyamaIntegrator(
        device=device, dtype=torch.float64, solver_max_iter=50
    )

    x0 = torch.ones(1, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_
    step_size, n_steps = 0.2, 5  # total horizon = 1.0, dt*L = 0.2

    fwd_res = fwd.integrate({"x": x0.clone()}, step_size=step_size, n_steps=n_steps, drift=drift)
    bwd_res = bwd.integrate({"x": x0.clone()}, step_size=step_size, n_steps=n_steps, drift=drift)

    target = x0 * math.exp(-1.0)
    fwd_err = (fwd_res["x"] - target).abs().max().item()
    bwd_err = (bwd_res["x"] - target).abs().max().item()

    # Backward Euler should be the more accurate of the two here
    assert bwd_err < fwd_err


################################ Integrate Method Tests ##########################################


def test_bwd_integrate_ode(bwd_integrator):
    """Test ODE integration produces finite, shape-preserving output."""
    device = bwd_integrator.device
    x = torch.ones(10, 2, device=device)
    drift = lambda x_, t_: -x_

    result = bwd_integrator.integrate(
        {"x": x}, step_size=0.01, n_steps=100, drift=drift
    )

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, rtol=0.1)


def test_bwd_integrate_sde(bwd_integrator):
    """Test SDE integration produces finite output."""
    device = bwd_integrator.device
    torch.manual_seed(42)
    x = torch.zeros(100, 2, device=device)
    diffusion = lambda x_, t_: torch.ones_like(x_)

    result = bwd_integrator.integrate(
        {"x": x}, step_size=0.01, n_steps=100,
        drift=lambda x_, t_: -x_, diffusion=diffusion
    )

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_bwd_integrate_with_model(bwd_integrator, gaussian_model):
    """Integration using model gradient moves samples toward the mean."""
    device = bwd_integrator.device
    x = torch.randn(10, 2, device=device) * 3
    drift = lambda x_, t_: -gaussian_model.gradient(x_)

    result = bwd_integrator.integrate({"x": x}, step_size=0.01, n_steps=100, drift=drift)

    assert torch.all(torch.isfinite(result["x"]))
    assert torch.norm(result["x"], dim=-1).mean() < torch.norm(x, dim=-1).mean()


def test_bwd_integrate_invalid_n_steps(bwd_integrator):
    """Test that integrate raises error for non-positive n_steps."""
    device = bwd_integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -x_

    with pytest.raises(ValueError, match="n_steps must be positive"):
        bwd_integrator.integrate({"x": x}, step_size=0.01, n_steps=0, drift=drift)

    with pytest.raises(ValueError, match="n_steps must be positive"):
        bwd_integrator.integrate({"x": x}, step_size=0.01, n_steps=-5, drift=drift)


def test_bwd_integrate_with_custom_time_grid(bwd_integrator):
    """Test integration with an explicit time grid."""
    device = bwd_integrator.device
    x = torch.randn(10, 2, device=device)
    n_steps = 50
    t = torch.linspace(0, 1, n_steps + 1, device=device)

    result = bwd_integrator.integrate(
        {"x": x}, step_size=0.02, n_steps=n_steps,
        drift=lambda x_, t_: -x_, t=t
    )

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_bwd_integrate_time_grid_validation(bwd_integrator):
    """Malformed time grids raise clear errors."""
    device = bwd_integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -x_

    t_wrong_shape = torch.linspace(0, 1, 50, device=device).reshape(5, 10)
    with pytest.raises(ValueError, match="t must be a 1D tensor"):
        bwd_integrator.integrate(
            {"x": x}, step_size=0.02, n_steps=50, drift=drift, t=t_wrong_shape
        )

    t_too_short = torch.tensor([0.0], device=device)
    with pytest.raises(ValueError, match="t must be a 1D tensor with length >= 2"):
        bwd_integrator.integrate(
            {"x": x}, step_size=0.02, n_steps=1, drift=drift, t=t_too_short
        )


def test_bwd_single_step_integration(bwd_integrator):
    """1-step integration: x_new = x / (1 + dt) for linear drift.

    Uses dt = 0.1 (well within Picard convergence regime where dt*L = 0.1).
    """
    device = bwd_integrator.device
    x = torch.randn(10, 2, device=device)
    t = torch.tensor([0.0, 0.1], device=device)

    result = bwd_integrator.integrate(
        {"x": x}, step_size=0.1, n_steps=1, drift=lambda x_, t_: -x_, t=t
    )

    expected = x / 1.1  # x / (1 + dt)
    assert torch.allclose(result["x"], expected, atol=1e-4)


################################ Convergence and Statistical Tests ################################


def test_bwd_exponential_decay_convergence():
    """First-order convergence for dx/dt = -x with backward Euler."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bwd = BackwardEulerMaruyamaIntegrator(
        device=device, dtype=torch.float64, solver_max_iter=50
    )

    x0 = torch.tensor([[2.0, 3.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    step_sizes = [0.1, 0.01, 0.001]
    errors = []
    for step_size in step_sizes:
        n_steps = int(1.0 / step_size)
        result = bwd.integrate({"x": x0.clone()}, step_size=step_size, n_steps=n_steps, drift=drift)
        error = torch.abs(result["x"] - x0 * math.exp(-1.0)).max().item()
        errors.append(error)

    for i in range(len(errors) - 1):
        ratio = errors[i] / errors[i + 1]
        assert ratio > 5, f"Expected ~10x error reduction, got {ratio:.2f}"


def test_bwd_ornstein_uhlenbeck_statistics():
    """OU process converges to stationary distribution N(0, σ²/(2θ))."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bwd = BackwardEulerMaruyamaIntegrator(device=device, dtype=torch.float32)

    theta = 1.0
    sigma = math.sqrt(2.0)  # stationary variance = σ²/(2θ) = 1.0

    torch.manual_seed(42)
    x = torch.randn(2000, 1, device=device)
    drift = lambda x_, t_: -theta * x_
    diffusion = lambda x_, t_: torch.full_like(x_, sigma**2 / 2)

    result = bwd.integrate(
        {"x": x}, step_size=0.02, n_steps=500, drift=drift, diffusion=diffusion
    )

    sample_mean = result["x"].mean().item()
    sample_var = result["x"].var().item()
    expected_var = sigma**2 / (2 * theta)

    assert abs(sample_mean) < 0.15, f"Mean {sample_mean:.4f} should be near 0"
    assert abs(sample_var - expected_var) < 0.25, (
        f"Variance {sample_var:.4f} should be near {expected_var}"
    )


################################ Noise and Diffusion Tests #######################################


def test_bwd_noise_scale_diffusion_equivalence(bwd_integrator):
    """noise_scale and diffusion give identical results when diffusion = noise_scale²."""
    device = bwd_integrator.device
    torch.manual_seed(42)
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -x_
    noise_scale = 1.5
    noise = torch.randn_like(x)

    result1 = bwd_integrator.step(
        {"x": x.clone()}, step_size=0.01, drift=drift,
        noise_scale=noise_scale, noise=noise.clone()
    )
    diffusion = torch.tensor(noise_scale**2, device=device)
    result2 = bwd_integrator.step(
        {"x": x.clone()}, step_size=0.01, drift=drift,
        diffusion=diffusion, noise=noise.clone()
    )

    assert torch.allclose(result1["x"], result2["x"], atol=1e-6)


def test_bwd_ode_vs_sde_behavior(bwd_integrator):
    """ODE path is deterministic; SDE path is stochastic."""
    device = bwd_integrator.device
    x = torch.ones(10, 2, device=device)
    drift = lambda x_, t_: -x_

    r1 = bwd_integrator.step({"x": x.clone()}, step_size=0.01, drift=drift)
    r2 = bwd_integrator.step({"x": x.clone()}, step_size=0.01, drift=drift)
    assert torch.allclose(r1["x"], r2["x"])

    torch.manual_seed(1)
    s1 = bwd_integrator.step({"x": x.clone()}, step_size=0.01, drift=drift, noise_scale=1.0)
    torch.manual_seed(2)
    s2 = bwd_integrator.step({"x": x.clone()}, step_size=0.01, drift=drift, noise_scale=1.0)
    assert not torch.allclose(s1["x"], s2["x"])


################################ Reproducibility Tests ###########################################


def test_bwd_reproducibility(bwd_integrator):
    """Same seed produces identical results."""
    device = bwd_integrator.device

    torch.manual_seed(42)
    x1 = torch.randn(10, 2, device=device)
    result1 = bwd_integrator.step(
        {"x": x1}, step_size=0.01, drift=lambda x_, t_: -x_, noise_scale=1.0
    )

    torch.manual_seed(42)
    x2 = torch.randn(10, 2, device=device)
    result2 = bwd_integrator.step(
        {"x": x2}, step_size=0.01, drift=lambda x_, t_: -x_, noise_scale=1.0
    )

    assert torch.allclose(result1["x"], result2["x"])


################################ Edge Cases and Numerical Stability ##############################


def test_bwd_large_batch_size(bwd_integrator):
    """Test with large batch size."""
    device = bwd_integrator.device
    x = torch.randn(1000, 10, device=device)

    result = bwd_integrator.step({"x": x}, step_size=0.01, drift=lambda x_, t_: -x_)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_bwd_high_dimension(bwd_integrator):
    """Test with high-dimensional input."""
    device = bwd_integrator.device
    x = torch.randn(10, 100, device=device)

    result = bwd_integrator.step({"x": x}, step_size=0.01, drift=lambda x_, t_: -x_)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_bwd_small_step_size(bwd_integrator):
    """Test with very small step size."""
    device = bwd_integrator.device
    x = torch.randn(10, 2, device=device)

    result = bwd_integrator.step({"x": x}, step_size=1e-8, drift=lambda x_, t_: -x_)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_bwd_large_values(bwd_integrator):
    """Test numerical stability with large initial values."""
    device = bwd_integrator.device
    x = torch.randn(10, 2, device=device) * 1000

    result = bwd_integrator.step(
        {"x": x}, step_size=0.01, drift=lambda x_, t_: -x_ * 0.001
    )

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))
