"""Tests for HeunIntegrator."""

import math

import pytest
import torch
import numpy as np

from torchebm.core import BaseModel, GaussianModel, DoubleWellModel
from torchebm.integrators import HeunIntegrator
from tests.conftest import requires_cuda


################################ Test Fixtures ###########################################


@pytest.fixture
def integrator():
    """Create a default HeunIntegrator."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HeunIntegrator(device=device, dtype=torch.float32)


@pytest.fixture
def gaussian_model():
    """Create a Gaussian energy model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return GaussianModel(
        mean=torch.zeros(2, device=device), cov=torch.eye(2, device=device)
    ).to(device)


################################ Initialization Tests ###########################################


def test_heun_initialization():
    """Test basic initialization."""
    integrator = HeunIntegrator()
    assert isinstance(integrator, HeunIntegrator)


def test_heun_initialization_with_device():
    """Test initialization with specific device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = HeunIntegrator(device=device, dtype=torch.float32)
    assert integrator.device == torch.device(device)
    assert integrator.dtype == torch.float32


@requires_cuda
def test_heun_cuda():
    """Test CUDA initialization."""
    integrator = HeunIntegrator(device="cuda", dtype=torch.float32)
    assert integrator.device == torch.device("cuda")


################################ Step Method Tests ###########################################


def test_step_ode_only(integrator):
    """Test step for ODE (no diffusion)."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    t = torch.zeros(10, device=device)
    drift = lambda x_, t_: -x_  # Simple mean-reverting drift
    step_size = 0.01

    result = integrator.step(state, model=None, step_size=step_size, drift=drift, t=t)

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_step_with_diffusion(integrator):
    """Test step with diffusion (SDE case)."""
    device = integrator.device
    torch.manual_seed(42)
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    t = torch.zeros(10, device=device)
    drift = lambda x_, t_: -x_
    diffusion = torch.tensor(0.5, device=device)
    step_size = 0.01

    result = integrator.step(
        state, model=None, step_size=step_size, drift=drift, t=t, diffusion=diffusion
    )

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_step_with_noise_scale(integrator):
    """Test step with noise_scale parameter."""
    device = integrator.device
    torch.manual_seed(42)
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    t = torch.zeros(10, device=device)
    drift = lambda x_, t_: -x_
    step_size = 0.01

    result = integrator.step(
        state, model=None, step_size=step_size, drift=drift, t=t, noise_scale=1.0
    )

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_step_with_model(integrator, gaussian_model):
    """Test step using model gradient as drift."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    t = torch.zeros(10, device=device)
    step_size = 0.01

    result = integrator.step(state, model=gaussian_model, step_size=step_size, t=t)

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_step_requires_drift_or_model(integrator):
    """Test that step raises error when neither model nor drift is provided."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    t = torch.zeros(10, device=device)

    with pytest.raises(ValueError, match="Either `model` must be provided"):
        integrator.step(state, model=None, step_size=0.01, t=t)


def test_step_with_custom_noise(integrator):
    """Test step with pre-sampled noise."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    t = torch.zeros(10, device=device)
    drift = lambda x_, t_: -x_
    step_size = 0.01
    noise = torch.randn_like(x)

    result = integrator.step(
        state,
        model=None,
        step_size=step_size,
        drift=drift,
        t=t,
        noise=noise,
        noise_scale=1.0,
    )

    assert "x" in result
    assert result["x"].shape == x.shape


################################ Integrate Method Tests ###########################################


def test_integrate_ode(integrator):
    """Test integration for ODE."""
    device = integrator.device
    x = torch.ones(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_  # dx/dt = -x, solution: x(t) = x(0) * exp(-t)
    step_size = 0.02
    n_steps = 50
    t = torch.linspace(0, 1, n_steps, device=device)

    result = integrator.integrate(
        state, model=None, step_size=step_size, n_steps=n_steps, drift=drift, t=t
    )

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    # Verify approximate exponential decay: x(t=1) ≈ x(0) * exp(-1)
    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, rtol=0.05)  # Heun is 2nd order


def test_integrate_sde(integrator):
    """Test integration for SDE."""
    device = integrator.device
    torch.manual_seed(42)
    x = torch.zeros(100, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_
    diffusion = lambda x_, t_: torch.full_like(x_, 0.5)
    n_steps = 100
    t = torch.linspace(0, 1, n_steps, device=device)

    result = integrator.integrate(
        state,
        model=None,
        step_size=0.01,
        n_steps=n_steps,
        drift=drift,
        diffusion=diffusion,
        t=t,
    )

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_integrate_with_model(integrator, gaussian_model):
    """Test integration using model gradient."""
    device = integrator.device
    x = torch.randn(10, 2, device=device) * 3
    state = {"x": x}
    n_steps = 100
    t = torch.linspace(0, 1, n_steps, device=device)

    result = integrator.integrate(
        state, model=gaussian_model, step_size=0.01, n_steps=n_steps, t=t
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
    t = torch.linspace(0, 1, 50, device=device)

    with pytest.raises(ValueError, match="n_steps must be positive"):
        integrator.integrate(
            state, model=None, step_size=0.01, n_steps=0, drift=drift, t=t
        )


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
            state, model=None, step_size=0.02, n_steps=n_steps, drift=drift, t=t_wrong_shape
        )

    # Wrong length
    t_wrong_len = torch.linspace(0, 1, n_steps + 10, device=device)
    with pytest.raises(ValueError, match="t must be a 1D tensor with length n_steps"):
        integrator.integrate(
            state, model=None, step_size=0.02, n_steps=n_steps, drift=drift, t=t_wrong_len
        )


################################# Manual Verification Tests - Heun Method (Predictor-Corrector) ############################################


def test_manual_heun_step():
    """Manual verification of single Heun step (ODE case).

    Heun method (improved Euler / explicit trapezoidal):
    k1 = f(x, t)
    x_pred = x + dt * k1
    k2 = f(x_pred, t + dt)
    x_new = x + 0.5 * dt * (k1 + k2)

    For f(x,t) = -x:
    k1 = -x
    x_pred = x - dt*x = x(1 - dt)
    k2 = -x_pred = -x(1 - dt)
    x_new = x + 0.5*dt*(-x - x(1-dt))
          = x + 0.5*dt*(-x)(2 - dt)
          = x - dt*x + 0.5*dt²*x
          = x*(1 - dt + 0.5*dt²)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = HeunIntegrator(device=device, dtype=torch.float64)

    x = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    state = {"x": x}
    t = torch.zeros(1, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_
    step_size = 0.1

    result = integrator.step(state, model=None, step_size=step_size, drift=drift, t=t)

    # Expected: x_new = x * (1 - dt + 0.5*dt²)
    expected = x * (1 - step_size + 0.5 * step_size**2)
    assert torch.allclose(result["x"], expected, atol=1e-10)


def test_heun_second_order_convergence():
    """Test that Heun method shows second-order convergence."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = HeunIntegrator(device=device, dtype=torch.float64)

    # dx/dt = -x, solution: x(t) = x(0) * exp(-t)
    x0 = torch.tensor([[2.0, 3.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    step_sizes = [0.1, 0.05, 0.025]
    errors = []

    for step_size in step_sizes:
        n_steps = int(1.0 / step_size)
        t = torch.linspace(0, 1.0 - step_size, n_steps, device=device, dtype=torch.float64)
        result = integrator.integrate(
            {"x": x0.clone()}, model=None, step_size=step_size, n_steps=n_steps, drift=drift, t=t
        )
        expected = x0 * math.exp(-1.0)
        # Adjust for incomplete integration
        adjusted_expected = x0 * math.exp(-(1.0 - step_size))
        error = torch.abs(result["x"] - adjusted_expected).max().item()
        errors.append(error)

    # For second-order method, halving step size should reduce error by ~4x
    for i in range(len(errors) - 1):
        ratio = errors[i] / errors[i + 1]
        # Allow some tolerance but expect roughly 4x improvement
        assert ratio > 3.0, f"Expected ~4x error reduction for 2nd order method, got {ratio}"


def test_heun_vs_euler_accuracy():
    """Verify Heun is more accurate than Euler for same step size."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    from torchebm.integrators import EulerMaruyamaIntegrator

    heun = HeunIntegrator(device=device, dtype=dtype)
    euler = EulerMaruyamaIntegrator(device=device, dtype=dtype)

    x0 = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
    drift = lambda x_, t_: -x_
    step_size = 0.1
    n_steps = 10
    t = torch.linspace(0, 1, n_steps, device=device, dtype=dtype)

    heun_result = heun.integrate(
        {"x": x0.clone()}, model=None, step_size=step_size, n_steps=n_steps, drift=drift, t=t
    )
    euler_result = euler.integrate(
        {"x": x0.clone()}, model=None, step_size=step_size, n_steps=n_steps, drift=drift
    )

    expected = x0 * math.exp(-1.0)
    heun_error = torch.abs(heun_result["x"] - expected).max().item()
    euler_error = torch.abs(euler_result["x"] - expected).max().item()

    # Heun should have smaller error
    assert heun_error < euler_error, f"Heun error {heun_error} should be less than Euler error {euler_error}"


def test_quadratic_drift():
    """Test with non-linear drift."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = HeunIntegrator(device=device, dtype=torch.float32)

    x = torch.ones(10, 2, device=device)
    state = {"x": x}
    # Quadratic drift: f(x) = -x^2
    drift = lambda x_, t_: -(x_**2)
    n_steps = 50
    t = torch.linspace(0, 0.5, n_steps, device=device)

    result = integrator.integrate(
        state, model=None, step_size=0.01, n_steps=n_steps, drift=drift, t=t
    )

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))
    # x should decrease (starting from 1 with negative drift)
    assert torch.all(result["x"] < x)


################################# Reproducibility Tests ############################################


def test_reproducibility(integrator):
    """Test that same seed produces same results."""
    device = integrator.device

    torch.manual_seed(42)
    x1 = torch.randn(10, 2, device=device)
    state1 = {"x": x1}
    t1 = torch.zeros(10, device=device)
    result1 = integrator.step(
        state1, model=None, step_size=0.01, drift=lambda x_, t_: -x_, noise_scale=1.0, t=t1
    )

    torch.manual_seed(42)
    x2 = torch.randn(10, 2, device=device)
    state2 = {"x": x2}
    t2 = torch.zeros(10, device=device)
    result2 = integrator.step(
        state2, model=None, step_size=0.01, drift=lambda x_, t_: -x_, noise_scale=1.0, t=t2
    )

    assert torch.allclose(result1["x"], result2["x"])


################################# Edge Cases ############################################


def test_large_batch_size(integrator):
    """Test with large batch size."""
    device = integrator.device
    x = torch.randn(1000, 10, device=device)
    state = {"x": x}
    t = torch.zeros(1000, device=device)
    drift = lambda x_, t_: -x_

    result = integrator.step(state, model=None, step_size=0.01, drift=drift, t=t)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_high_dimension(integrator):
    """Test with high-dimensional input."""
    device = integrator.device
    x = torch.randn(10, 100, device=device)
    state = {"x": x}
    t = torch.zeros(10, device=device)
    drift = lambda x_, t_: -x_

    result = integrator.step(state, model=None, step_size=0.01, drift=drift, t=t)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_time_dependent_drift(integrator):
    """Test with time-dependent drift."""
    device = integrator.device
    x = torch.ones(10, 2, device=device)
    state = {"x": x}
    t = torch.ones(10, device=device) * 0.5
    # Drift depends on time: f(x, t) = -x * (1 + t)
    drift = lambda x_, t_: -x_ * (1 + t_[:, None])

    result = integrator.step(state, model=None, step_size=0.01, drift=drift, t=t)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))
