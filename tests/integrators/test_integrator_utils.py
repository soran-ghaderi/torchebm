"""Tests for integrator utilities (_integrate_time_grid)."""

import pytest
import torch

from torchebm.integrators import _integrate_time_grid
from tests.conftest import requires_cuda


################################ Test Fixtures ############################################


@pytest.fixture
def device():
    """Get default device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


################################ Basic Functionality Tests ########################################


def test_integrate_time_grid_basic(device):
    """Test basic time grid integration."""
    x = torch.ones(10, 2, device=device)
    t = torch.linspace(0, 1, 50, device=device)

    # Simple step function: x_new = x * 0.99 (independent of t, dt)
    step_fn = lambda x_, t_batch, dt: x_ * 0.99

    result = _integrate_time_grid(x, t, step_fn)

    assert result.shape == x.shape
    assert torch.all(torch.isfinite(result))

    # After 49 steps (50 time points = 49 intervals), expected: 0.99^49 ≈ 0.613
    expected = x * (0.99 ** 49)
    assert torch.allclose(result, expected, rtol=1e-5)


def test_integrate_time_grid_uses_dt(device):
    """Test that dt is passed correctly to step function."""
    x = torch.ones(10, 2, device=device)
    t = torch.tensor([0.0, 0.1, 0.3, 0.6, 1.0], device=device)  # Non-uniform spacing

    captured_dts = []

    def step_fn(x_, t_batch, dt):
        captured_dts.append(dt.item())
        return x_ + dt  # Just add dt to see the effect

    result = _integrate_time_grid(x, t, step_fn)

    # Should have captured 4 dt values
    assert len(captured_dts) == 4
    expected_dts = [0.1, 0.2, 0.3, 0.4]
    for captured, expected in zip(captured_dts, expected_dts):
        assert abs(captured - expected) < 1e-5


def test_integrate_time_grid_uses_t_batch(device):
    """Test that t_batch is correctly expanded for batch dimension."""
    batch_size = 10
    x = torch.ones(batch_size, 2, device=device)
    t = torch.tensor([0.0, 0.5, 1.0], device=device)

    captured_t_batches = []

    def step_fn(x_, t_batch, dt):
        captured_t_batches.append(t_batch.clone())
        return x_

    _integrate_time_grid(x, t, step_fn)

    # Should have captured 2 t_batch values
    assert len(captured_t_batches) == 2

    # Each t_batch should have batch_size elements, all equal
    for t_batch in captured_t_batches:
        assert t_batch.shape == (batch_size,)
        assert torch.all(t_batch == t_batch[0])

    # First call should have t=0.0, second should have t=0.5
    assert torch.allclose(captured_t_batches[0], torch.zeros(batch_size, device=device))
    assert torch.allclose(captured_t_batches[1], torch.ones(batch_size, device=device) * 0.5)


################################ Input Validation Tests ###########################################


def test_integrate_time_grid_requires_1d_tensor(device):
    """Test that t must be a 1D tensor."""
    x = torch.ones(10, 2, device=device)
    t_2d = torch.linspace(0, 1, 50, device=device).reshape(5, 10)
    step_fn = lambda x_, t_batch, dt: x_

    with pytest.raises(ValueError, match="t must be a 1D tensor"):
        _integrate_time_grid(x, t_2d, step_fn)


def test_integrate_time_grid_requires_min_length(device):
    """Test that t must have at least 2 elements."""
    x = torch.ones(10, 2, device=device)
    step_fn = lambda x_, t_batch, dt: x_

    # Single element
    t_single = torch.tensor([0.0], device=device)
    with pytest.raises(ValueError, match="t must have length >= 2"):
        _integrate_time_grid(x, t_single, step_fn)

    # Empty tensor
    t_empty = torch.tensor([], device=device)
    with pytest.raises(ValueError, match="t must have length >= 2"):
        _integrate_time_grid(x, t_empty, step_fn)


def test_integrate_time_grid_accepts_min_length(device):
    """Test that t with exactly 2 elements works."""
    x = torch.ones(10, 2, device=device)
    t = torch.tensor([0.0, 1.0], device=device)
    step_fn = lambda x_, t_batch, dt: x_ * 2.0

    result = _integrate_time_grid(x, t, step_fn)

    # Should have applied step_fn once
    expected = x * 2.0
    assert torch.allclose(result, expected)


################################ Manual Verification Tests ########################################


def test_manual_euler_integration(device):
    """Manually verify Euler integration through time grid.

    dx/dt = -x, solution: x(t) = x(0) * exp(-t)
    Using Euler: x_{n+1} = x_n + dt * (-x_n) = x_n * (1 - dt)
    """
    x0 = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    dt = 0.01
    n_steps = 100
    t = torch.linspace(0, 1.0 - dt, n_steps, device=device, dtype=torch.float64)

    def euler_step(x_, t_batch, dt_):
        return x_ + dt_ * (-x_)  # dx/dt = -x

    result = _integrate_time_grid(x0, t, euler_step)

    # After n_steps-1 steps with dt=0.01
    # Expected: x0 * (1 - dt)^(n_steps-1) ≈ x0 * exp(-0.99)
    expected_numerical = x0 * ((1 - dt) ** (n_steps - 1))
    assert torch.allclose(result, expected_numerical, rtol=1e-10)


def test_non_uniform_time_grid(device):
    """Test with non-uniform time spacing."""
    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    # Non-uniform time spacing
    t = torch.tensor([0.0, 0.1, 0.15, 0.25, 0.5, 1.0], device=device, dtype=torch.float64)

    # Use a step function that depends on dt
    def step_fn(x_, t_batch, dt_):
        return x_ * (1 - dt_)  # Exponential decay

    result = _integrate_time_grid(x, t, step_fn)

    # Manual calculation:
    # x1 = x0 * (1 - 0.1)
    # x2 = x1 * (1 - 0.05)
    # x3 = x2 * (1 - 0.1)
    # x4 = x3 * (1 - 0.25)
    # x5 = x4 * (1 - 0.5)
    expected = x * (1 - 0.1) * (1 - 0.05) * (1 - 0.1) * (1 - 0.25) * (1 - 0.5)
    assert torch.allclose(result, expected, rtol=1e-10)


def test_state_accumulation(device):
    """Test that state is accumulated correctly across steps."""
    x = torch.zeros(5, 3, device=device)
    t = torch.linspace(0, 1, 11, device=device)  # 10 intervals

    # Step function that adds 1 each step
    def step_fn(x_, t_batch, dt_):
        return x_ + 1.0

    result = _integrate_time_grid(x, t, step_fn)

    # After 10 steps, each element should be 10
    expected = torch.full_like(x, 10.0)
    assert torch.allclose(result, expected)


################################ Edge Cases ###########################################


def test_single_sample_batch(device):
    """Test with batch size of 1."""
    x = torch.ones(1, 5, device=device)
    t = torch.linspace(0, 1, 20, device=device)
    step_fn = lambda x_, t_batch, dt: x_ * 0.95

    result = _integrate_time_grid(x, t, step_fn)

    assert result.shape == (1, 5)
    assert torch.all(torch.isfinite(result))


def test_high_dimension(device):
    """Test with high-dimensional x."""
    x = torch.randn(10, 100, device=device)
    t = torch.linspace(0, 1, 50, device=device)
    step_fn = lambda x_, t_batch, dt: x_ * 0.99

    result = _integrate_time_grid(x, t, step_fn)

    assert result.shape == x.shape
    assert torch.all(torch.isfinite(result))


def test_large_batch(device):
    """Test with large batch size."""
    x = torch.randn(1000, 10, device=device)
    t = torch.linspace(0, 1, 20, device=device)
    step_fn = lambda x_, t_batch, dt: x_ - 0.1 * x_

    result = _integrate_time_grid(x, t, step_fn)

    assert result.shape == x.shape
    assert torch.all(torch.isfinite(result))


def test_many_time_steps(device):
    """Test with many time steps."""
    x = torch.ones(10, 2, device=device)
    t = torch.linspace(0, 1, 1000, device=device)
    step_fn = lambda x_, t_batch, dt: x_ * (1 - dt)

    result = _integrate_time_grid(x, t, step_fn)

    assert result.shape == x.shape
    assert torch.all(torch.isfinite(result))
    # Should approximate exp(-1) ≈ 0.368
    assert torch.allclose(result, x * 0.368, rtol=0.05)


################################ Numerical Precision Tests ########################################


def test_double_precision(device):
    """Test with double precision."""
    x = torch.ones(10, 2, device=device, dtype=torch.float64)
    t = torch.linspace(0, 1, 100, device=device, dtype=torch.float64)
    step_fn = lambda x_, t_batch, dt: x_ * (1 - dt)

    result = _integrate_time_grid(x, t, step_fn)

    assert result.dtype == torch.float64
    assert torch.all(torch.isfinite(result))


@requires_cuda
def test_cuda_execution():
    """Test execution on CUDA."""
    x = torch.ones(10, 2, device="cuda")
    t = torch.linspace(0, 1, 50, device="cuda")
    step_fn = lambda x_, t_batch, dt: x_ * 0.99

    result = _integrate_time_grid(x, t, step_fn)

    assert result.device.type == "cuda"
    assert result.shape == x.shape


################################ Time-Dependent Step Function Tests ################################


def test_time_dependent_drift(device):
    """Test with step function that depends on time."""
    x = torch.ones(10, 2, device=device, dtype=torch.float64)
    t = torch.linspace(0, 1, 101, device=device, dtype=torch.float64)

    # Drift that increases with time: f(x, t) = -(1 + t) * x
    def step_fn(x_, t_batch, dt_):
        # Average time in interval
        drift = -(1 + t_batch[:, None]) * x_
        return x_ + dt_ * drift

    result = _integrate_time_grid(x, t, step_fn)

    assert result.shape == x.shape
    assert torch.all(torch.isfinite(result))
    # x should decrease (negative drift)
    assert torch.all(result < x)


def test_position_dependent_step(device):
    """Test with step function that depends on position."""
    x = torch.ones(10, 2, device=device, dtype=torch.float64)
    t = torch.linspace(0, 1, 101, device=device, dtype=torch.float64)

    # Nonlinear: dx/dt = -x^2
    def step_fn(x_, t_batch, dt_):
        return x_ - dt_ * x_**2

    result = _integrate_time_grid(x, t, step_fn)

    assert result.shape == x.shape
    assert torch.all(torch.isfinite(result))
    assert torch.all(result < x)  # x should decrease
    assert torch.all(result > 0)  # But stay positive


################################ Integration with Actual Integrators ################################


def test_euler_maruyama_uses_integrate_time_grid(device):
    """Verify EulerMaruyamaIntegrator uses _integrate_time_grid correctly."""
    from torchebm.integrators import EulerMaruyamaIntegrator

    integrator = EulerMaruyamaIntegrator(device=device, dtype=torch.float32)

    x = torch.ones(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_
    n_steps = 50
    step_size = 0.02

    result = integrator.integrate(
        state, model=None, step_size=step_size, n_steps=n_steps, drift=drift
    )

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_heun_uses_integrate_time_grid(device):
    """Verify HeunIntegrator uses _integrate_time_grid correctly."""
    from torchebm.integrators import HeunIntegrator

    integrator = HeunIntegrator(device=device, dtype=torch.float32)

    x = torch.ones(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_
    n_steps = 50
    t = torch.linspace(0, 1, n_steps, device=device)

    result = integrator.integrate(
        state, model=None, step_size=0.02, n_steps=n_steps, drift=drift, t=t
    )

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))
