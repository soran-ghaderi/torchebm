"""Tests for BackwardEulerMaruyamaIntegrator."""

import math

import pytest
import torch

from torchebm.core import GaussianModel
from torchebm.integrators import BackwardEulerMaruyamaIntegrator
from tests.conftest import requires_cuda


@pytest.fixture
def integrator():
    """Create a default BackwardEulerMaruyamaIntegrator."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return BackwardEulerMaruyamaIntegrator(device=device, dtype=torch.float32)


@pytest.fixture
def gaussian_model():
    """Create a Gaussian energy model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return GaussianModel(
        mean=torch.zeros(2, device=device), cov=torch.eye(2, device=device)
    ).to(device)


def test_initialization():
    """Test basic initialization."""
    integrator = BackwardEulerMaruyamaIntegrator()
    assert isinstance(integrator, BackwardEulerMaruyamaIntegrator)
    assert integrator.max_iter == 50


def test_initialization_with_device():
    """Test initialization with specific device and dtype."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = BackwardEulerMaruyamaIntegrator(device=device, dtype=torch.float32)
    assert integrator.device == torch.device(device)
    assert integrator.dtype == torch.float32


def test_initialization_with_max_iter():
    """Test that max_iter is configurable."""
    integrator = BackwardEulerMaruyamaIntegrator(max_iter=10)
    assert integrator.max_iter == 10


@requires_cuda
def test_cuda():
    """Test CUDA initialization."""
    integrator = BackwardEulerMaruyamaIntegrator(device="cuda", dtype=torch.float32)
    assert integrator.device == torch.device("cuda")


################################ Step Method Tests ###############################################


def test_step_matches_implicit_euler_closed_form(integrator):
    """Step result matches the closed-form implicit Euler solution for linear drift.

    For dx/dt = -x and step size h, backward Euler gives x_new = x / (1+h),
    not x * (1-h) like the forward (explicit) Euler.
    """
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_
    step_size = 0.05  # h*L = 0.05, well within Picard contraction

    result = integrator.step(state, step_size=step_size, drift=drift)

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    expected = x / (1.0 + step_size)
    assert torch.allclose(result["x"], expected, atol=1e-5)


def test_step_differs_from_forward_euler(integrator):
    """Backward Euler must produce a different result than forward Euler."""
    device = integrator.device
    x = torch.ones(10, 2, device=device)
    drift = lambda x_, t_: -x_
    step_size = 0.1

    result = integrator.step({"x": x.clone()}, step_size=step_size, drift=drift)

    forward = x * (1 - step_size)         # explicit Euler:  x*(1-h)
    backward = x / (1.0 + step_size)      # implicit Euler:  x/(1+h)

    assert torch.allclose(result["x"], backward, atol=1e-5)
    assert not torch.allclose(result["x"], forward, atol=1e-3)


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

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    # Noise must perturb the deterministic implicit-Euler solution.
    deterministic = x / (1.0 + step_size)
    assert not torch.allclose(result["x"], deterministic, atol=1e-5)


def test_step_with_explicit_diffusion(integrator):
    """Test step with explicit diffusion coefficient."""
    device = integrator.device
    torch.manual_seed(42)
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -x_
    diffusion = torch.tensor(0.5, device=device)

    result = integrator.step(
        {"x": x}, step_size=0.01, drift=drift, diffusion=diffusion
    )

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_step_with_time_input(integrator):
    """Test step with explicit time input."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    t = torch.ones(10, device=device) * 0.5
    drift = lambda x_, t_: -x_ * (1 + t_[:, None])

    result = integrator.step({"x": x}, step_size=0.01, drift=drift, t=t)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_step_requires_drift(integrator):
    """Step raises when no drift is supplied."""
    device = integrator.device
    state = {"x": torch.randn(10, 2, device=device)}
    with pytest.raises(ValueError, match="drift must be provided explicitly"):
        integrator.step(state, step_size=0.01)


def test_step_with_model_gradient(integrator, gaussian_model):
    """Step works with model.gradient as drift source."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -gaussian_model.gradient(x_)

    result = integrator.step({"x": x}, step_size=0.01, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_step_drift_evaluated_at_end_time(integrator):
    """Backward Euler evaluates drift at t+h, not t."""
    device = integrator.device
    x = torch.ones(4, 2, device=device)
    seen_times = []

    def drift(x_, t_):
        seen_times.append(t_.clone())
        return -x_

    t = torch.full((4,), 0.3, device=device)
    integrator.step({"x": x}, step_size=0.1, drift=drift, t=t)

    # First call is the predictor at t; subsequent Picard calls evaluate at t+h.
    assert torch.allclose(seen_times[0], torch.full((4,), 0.3, device=device))
    assert torch.allclose(seen_times[1], torch.full((4,), 0.4, device=device))


################################ Integrate Method Tests ############################################


def test_integrate_ode(integrator):
    """Test integration for ODE (no diffusion)."""
    device = integrator.device
    x = torch.ones(10, 2, device=device)
    drift = lambda x_, t_: -x_   # dx/dt = -x  =>  x(1) = x(0) * exp(-1)
    step_size = 0.01
    n_steps = 100

    result = integrator.integrate(
        {"x": x}, step_size=step_size, n_steps=n_steps, drift=drift
    )

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    # Approximate exponential decay (first-order accuracy).
    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, rtol=0.1)


def test_integrate_matches_per_step_closed_form(integrator):
    """Integrate over many steps matches x0 / (1+h)^n for linear drift."""
    device = integrator.device
    x0 = torch.tensor([[1.0, -2.0, 3.0]], device=device)
    drift = lambda x_, t_: -x_
    h, n = 0.05, 40

    result = integrator.integrate({"x": x0}, step_size=h, n_steps=n, drift=drift)
    expected = x0 / (1.0 + h) ** n
    assert torch.allclose(result["x"], expected, atol=1e-5)


def test_integrate_sde(integrator):
    """Test integration for SDE."""
    device = integrator.device
    torch.manual_seed(42)
    x = torch.zeros(100, 2, device=device)
    drift = lambda x_, t_: -x_
    diffusion = lambda x_, t_: torch.ones_like(x_)

    result = integrator.integrate(
        {"x": x},
        step_size=0.01,
        n_steps=100,
        drift=drift,
        diffusion=diffusion,
    )

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_integrate_invalid_n_steps(integrator):
    """integrate raises on non-positive n_steps."""
    device = integrator.device
    state = {"x": torch.randn(10, 2, device=device)}
    drift = lambda x_, t_: -x_

    with pytest.raises(ValueError, match="n_steps must be positive"):
        integrator.integrate(state, step_size=0.01, n_steps=0, drift=drift)
    with pytest.raises(ValueError, match="n_steps must be positive"):
        integrator.integrate(state, step_size=0.01, n_steps=-3, drift=drift)


################################ Stability and Stiffness Tests ######################################


def test_unconditional_stability_for_stiff_linear_drift():
    """Implicit Euler stays bounded for stiff linear drift even when h*L is large.

    For dx/dt = -L*x with L >> 1, the exact decay is fast (x -> 0).  Forward
    Euler explodes when h*L > 2; backward Euler stays bounded for any h > 0
    because the per-step factor 1/(1+h*L) < 1.  We use a small h so Picard
    iteration still converges, but choose L large enough that forward Euler
    would visibly overshoot.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = BackwardEulerMaruyamaIntegrator(device=device, dtype=torch.float64)

    x0 = torch.tensor([[5.0, -7.0]], device=device, dtype=torch.float64)
    L, h, n = 50.0, 0.01, 50  # h*L = 0.5  (Picard contraction OK; forward Euler would oscillate hard)
    drift = lambda x_, t_: -L * x_

    result = integrator.integrate(
        {"x": x0.clone()}, step_size=h, n_steps=n, drift=drift,
    )

    # Implicit Euler: x_n = x0 / (1 + h*L)^n  -> monotonically decaying, bounded by |x0|.
    expected = x0 / (1.0 + h * L) ** n
    assert torch.all(result["x"].abs() <= x0.abs())
    assert torch.allclose(result["x"], expected, atol=1e-6)


def test_first_order_convergence():
    """Halving the step size halves the error for linear drift (first-order method)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = BackwardEulerMaruyamaIntegrator(device=device, dtype=torch.float64)

    x0 = torch.tensor([[2.0, 3.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_
    target = x0 * math.exp(-1.0)

    errors = []
    for step_size in (0.1, 0.01, 0.001):
        n_steps = int(round(1.0 / step_size))
        result = integrator.integrate(
            {"x": x0.clone()}, step_size=step_size, n_steps=n_steps, drift=drift,
        )
        errors.append((result["x"] - target).abs().max().item())

    # First-order method: each 10x reduction in h reduces error by ~10x.
    for i in range(len(errors) - 1):
        ratio = errors[i] / errors[i + 1]
        assert ratio > 5, f"Expected ~10x error reduction, got {ratio}"


################################ Reproducibility and Edge Cases ######################################


def test_reproducibility(integrator):
    """Same seed produces identical SDE step output."""
    device = integrator.device

    torch.manual_seed(42)
    x1 = torch.randn(10, 2, device=device)
    result1 = integrator.step(
        {"x": x1}, step_size=0.01, drift=lambda x_, t_: -x_, noise_scale=1.0,
    )

    torch.manual_seed(42)
    x2 = torch.randn(10, 2, device=device)
    result2 = integrator.step(
        {"x": x2}, step_size=0.01, drift=lambda x_, t_: -x_, noise_scale=1.0,
    )

    assert torch.allclose(result1["x"], result2["x"])


def test_large_batch_size(integrator):
    """Works with large batches."""
    device = integrator.device
    x = torch.randn(1000, 10, device=device)
    result = integrator.step({"x": x}, step_size=0.01, drift=lambda x_, t_: -x_)
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_high_dimension(integrator):
    """Works with high-dimensional inputs."""
    device = integrator.device
    x = torch.randn(10, 100, device=device)
    result = integrator.step({"x": x}, step_size=0.01, drift=lambda x_, t_: -x_)
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_max_iter_one_falls_back_to_predictor(integrator):
    """With max_iter=1 the result equals the explicit-Euler predictor."""
    device = integrator.device
    integ = BackwardEulerMaruyamaIntegrator(
        device=device, dtype=torch.float32, max_iter=1,
    )
    x = torch.tensor([[1.0, -2.0]], device=device)
    drift = lambda x_, t_: -x_
    h = 0.05

    out = integ.step({"x": x}, step_size=h, drift=drift)
    # Predictor seed:    x0 + h * f(x0, t) = x0 * (1 - h)
    # One iteration:     x0 + h * f(predictor, t+h) = x0 * (1 - h + h^2)
    expected = x * (1 - h + h * h)
    assert torch.allclose(out["x"], expected, atol=1e-6)


def test_deprecated_model_kwarg_warns(integrator, gaussian_model):
    """Passing the deprecated ``model`` kwarg emits a DeprecationWarning."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)

    with pytest.warns(DeprecationWarning, match="Passing 'model'"):
        integrator.step({"x": x}, step_size=0.01, model=gaussian_model)
