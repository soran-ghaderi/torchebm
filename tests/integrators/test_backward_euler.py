"""Tests for BackwardEulerIntegrator."""

import math
import warnings

import pytest
import torch

from torchebm.core import GaussianModel, DoubleWellModel
from torchebm.integrators import BackwardEulerIntegrator, EulerMaruyamaIntegrator
from tests.conftest import requires_cuda


@pytest.fixture
def integrator():
    """Create a default BackwardEulerIntegrator."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return BackwardEulerIntegrator(device=device, dtype=torch.float32)


@pytest.fixture
def integrator_f64():
    """Create a tight-tolerance BackwardEulerIntegrator in float64."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return BackwardEulerIntegrator(
        device=device, dtype=torch.float64, tol=1e-13, n_iter=200
    )


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


def test_backward_euler_initialization():
    """Test basic initialization."""
    integrator = BackwardEulerIntegrator()
    assert isinstance(integrator, BackwardEulerIntegrator)
    assert integrator.n_iter == 50
    assert integrator.tol == 1e-6


def test_backward_euler_initialization_with_device():
    """Test initialization with specific device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = BackwardEulerIntegrator(device=device, dtype=torch.float32)
    assert integrator.device == torch.device(device)
    assert integrator.dtype == torch.float32


def test_backward_euler_initialization_with_solver_params():
    """Test initialization with custom solver parameters."""
    integrator = BackwardEulerIntegrator(n_iter=100, tol=1e-9)
    assert integrator.n_iter == 100
    assert integrator.tol == 1e-9


@requires_cuda
def test_backward_euler_cuda():
    """Test CUDA initialization."""
    integrator = BackwardEulerIntegrator(device="cuda", dtype=torch.float32)
    assert integrator.device == torch.device("cuda")


################################ Step Method Tests ###############################################


def test_step_with_drift_only(integrator):
    """Test step with drift only (ODE case).

    Backward Euler solves x_new = x + h * f(x_new, t+h).
    For f(x) = -x: x_new * (1 + h) = x  =>  x_new = x / (1 + h).
    """
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_
    step_size = 0.01

    result = integrator.step(state, step_size=step_size, drift=drift)

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    expected = x / (1.0 + step_size)
    assert torch.allclose(result["x"], expected, atol=1e-5)


def test_step_with_drift_and_noise(integrator):
    """Test step with drift and noise (SDE case)."""
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

    # Should differ from the deterministic backward-Euler update due to noise
    expected_ode = x / (1.0 + step_size)
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
    """Test step with explicit time input (drift evaluated at t+h)."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    t = torch.ones(10, device=device) * 0.5
    drift = lambda x_, t_: -x_ * (1 + t_[:, None])
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
    """Verify noise_scale and diffusion produce identical results when D = noise_scale²."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -x_
    step_size = 0.01
    noise_scale = 1.5

    noise = torch.randn_like(x)

    result1 = integrator.step(
        {"x": x.clone()}, step_size=step_size, drift=drift,
        noise_scale=noise_scale, noise=noise.clone()
    )

    diffusion = torch.tensor(noise_scale ** 2, device=device)
    result2 = integrator.step(
        {"x": x.clone()}, step_size=step_size, drift=drift,
        diffusion=diffusion, noise=noise.clone()
    )

    assert torch.allclose(result1["x"], result2["x"], atol=1e-6)


def test_ode_vs_sde_behavior(integrator):
    """Verify ODE produces deterministic results while SDE adds stochasticity."""
    device = integrator.device
    x = torch.ones(10, 2, device=device)
    drift = lambda x_, t_: -x_
    step_size = 0.01

    # ODE: deterministic, two calls match
    result_ode = integrator.step({"x": x.clone()}, step_size=step_size, drift=drift)
    result_ode2 = integrator.step({"x": x.clone()}, step_size=step_size, drift=drift)
    assert torch.allclose(result_ode["x"], result_ode2["x"])

    # ODE matches BE analytic update: x / (1 + h)
    expected = x / (1.0 + step_size)
    assert torch.allclose(result_ode["x"], expected, atol=1e-5)

    # SDE produces different results across seeds
    torch.manual_seed(1)
    result_sde1 = integrator.step({"x": x.clone()}, step_size=step_size,
                                   drift=drift, noise_scale=1.0)
    torch.manual_seed(2)
    result_sde2 = integrator.step({"x": x.clone()}, step_size=step_size,
                                   drift=drift, noise_scale=1.0)
    assert not torch.allclose(result_sde1["x"], result_sde2["x"])


################################ Integrate Method Tests ############################################


def test_integrate_ode(integrator):
    """Test integration for ODE (no diffusion).

    For dx/dt = -x with backward Euler, x_n = x_0 / (1 + h)^n
    which approximates x(t) = x_0 * exp(-t).
    """
    device = integrator.device
    x = torch.ones(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_
    step_size = 0.01
    n_steps = 100

    result = integrator.integrate(
        state, step_size=step_size, n_steps=n_steps, drift=drift
    )

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    # Backward Euler: x_n = x_0 / (1 + h)^n  ≈ x_0 * exp(-t) for small h
    expected = x * (1.0 + step_size) ** (-n_steps)
    assert torch.allclose(result["x"], expected, atol=1e-4)


def test_integrate_sde(integrator):
    """Test integration for SDE."""
    device = integrator.device
    torch.manual_seed(42)
    x = torch.zeros(100, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_
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

    # Samples should move toward the mean (0)
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
        integrator.integrate(state, step_size=0.01, n_steps=0, drift=drift)

    with pytest.raises(ValueError, match="n_steps must be positive"):
        integrator.integrate(state, step_size=0.01, n_steps=-5, drift=drift)


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

    t_wrong_shape = torch.linspace(0, 1, n_steps, device=device).reshape(5, 10)
    with pytest.raises(ValueError, match="t must be a 1D tensor"):
        integrator.integrate(
            state, step_size=0.02, n_steps=n_steps, drift=drift, t=t_wrong_shape
        )

    t_too_short = torch.tensor([0.0], device=device)
    with pytest.raises(ValueError, match="t must be a 1D tensor with length >= 2"):
        integrator.integrate(
            state, step_size=0.02, n_steps=1, drift=drift, t=t_too_short
        )


def test_single_step_integration(integrator_f64):
    """Test that 1-step integration works.

    Uses h=0.5 (h*L=0.5) which stays within the fixed-point solver's
    contraction radius.  For f=-x: x_new = x / (1 + h) = x / 1.5.
    """
    device = integrator_f64.device
    x = torch.randn(10, 2, device=device, dtype=torch.float64)
    state = {"x": x}
    drift = lambda x_, t_: -x_

    t = torch.tensor([0.0, 0.5], device=device, dtype=torch.float64)
    result = integrator_f64.integrate(
        state, step_size=0.5, n_steps=1, drift=drift, t=t
    )

    expected = x / 1.5
    assert result["x"].shape == x.shape
    assert torch.allclose(result["x"], expected, atol=1e-10)


################################ Manual Verification Tests ################################


def test_manual_backward_euler_step():
    """Manual verification of single backward-Euler step.

    For f(x) = -x, h = 0.1: x_new * (1+h) = x  =>  x_new = x / 1.1.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = BackwardEulerIntegrator(
        device=device, dtype=torch.float64, tol=1e-13, n_iter=200
    )

    x = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    state = {"x": x}
    drift = lambda x_, t_: -x_
    step_size = 0.1

    result = integrator.step(state, step_size=step_size, drift=drift)

    expected = x / 1.1
    assert torch.allclose(result["x"], expected, atol=1e-10)


def test_manual_backward_euler_sde_step():
    """Manual verification of SDE step with known noise.

    Drift-implicit, diffusion-explicit (matches the SDE-RK family in
    this package — noise is added after the deterministic update).  For
    f = -x:
        x_det = x / (1 + h)               # implicit solve
        x_new = x_det + sqrt(2*D) * dW    # explicit noise increment
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = BackwardEulerIntegrator(
        device=device, dtype=torch.float64, tol=1e-13, n_iter=200
    )

    x = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    state = {"x": x}
    drift = lambda x_, t_: -x_
    step_size = 0.1
    diffusion = torch.tensor(0.5, device=device, dtype=torch.float64)
    noise = torch.tensor([[1.0, -1.0]], device=device, dtype=torch.float64)

    result = integrator.step(
        state,
        step_size=step_size,
        drift=drift,
        diffusion=diffusion,
        noise=noise,
    )

    dw = noise * math.sqrt(step_size)
    stochastic_term = math.sqrt(2.0 * 0.5) * dw
    expected = x / (1.0 + step_size) + stochastic_term
    assert torch.allclose(result["x"], expected, atol=1e-10)


def test_manual_zero_drift_sde_step():
    """With zero drift, BE reduces to a pure diffusion update.

    x_new = x + sqrt(2*D*h) * noise   (no implicit term)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = BackwardEulerIntegrator(
        device=device, dtype=torch.float64, tol=1e-13, n_iter=200
    )
    torch.manual_seed(0)
    x = torch.randn(4, 2, device=device, dtype=torch.float64)
    noise = torch.randn(4, 2, device=device, dtype=torch.float64)

    result = integrator.step(
        {"x": x},
        step_size=0.1,
        drift=lambda x_, t_: torch.zeros_like(x_),
        noise=noise,
        noise_scale=1.0,
    )

    expected = x + math.sqrt(0.2) * noise
    assert torch.allclose(result["x"], expected, atol=1e-12)


def test_exponential_decay_convergence():
    """Test that ODE integration converges with first-order accuracy.

    Both forward and backward Euler are first-order, so global error should
    decrease roughly linearly with step size.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = BackwardEulerIntegrator(
        device=device, dtype=torch.float64, tol=1e-12, n_iter=200
    )

    x0 = torch.tensor([[2.0, 3.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

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

    # First-order: ~10x error reduction per 10x step-size reduction
    for i in range(len(errors) - 1):
        ratio = errors[i] / errors[i + 1]
        assert ratio > 5, f"Expected ~10x error reduction, got {ratio}"


def test_ornstein_uhlenbeck_statistics():
    """Test OU process under BE converges to its stationary distribution.

    Drift-implicit BE update for OU dx = -theta*x dt + sigma dW:
        x_new = x / (1 + theta*h) + sigma * sqrt(h) * z
    Stationary-variance recursion V = V/(1+theta*h)^2 + sigma^2 * h gives
        V = sigma^2 * (1 + theta*h)^2 / (2*theta + theta^2 * h)
    which approaches sigma^2 / (2*theta) as h -> 0.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = BackwardEulerIntegrator(device=device, dtype=torch.float32)

    theta = 1.0
    sigma = math.sqrt(2.0)  # continuous-time stationary variance = 1.0
    n_samples = 2000
    n_steps = 500
    step_size = 0.02

    torch.manual_seed(42)
    x = torch.randn(n_samples, 1, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -theta * x_
    diffusion = lambda x_, t_: torch.full_like(x_, sigma ** 2 / 2)

    result = integrator.integrate(
        state,
        step_size=step_size,
        n_steps=n_steps,
        drift=drift,
        diffusion=diffusion,
    )

    sample_mean = result["x"].mean().item()
    sample_var = result["x"].var().item()
    # BE-discrete stationary variance for outside-solve (drift implicit, noise explicit)
    expected_var = (
        sigma ** 2 * (1.0 + theta * step_size) ** 2
        / (2.0 * theta + theta ** 2 * step_size)
    )

    assert abs(sample_mean) < 0.15, f"Mean {sample_mean} should be close to 0"
    assert abs(sample_var - expected_var) < 0.25, (
        f"Variance {sample_var} should be close to {expected_var}"
    )


########################### Backward-Euler Specific Tests ###########################


def test_backward_euler_vs_forward_euler_small_step(integrator_f64):
    """Backward and forward Euler should agree in the small-step limit."""
    device = integrator_f64.device
    forward = EulerMaruyamaIntegrator(device=device, dtype=torch.float64)

    x = torch.randn(8, 3, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_ + 0.1 * torch.sin(x_)
    step_size = 1e-4

    r_back = integrator_f64.step({"x": x}, step_size=step_size, drift=drift)["x"]
    r_fwd = forward.step({"x": x}, step_size=step_size, drift=drift)["x"]

    assert torch.allclose(r_back, r_fwd, atol=1e-7)


def test_backward_euler_l_stable_in_solver_range():
    """BE remains stable for moderately stiff problems where FE would oscillate.

    For dx/dt = -lambda * x with lambda*h > 1, forward Euler oscillates and
    eventually blows up.  Backward Euler decays monotonically (analytic
    update is x / (1 + lambda*h)).  This test uses lambda*h = 0.9 which is
    within the fixed-point solver's contraction radius.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    back = BackwardEulerIntegrator(
        device=device, dtype=torch.float64, tol=1e-12, n_iter=500
    )

    lam = 9.0
    h = 0.1  # lambda * h = 0.9
    x = torch.ones(4, dtype=torch.float64, device=device)
    drift = lambda x_, t_: -lam * x_

    out = back.step({"x": x}, step_size=h, drift=drift)["x"]
    expected = x / (1.0 + lam * h)
    assert torch.allclose(out, expected, atol=1e-10)
    # And stays bounded — never grows past the input
    assert out.abs().max().item() < x.abs().max().item()


def test_solver_tolerance_respected():
    """The solver should converge within `tol` for a contractive problem."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = BackwardEulerIntegrator(
        device=device, dtype=torch.float64, tol=1e-10, n_iter=200
    )
    x = torch.randn(8, 3, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_
    h = 0.5
    out = integrator.step({"x": x}, step_size=h, drift=drift)["x"]
    expected = x / (1.0 + h)
    assert (out - expected).abs().max().item() < 1e-9


def test_loose_tolerance_respected():
    """Loosening `tol` should give correspondingly looser results."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = BackwardEulerIntegrator(
        device=device, dtype=torch.float64, tol=1e-2, n_iter=200
    )
    x = torch.randn(8, 3, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_
    h = 0.5
    out = integrator.step({"x": x}, step_size=h, drift=drift)["x"]
    expected = x / (1.0 + h)
    err = (out - expected).abs().max().item()
    assert err < 1e-1
    assert err > 0.0  # not converged to machine precision


def test_deprecated_model_kwarg_emits_warning(integrator):
    """Passing `model=` should emit a DeprecationWarning."""
    device = integrator.device

    class _Model:
        def gradient(self, x):
            return x

    x = torch.randn(4, 2, device=device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        integrator.step({"x": x}, step_size=0.01, model=_Model())
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1
        assert "deprecated" in str(deprecations[0].message).lower()


def test_inference_mode(integrator):
    """`inference_mode=True` should run without error and produce finite output."""
    device = integrator.device
    x = torch.randn(8, 3, device=device)
    drift = lambda x_, t_: -x_

    result = integrator.integrate(
        {"x": x}, step_size=0.05, n_steps=10, drift=drift, inference_mode=True
    )
    assert torch.all(torch.isfinite(result["x"]))
    assert result["x"].shape == x.shape


def test_adaptive_false_accepted(integrator):
    """`adaptive=False`/`None` should be accepted for API parity with the SDE-RK base."""
    device = integrator.device
    x = torch.randn(8, 3, device=device)
    drift = lambda x_, t_: -x_

    # Both False and None should run identically to no-kwarg
    out_none = integrator.integrate(
        {"x": x.clone()}, step_size=0.05, n_steps=10, drift=drift
    )
    out_false = integrator.integrate(
        {"x": x.clone()}, step_size=0.05, n_steps=10, drift=drift, adaptive=False
    )
    out_explicit_none = integrator.integrate(
        {"x": x.clone()}, step_size=0.05, n_steps=10, drift=drift, adaptive=None
    )
    assert torch.allclose(out_none["x"], out_false["x"])
    assert torch.allclose(out_none["x"], out_explicit_none["x"])


def test_adaptive_true_rejected(integrator):
    """`adaptive=True` should raise — BE has no embedded error pair."""
    device = integrator.device
    x = torch.randn(8, 3, device=device)
    drift = lambda x_, t_: -x_

    with pytest.raises(ValueError, match="does not define error_weights"):
        integrator.integrate(
            {"x": x}, step_size=0.05, n_steps=10, drift=drift, adaptive=True
        )


########################### Reproducibility Tests ###############################################


def test_reproducibility(integrator):
    """Test that same seed produces same results."""
    device = integrator.device

    torch.manual_seed(42)
    x1 = torch.randn(10, 2, device=device)
    result1 = integrator.step(
        {"x": x1}, step_size=0.01, drift=lambda x_, t_: -x_, noise_scale=1.0
    )

    torch.manual_seed(42)
    x2 = torch.randn(10, 2, device=device)
    result2 = integrator.step(
        {"x": x2}, step_size=0.01, drift=lambda x_, t_: -x_, noise_scale=1.0
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
    drift = lambda x_, t_: -x_ * 0.001

    result = integrator.step(state, step_size=0.01, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))
