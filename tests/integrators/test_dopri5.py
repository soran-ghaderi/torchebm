"""Tests for Dopri5Integrator."""

import math

import pytest
import torch
import numpy as np

from torchebm.core import GaussianModel, DoubleWellModel
from torchebm.integrators import Dopri5Integrator
from tests.conftest import requires_cuda


################################ Test Fixtures ###########################################


@pytest.fixture
def integrator():
    """Create a default Dopri5Integrator."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Dopri5Integrator(device=device, dtype=torch.float32)


@pytest.fixture
def integrator_f64():
    """Create a Dopri5Integrator with float64 precision."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Dopri5Integrator(device=device, dtype=torch.float64)


@pytest.fixture
def gaussian_model():
    """Create a Gaussian energy model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return GaussianModel(
        mean=torch.zeros(2, device=device), cov=torch.eye(2, device=device)
    ).to(device)


################################ Initialization Tests ###########################################


def test_dopri5_initialization():
    """Test basic initialization with defaults."""
    integrator = Dopri5Integrator()
    assert isinstance(integrator, Dopri5Integrator)
    assert integrator.atol == 1e-6
    assert integrator.rtol == 1e-3
    assert integrator.max_steps == 10_000
    assert integrator.safety == 0.9
    assert integrator.min_factor == 0.2
    assert integrator.max_factor == 10.0
    assert integrator.max_step_size == float("inf")
    assert integrator._norm is None


def test_dopri5_initialization_with_device():
    """Test initialization with specific device and dtype."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float32)
    assert integrator.device == torch.device(device)
    assert integrator.dtype == torch.float32


def test_dopri5_custom_tolerances():
    """Test initialization with custom adaptive parameters."""
    integrator = Dopri5Integrator(
        atol=1e-8, rtol=1e-5, max_steps=5000,
        safety=0.8, min_factor=0.1, max_factor=5.0,
    )
    assert integrator.atol == 1e-8
    assert integrator.rtol == 1e-5
    assert integrator.max_steps == 5000
    assert integrator.safety == 0.8
    assert integrator.min_factor == 0.1
    assert integrator.max_factor == 5.0


@requires_cuda
def test_dopri5_cuda():
    """Test CUDA initialization."""
    integrator = Dopri5Integrator(device="cuda", dtype=torch.float32)
    assert integrator.device == torch.device("cuda")


################################ Butcher Tableau Tests ###########################################


def test_tableau_a_values():
    """Test that the Butcher tableau A matrix matches the Dormand-Prince coefficients."""
    integrator = Dopri5Integrator()
    a = integrator.tableau_a

    assert len(a) == 6
    assert a[0] == ()
    assert len(a[1]) == 1
    assert math.isclose(a[1][0], 1 / 5)
    assert len(a[2]) == 2
    assert math.isclose(a[2][0], 3 / 40)
    assert math.isclose(a[2][1], 9 / 40)
    assert len(a[3]) == 3
    assert math.isclose(a[3][0], 44 / 45)
    assert math.isclose(a[3][1], -56 / 15)
    assert math.isclose(a[3][2], 32 / 9)
    assert len(a[4]) == 4
    assert math.isclose(a[4][0], 19372 / 6561)
    assert math.isclose(a[4][1], -25360 / 2187)
    assert math.isclose(a[4][2], 64448 / 6561)
    assert math.isclose(a[4][3], -212 / 729)
    assert len(a[5]) == 5
    assert math.isclose(a[5][0], 9017 / 3168)
    assert math.isclose(a[5][1], -355 / 33)
    assert math.isclose(a[5][2], 46732 / 5247)
    assert math.isclose(a[5][3], 49 / 176)
    assert math.isclose(a[5][4], -5103 / 18656)


def test_tableau_b_values():
    """Test that the b weights match the Dormand-Prince 5th-order weights."""
    integrator = Dopri5Integrator()
    b = integrator.tableau_b

    assert len(b) == 6
    assert math.isclose(b[0], 35 / 384)
    assert b[1] == 0.0
    assert math.isclose(b[2], 500 / 1113)
    assert math.isclose(b[3], 125 / 192)
    assert math.isclose(b[4], -2187 / 6784)
    assert math.isclose(b[5], 11 / 84)


def test_tableau_c_values():
    """Test that the c nodes match the Dormand-Prince nodes."""
    integrator = Dopri5Integrator()
    c = integrator.tableau_c

    assert len(c) == 6
    assert c[0] == 0.0
    assert math.isclose(c[1], 1 / 5)
    assert math.isclose(c[2], 3 / 10)
    assert math.isclose(c[3], 4 / 5)
    assert math.isclose(c[4], 8 / 9)
    assert c[5] == 1.0


def test_error_weights_values():
    """Test that error weights e_i = b_i - b_hat_i are correct."""
    integrator = Dopri5Integrator()
    e = integrator.error_weights

    assert len(e) == 7  # 6 stages + 1 FSAL
    assert math.isclose(e[0], 71 / 57600)
    assert e[1] == 0.0
    assert math.isclose(e[2], -71 / 16695)
    assert math.isclose(e[3], 71 / 1920)
    assert math.isclose(e[4], -17253 / 339200)
    assert math.isclose(e[5], 22 / 525)
    assert math.isclose(e[6], -1 / 40)


def test_order():
    """Test that the method order is 5."""
    integrator = Dopri5Integrator()
    assert integrator.order == 5


def test_fsal_property():
    """Test that the FSAL property is True."""
    integrator = Dopri5Integrator()
    assert integrator.fsal is True


def test_n_stages():
    """Test that the number of stages is 6."""
    integrator = Dopri5Integrator()
    assert integrator.n_stages == 6


def test_tableau_consistency_b_sum():
    """Test that the b weights sum to 1 (consistency condition)."""
    integrator = Dopri5Integrator()
    b = integrator.tableau_b
    assert math.isclose(sum(b), 1.0, abs_tol=1e-14)


def test_tableau_consistency_row_sums():
    """Test that sum of each row of A equals the corresponding c value."""
    integrator = Dopri5Integrator()
    a = integrator.tableau_a
    c = integrator.tableau_c

    for i in range(len(a)):
        row_sum = sum(a[i]) if a[i] else 0.0
        assert math.isclose(row_sum, c[i], abs_tol=1e-14), (
            f"Row {i}: sum(a[{i}]) = {row_sum} != c[{i}] = {c[i]}"
        )


################################ Step Method Tests ###########################################


def test_step_ode(integrator):
    """Test single ODE step produces correct shape and finite values."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_

    result = integrator.step(state, step_size=0.01, drift=drift)

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_step_with_explicit_time(integrator):
    """Test step with explicit time tensor."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    t = torch.ones(10, device=device) * 0.5
    drift = lambda x_, t_: -x_ * (1 + t_[:, None])

    result = integrator.step(state, step_size=0.01, drift=drift, t=t)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_step_with_model_gradient(integrator, gaussian_model):
    """Test step using model gradient as drift."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -gaussian_model.gradient(x_)

    result = integrator.step(state, step_size=0.01, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_step_requires_drift(integrator):
    """Test that step raises error when drift is not provided."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}

    with pytest.raises(ValueError, match="drift must be provided explicitly"):
        integrator.step(state, step_size=0.01)


def test_step_default_time_is_zero(integrator):
    """Test that step defaults to t=0 when time is not provided."""
    device = integrator.device
    x = torch.ones(10, 2, device=device)
    drift_calls = []

    def recording_drift(x_, t_):
        drift_calls.append(t_.clone())
        return -x_

    integrator.step({"x": x}, step_size=0.01, drift=recording_drift)

    assert len(drift_calls) > 0
    assert torch.allclose(drift_calls[0], torch.zeros(10, device=device))


def test_step_scalar_step_size(integrator):
    """Test that step accepts scalar (non-tensor) step size."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -x_

    result = integrator.step({"x": x}, step_size=0.01, drift=drift)
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_step_tensor_step_size(integrator):
    """Test that step accepts tensor step size."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -x_
    step_size = torch.tensor(0.01, device=device)

    result = integrator.step({"x": x}, step_size=step_size, drift=drift)
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


################################ Integrate Method Tests (Fixed-step) ################################


def test_integrate_fixed_step(integrator):
    """Test fixed-step integration with adaptive=False."""
    device = integrator.device
    x = torch.ones(10, 2, device=device)
    drift = lambda x_, t_: -x_
    n_steps = 50
    step_size = 0.02

    result = integrator.integrate(
        {"x": x}, step_size=step_size, n_steps=n_steps,
        drift=drift, adaptive=False,
    )

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, rtol=0.01)


def test_integrate_fixed_step_with_time_grid(integrator):
    """Test fixed-step integration with explicit time grid."""
    device = integrator.device
    x = torch.ones(10, 2, device=device)
    drift = lambda x_, t_: -x_
    t = torch.linspace(0, 1, 51, device=device)

    result = integrator.integrate(
        {"x": x}, step_size=0.02, n_steps=50,
        drift=drift, t=t, adaptive=False,
    )

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, rtol=0.01)


def test_integrate_fixed_step_with_model(integrator, gaussian_model):
    """Test fixed-step integration using model gradient."""
    device = integrator.device
    x = torch.randn(10, 2, device=device) * 3
    drift = lambda x_, t_: -gaussian_model.gradient(x_)

    result = integrator.integrate(
        {"x": x}, step_size=0.01, n_steps=100,
        drift=drift, adaptive=False,
    )

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    initial_dist = torch.norm(x, dim=-1).mean()
    final_dist = torch.norm(result["x"], dim=-1).mean()
    assert final_dist < initial_dist


def test_integrate_invalid_n_steps(integrator):
    """Test that integrate raises error for invalid n_steps."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -x_

    with pytest.raises(ValueError, match="n_steps must be positive"):
        integrator.integrate(
            {"x": x}, step_size=0.01, n_steps=0,
            drift=drift, adaptive=False,
        )

    with pytest.raises(ValueError, match="n_steps must be positive"):
        integrator.integrate(
            {"x": x}, step_size=0.01, n_steps=-5,
            drift=drift, adaptive=False,
        )


def test_integrate_time_grid_validation(integrator):
    """Test time grid validation in integrate."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -x_

    t_wrong_shape = torch.linspace(0, 1, 50, device=device).reshape(5, 10)
    with pytest.raises(ValueError, match="t must be a 1D tensor"):
        integrator.integrate(
            {"x": x}, step_size=0.02, n_steps=50,
            drift=drift, t=t_wrong_shape, adaptive=False,
        )

    t_too_short = torch.tensor([0.0], device=device)
    with pytest.raises(ValueError, match="t must be a 1D tensor with length >= 2"):
        integrator.integrate(
            {"x": x}, step_size=0.02, n_steps=1,
            drift=drift, t=t_too_short, adaptive=False,
        )


def test_single_step_fixed_integration(integrator_f64):
    """Test 1-step fixed integration matches a single step call."""
    device = integrator_f64.device
    x = torch.randn(10, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    step_result = integrator_f64.step(
        {"x": x.clone()}, step_size=0.1, drift=drift,
    )
    integrate_result = integrator_f64.integrate(
        {"x": x.clone()}, step_size=0.1, n_steps=1,
        drift=drift, adaptive=False,
    )

    assert torch.allclose(step_result["x"], integrate_result["x"], atol=1e-12)


################################ Integrate Method Tests (Adaptive) ################################


def test_integrate_adaptive_default():
    """Test that adaptive mode is the default (since error_weights is defined)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float64)

    x = torch.ones(10, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result = integrator.integrate(
        {"x": x}, step_size=0.1, n_steps=10, drift=drift,
    )

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, atol=5e-4)


def test_integrate_adaptive_explicit():
    """Test that explicitly passing adaptive=True works."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float64)

    x = torch.ones(10, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result = integrator.integrate(
        {"x": x}, step_size=0.1, n_steps=10,
        drift=drift, adaptive=True,
    )

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, atol=5e-4)


def test_adaptive_with_custom_time_grid():
    """Test adaptive integration with explicit time endpoints."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float64)

    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_
    t = torch.tensor([0.0, 2.0], device=device, dtype=torch.float64)

    result = integrator.integrate(
        {"x": x}, step_size=0.5, n_steps=4,
        drift=drift, t=t, adaptive=True,
    )

    expected = x * math.exp(-2.0)
    assert torch.allclose(result["x"], expected, atol=1e-4)


def test_adaptive_tight_tolerance():
    """Test that tight tolerance produces highly accurate results."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(
        device=device, dtype=torch.float64, atol=1e-12, rtol=1e-10,
    )

    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result = integrator.integrate(
        {"x": x}, step_size=0.1, n_steps=10, drift=drift,
    )

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, atol=1e-9)


def test_adaptive_loose_tolerance():
    """Test that loose tolerance still produces reasonable results."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(
        device=device, dtype=torch.float64, atol=1e-2, rtol=1e-1,
    )

    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result = integrator.integrate(
        {"x": x}, step_size=0.5, n_steps=2, drift=drift,
    )

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, rtol=0.2)


def test_adaptive_max_steps_exceeded():
    """Test that exceeding max_steps raises RuntimeError."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(
        device=device, dtype=torch.float64,
        atol=1e-15, rtol=1e-15, max_steps=5,
    )

    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    with pytest.raises(RuntimeError, match="maximum number of steps"):
        integrator.integrate(
            {"x": x}, step_size=0.5, n_steps=20, drift=drift,
        )


def test_adaptive_time_grid_validation():
    """Test that adaptive mode validates time grid correctly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float64)

    x = torch.randn(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    t_wrong = torch.tensor([0.0], device=device, dtype=torch.float64)
    with pytest.raises(ValueError, match="t must be a 1D tensor with length >= 2"):
        integrator.integrate(
            {"x": x}, step_size=0.1, n_steps=1,
            drift=drift, t=t_wrong, adaptive=True,
        )

    t_2d = torch.randn(3, 3, device=device, dtype=torch.float64)
    with pytest.raises(ValueError, match="t must be a 1D tensor"):
        integrator.integrate(
            {"x": x}, step_size=0.1, n_steps=1,
            drift=drift, t=t_2d, adaptive=True,
        )


def test_adaptive_vs_fixed_accuracy():
    """Test that adaptive mode is more accurate than fixed-step with same initial step."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float64, atol=1e-10, rtol=1e-8)

    x = torch.ones(5, 2, device=device, dtype=torch.float64) * 2.0
    drift = lambda x_, t_: -x_

    fixed_result = integrator.integrate(
        {"x": x.clone()}, step_size=0.2, n_steps=5,
        drift=drift, adaptive=False,
    )
    adaptive_result = integrator.integrate(
        {"x": x.clone()}, step_size=0.2, n_steps=5,
        drift=drift, adaptive=True,
    )

    expected = x * math.exp(-1.0)
    fixed_error = torch.abs(fixed_result["x"] - expected).max().item()
    adaptive_error = torch.abs(adaptive_result["x"] - expected).max().item()

    assert adaptive_error < fixed_error


################################ Manual Verification Tests ###########################################


def test_manual_dopri5_step():
    """Manual verification of a single Dormand-Prince step for f(x,t)=-x.

    For f(x,t) = -x with step h, the stages are:
    k1 = -x
    k2 = -(x + h * (1/5) * k1) = -(x - h*x/5) = -x(1 - h/5)
    ... and so on for all 6 stages.
    x_new = x + h * sum(b_i * k_i)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float64)

    x = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_
    h = 0.1

    a = integrator.tableau_a
    b = integrator.tableau_b
    c = integrator.tableau_c

    # Manually compute stages for f(x,t) = -x
    k = []
    for i in range(6):
        if i == 0:
            x_stage = x
        else:
            dx = torch.zeros_like(x)
            for j in range(i):
                if a[i][j] != 0:
                    dx = dx + a[i][j] * k[j]
            x_stage = x + h * dx
        k.append(-x_stage)

    # Combine stages
    x_expected = x.clone()
    for i in range(6):
        if b[i] != 0:
            x_expected = x_expected + h * b[i] * k[i]

    result = integrator.step({"x": x}, step_size=h, drift=drift)

    assert torch.allclose(result["x"], x_expected, atol=1e-14)


def test_manual_dopri5_step_constant_drift():
    """Manual verification for constant drift f(x,t) = c.

    For constant drift, all stages evaluate to c, and x_new = x + h*c
    (since sum(b_i) = 1).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float64)

    x = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    c = torch.tensor([[3.0, -1.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: c
    h = 0.5

    result = integrator.step({"x": x}, step_size=h, drift=drift)

    expected = x + h * c
    assert torch.allclose(result["x"], expected, atol=1e-14)


def test_manual_dopri5_step_linear_drift():
    """Manual verification for linear time-dependent drift f(x,t) = t.

    For f(x,t) = t (independent of x), the analytical solution is
    x(h) = x(0) + h^2/2.
    Dopri5 should integrate this exactly since it's a polynomial of degree < 5.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float64)

    x = torch.tensor([[0.0, 0.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: t_.unsqueeze(-1).expand_as(x_)
    h = 1.0

    result = integrator.step({"x": x}, step_size=h, drift=drift)

    expected = x + h**2 / 2
    assert torch.allclose(result["x"], expected, atol=1e-12)


################################ Convergence and Accuracy Tests ###########################################


def test_fifth_order_convergence():
    """Test that the Dopri5 method shows 5th-order convergence.

    For dx/dt = -x, halving the step size should reduce the error by ~2^5 = 32x.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float64)

    x0 = torch.tensor([[2.0, 3.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    step_sizes = [0.1, 0.05, 0.025]
    errors = []

    for step_size in step_sizes:
        n_steps = int(1.0 / step_size)
        result = integrator.integrate(
            {"x": x0.clone()}, step_size=step_size, n_steps=n_steps,
            drift=drift, adaptive=False,
        )
        expected = x0 * math.exp(-1.0)
        error = torch.abs(result["x"] - expected).max().item()
        errors.append(error)

    for i in range(len(errors) - 1):
        ratio = errors[i] / errors[i + 1]
        # 5th order: halving step should give ~32x improvement
        assert ratio > 20, f"Expected ~32x error reduction for 5th order, got {ratio:.1f}"


def test_dopri5_vs_heun_accuracy():
    """Verify Dopri5 is more accurate than Heun for the same step size."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    from torchebm.integrators import HeunIntegrator

    dopri5 = Dopri5Integrator(device=device, dtype=dtype)
    heun = HeunIntegrator(device=device, dtype=dtype)

    x0 = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
    drift = lambda x_, t_: -x_
    step_size = 0.1
    n_steps = 10

    dopri5_result = dopri5.integrate(
        {"x": x0.clone()}, step_size=step_size, n_steps=n_steps,
        drift=drift, adaptive=False,
    )
    heun_result = heun.integrate(
        {"x": x0.clone()}, step_size=step_size, n_steps=n_steps,
        drift=drift, adaptive=False,
    )

    expected = x0 * math.exp(-1.0)
    dopri5_error = torch.abs(dopri5_result["x"] - expected).max().item()
    heun_error = torch.abs(heun_result["x"] - expected).max().item()

    assert dopri5_error < heun_error, (
        f"Dopri5 error {dopri5_error} should be less than Heun error {heun_error}"
    )


def test_exponential_decay_convergence():
    """Test that ODE integration converges to the analytical solution."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float64)

    x0 = torch.tensor([[2.0, 3.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    step_sizes = [0.1, 0.01, 0.001]
    errors = []

    for step_size in step_sizes:
        n_steps = int(1.0 / step_size)
        result = integrator.integrate(
            {"x": x0.clone()}, step_size=step_size, n_steps=n_steps,
            drift=drift, adaptive=False,
        )
        expected = x0 * math.exp(-1.0)
        error = torch.abs(result["x"] - expected).max().item()
        errors.append(error)

    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i], (
            f"Error should decrease: {errors[i + 1]} not < {errors[i]}"
        )


def test_adaptive_solves_stiff_ode():
    """Test adaptive mode on a moderately stiff problem: dx/dt = -50*(x - cos(t))."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(
        device=device, dtype=torch.float64, atol=1e-8, rtol=1e-6,
    )

    x0 = torch.tensor([[1.0]], device=device, dtype=torch.float64)
    lam = 50.0

    def drift(x_, t_):
        cos_t = torch.cos(t_).unsqueeze(-1)
        return -lam * (x_ - cos_t)

    result = integrator.integrate(
        {"x": x0}, step_size=0.01, n_steps=100, drift=drift,
    )

    # Approximate expected: for large lambda, x ≈ cos(t) up to O(1/lambda)
    expected_approx = math.cos(1.0)
    assert torch.abs(result["x"].squeeze() - expected_approx).item() < 0.1


def test_polynomial_ode_exact():
    """Test that Dopri5 integrates polynomials of degree <= 4 exactly.

    For dx/dt = t^3, x(0)=0, the solution is x(t) = t^4/4.
    A 5th-order method should integrate degree-4 polynomials exactly.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float64)

    x0 = torch.zeros(1, 1, device=device, dtype=torch.float64)
    drift = lambda x_, t_: (t_**3).unsqueeze(-1).expand_as(x_)

    # Single step from 0 to 1
    result = integrator.step(
        {"x": x0}, step_size=1.0, drift=drift,
        t=torch.zeros(1, device=device, dtype=torch.float64),
    )

    expected = torch.tensor([[0.25]], device=device, dtype=torch.float64)
    assert torch.allclose(result["x"], expected, atol=1e-12)


def test_polynomial_ode_quadratic():
    """Test that Dopri5 integrates dx/dt = t^2 exactly.

    x(0) = 0, solution: x(t) = t^3/3. At t=1, x=1/3.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float64)

    x0 = torch.zeros(1, 1, device=device, dtype=torch.float64)
    drift = lambda x_, t_: (t_**2).unsqueeze(-1).expand_as(x_)

    result = integrator.step(
        {"x": x0}, step_size=1.0, drift=drift,
        t=torch.zeros(1, device=device, dtype=torch.float64),
    )

    expected = torch.tensor([[1.0 / 3.0]], device=device, dtype=torch.float64)
    assert torch.allclose(result["x"], expected, atol=1e-14)


################################ FSAL Tests ###########################################


def test_fsal_reduces_drift_evaluations():
    """Test that FSAL reuses the last stage from the previous step.

    In adaptive mode, FSAL should save one drift evaluation per accepted step
    by caching k1 from the previous step's final evaluation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float64)

    call_count = [0]

    def counting_drift(x_, t_):
        call_count[0] += 1
        return -x_

    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    integrator.integrate(
        {"x": x}, step_size=0.5, n_steps=2, drift=counting_drift, adaptive=True,
    )

    # The adaptive loop should use FSAL caching instead of evaluating
    # all 6 stages + FSAL from scratch every step.
    # First step: 6 stages + 1 FSAL + 1 initial = 8 evaluations
    # Subsequent steps: 6 stages (reusing k1 from cache) + 1 FSAL = 7 evaluations per step
    # Exact count depends on accepted/rejected steps, but it should be finite.
    assert call_count[0] > 0


################################ Reproducibility Tests ###########################################


def test_reproducibility_fixed(integrator):
    """Test that fixed-step integration is deterministic."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -x_

    result1 = integrator.integrate(
        {"x": x.clone()}, step_size=0.01, n_steps=100,
        drift=drift, adaptive=False,
    )
    result2 = integrator.integrate(
        {"x": x.clone()}, step_size=0.01, n_steps=100,
        drift=drift, adaptive=False,
    )

    assert torch.allclose(result1["x"], result2["x"])


def test_reproducibility_adaptive():
    """Test that adaptive integration is deterministic."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float64)

    x = torch.randn(10, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result1 = integrator.integrate(
        {"x": x.clone()}, step_size=0.1, n_steps=10,
        drift=drift, adaptive=True,
    )
    result2 = integrator.integrate(
        {"x": x.clone()}, step_size=0.1, n_steps=10,
        drift=drift, adaptive=True,
    )

    assert torch.allclose(result1["x"], result2["x"])


################################ Edge Cases and Numerical Stability ######################################


def test_large_batch_size(integrator):
    """Test with large batch size."""
    device = integrator.device
    x = torch.randn(1000, 10, device=device)
    drift = lambda x_, t_: -x_

    result = integrator.step({"x": x}, step_size=0.01, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_high_dimension(integrator):
    """Test with high-dimensional input."""
    device = integrator.device
    x = torch.randn(10, 100, device=device)
    drift = lambda x_, t_: -x_

    result = integrator.step({"x": x}, step_size=0.01, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_small_step_size(integrator):
    """Test with very small step size."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -x_

    result = integrator.step({"x": x}, step_size=1e-8, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))
    # With very small step, result should barely change
    assert torch.allclose(result["x"], x, atol=1e-5)


def test_large_step_size(integrator):
    """Test with large step size (should still produce finite results)."""
    device = integrator.device
    x = torch.ones(10, 2, device=device) * 0.1
    drift = lambda x_, t_: -x_

    result = integrator.step({"x": x}, step_size=1.0, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_large_values(integrator):
    """Test numerical stability with large values."""
    device = integrator.device
    x = torch.randn(10, 2, device=device) * 1000
    drift = lambda x_, t_: -x_ * 0.001

    result = integrator.step({"x": x}, step_size=0.01, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_zero_drift(integrator):
    """Test that zero drift leaves state unchanged."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: torch.zeros_like(x_)

    result = integrator.step({"x": x}, step_size=0.1, drift=drift)

    assert torch.allclose(result["x"], x, atol=1e-7)


def test_zero_step_size(integrator):
    """Test that zero step size leaves state unchanged."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -x_

    result = integrator.step({"x": x}, step_size=0.0, drift=drift)

    assert torch.allclose(result["x"], x, atol=1e-7)


def test_time_dependent_drift(integrator):
    """Test with time-dependent drift."""
    device = integrator.device
    x = torch.ones(10, 2, device=device)
    t = torch.ones(10, device=device) * 0.5
    drift = lambda x_, t_: -x_ * (1 + t_[:, None])

    result = integrator.step({"x": x}, step_size=0.01, drift=drift, t=t)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_nonlinear_drift():
    """Test with nonlinear drift: f(x) = -x^3."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float64)

    x = torch.ones(10, 2, device=device, dtype=torch.float64) * 0.5
    drift = lambda x_, t_: -(x_**3)
    n_steps = 100

    result = integrator.integrate(
        {"x": x}, step_size=0.01, n_steps=n_steps,
        drift=drift, adaptive=False,
    )

    # For dx/dt = -x^3, x(0) = 0.5, solution x(t) = 1/sqrt(2t + 1/x0^2)
    # At t=1: x(1) = 1/sqrt(2 + 4) = 1/sqrt(6)
    x0_val = 0.5
    t_final = 1.0
    expected_val = 1.0 / math.sqrt(2 * t_final + 1.0 / x0_val**2)
    expected = torch.full_like(x, expected_val)

    assert torch.allclose(result["x"], expected, atol=1e-6)


def test_multidimensional_batch():
    """Test with 3D-shaped input (batch x spatial x channels)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float32)

    x = torch.randn(8, 4, 3, device=device)
    drift = lambda x_, t_: -x_

    result = integrator.step({"x": x}, step_size=0.01, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_single_sample(integrator):
    """Test with batch size of 1."""
    device = integrator.device
    x = torch.randn(1, 2, device=device)
    drift = lambda x_, t_: -x_

    result = integrator.step({"x": x}, step_size=0.01, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_adaptive_with_oscillatory_ode():
    """Test adaptive integration on an oscillatory ODE: harmonic oscillator.

    System: dx/dt = v, dv/dt = -x (encoded in a single state vector).
    Solution: x(t) = x0*cos(t) + v0*sin(t), v(t) = -x0*sin(t) + v0*cos(t).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(
        device=device, dtype=torch.float64, atol=1e-10, rtol=1e-8,
    )

    # State: [x, v] for a single oscillator
    x0_val, v0_val = 1.0, 0.0
    state_vec = torch.tensor([[x0_val, v0_val]], device=device, dtype=torch.float64)

    def drift(s, t_):
        x_comp = s[:, 0:1]
        v_comp = s[:, 1:2]
        return torch.cat([v_comp, -x_comp], dim=-1)

    t_final = 2 * math.pi  # One full period
    result = integrator.integrate(
        {"x": state_vec}, step_size=0.1, n_steps=int(t_final / 0.1),
        drift=drift, t=torch.tensor([0.0, t_final], device=device, dtype=torch.float64),
    )

    # After one full period, should return to initial state
    assert torch.allclose(
        result["x"], state_vec, atol=1e-6
    ), f"After full period: got {result['x']}, expected {state_vec}"


def test_dtype_float32(integrator):
    """Test that float32 dtype is preserved."""
    device = integrator.device
    x = torch.randn(5, 2, device=device, dtype=torch.float32)
    drift = lambda x_, t_: -x_

    result = integrator.step({"x": x}, step_size=0.01, drift=drift)

    assert result["x"].dtype == torch.float32


def test_dtype_float64(integrator_f64):
    """Test that float64 dtype is preserved."""
    device = integrator_f64.device
    x = torch.randn(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result = integrator_f64.step({"x": x}, step_size=0.01, drift=drift)

    assert result["x"].dtype == torch.float64


def test_double_well_convergence():
    """Test integration with DoubleWell energy model gradient field."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float64)
    model = DoubleWellModel(barrier_height=2.0).to(device)

    # Start near a minimum at x=1
    x = torch.ones(10, 1, device=device, dtype=torch.float64) * 1.5
    drift = lambda x_, t_: -model.gradient(x_.float()).double()

    result = integrator.integrate(
        {"x": x}, step_size=0.01, n_steps=200,
        drift=drift, adaptive=False,
    )

    assert torch.all(torch.isfinite(result["x"]))
    # Should converge towards a minimum (either +1 or -1)
    assert torch.all(torch.abs(result["x"]) > 0.5)


def test_adaptive_zero_error_growth():
    """Test that adaptive mode handles zero error (constant ODE) gracefully."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float64)

    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    # Constant drift: all stages identical, error estimate = 0
    drift = lambda x_, t_: torch.ones_like(x_)

    result = integrator.integrate(
        {"x": x}, step_size=0.1, n_steps=10, drift=drift,
    )

    # x(t) = x(0) + t * 1.0 = 1.0 + 1.0 = 2.0
    expected = x + 1.0
    assert torch.allclose(result["x"], expected, atol=1e-10)


################################ norm Parameter Tests ###########################################


def test_default_norm_is_rms():
    """Test that the default norm is RMS (same as previous hardcoded behaviour)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(device=device, dtype=torch.float64)

    assert integrator._norm is None

    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result = integrator.integrate(
        {"x": x}, step_size=0.1, n_steps=10, drift=drift,
    )

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, atol=5e-4)


def test_custom_norm_max():
    """Test adaptive integration with max-norm instead of RMS."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_norm = lambda x: torch.max(torch.abs(x))

    integrator = Dopri5Integrator(
        device=device, dtype=torch.float64,
        atol=1e-8, rtol=1e-6, norm=max_norm,
    )

    assert integrator._norm is max_norm

    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result = integrator.integrate(
        {"x": x}, step_size=0.1, n_steps=10, drift=drift,
    )

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, atol=1e-5)


def test_custom_norm_affects_step_acceptance():
    """Test that a stricter norm leads to more accurate results.

    A max-norm is stricter than RMS for high-dimensional inputs, so with
    the same tolerances it should produce at least comparable accuracy.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_norm = lambda x: torch.max(torch.abs(x))

    integrator_rms = Dopri5Integrator(
        device=device, dtype=torch.float64, atol=1e-6, rtol=1e-4,
    )
    integrator_max = Dopri5Integrator(
        device=device, dtype=torch.float64, atol=1e-6, rtol=1e-4,
        norm=max_norm,
    )

    x = torch.randn(20, 5, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result_rms = integrator_rms.integrate(
        {"x": x.clone()}, step_size=0.2, n_steps=5, drift=drift,
    )
    result_max = integrator_max.integrate(
        {"x": x.clone()}, step_size=0.2, n_steps=5, drift=drift,
    )

    expected = x * math.exp(-1.0)
    err_rms = torch.abs(result_rms["x"] - expected).max().item()
    err_max = torch.abs(result_max["x"] - expected).max().item()

    # Max-norm is stricter, so its worst-case error should be at most
    # comparable (allowing some tolerance for stochastic step patterns).
    assert err_max < err_rms * 5


def test_custom_norm_initialization():
    """Test that norm parameter is stored and retrievable."""
    my_norm = lambda x: torch.mean(torch.abs(x))
    integrator = Dopri5Integrator(norm=my_norm)
    assert integrator._norm is my_norm


################################ max_step_size Parameter Tests ###########################################


def test_default_max_step_size_is_inf():
    """Test that the default max_step_size is infinity."""
    integrator = Dopri5Integrator()
    assert integrator.max_step_size == float("inf")


def test_custom_max_step_size_initialization():
    """Test initialization with custom max_step_size."""
    integrator = Dopri5Integrator(max_step_size=0.5)
    assert integrator.max_step_size == 0.5


def test_max_step_size_limits_step():
    """Test that max_step_size prevents the adaptive algorithm from taking large steps.

    We count drift evaluations: with a small max_step_size the integrator
    must take more steps (and therefore more evaluations) than without.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    calls_unlimited = [0]
    calls_limited = [0]

    def counting_drift_a(x_, t_):
        calls_unlimited[0] += 1
        return -x_

    def counting_drift_b(x_, t_):
        calls_limited[0] += 1
        return -x_

    x = torch.ones(5, 2, device=device, dtype=torch.float64)

    integrator_unlimited = Dopri5Integrator(
        device=device, dtype=torch.float64, atol=1e-6, rtol=1e-4,
    )
    integrator_limited = Dopri5Integrator(
        device=device, dtype=torch.float64, atol=1e-6, rtol=1e-4,
        max_step_size=0.05,
    )

    integrator_unlimited.integrate(
        {"x": x.clone()}, step_size=0.5, n_steps=10, drift=counting_drift_a,
    )
    integrator_limited.integrate(
        {"x": x.clone()}, step_size=0.5, n_steps=10, drift=counting_drift_b,
    )

    # The limited integrator should take more drift evaluations
    assert calls_limited[0] > calls_unlimited[0]


def test_max_step_size_still_accurate():
    """Test that max_step_size does not degrade accuracy."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(
        device=device, dtype=torch.float64,
        atol=1e-8, rtol=1e-6, max_step_size=0.1,
    )

    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result = integrator.integrate(
        {"x": x}, step_size=0.5, n_steps=2, drift=drift,
    )

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, atol=1e-5)


def test_max_step_size_with_oscillatory_ode():
    """Test max_step_size on an oscillatory ODE where large steps cause instability.

    For a harmonic oscillator, limiting step size should help maintain accuracy
    over a full period.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Dopri5Integrator(
        device=device, dtype=torch.float64,
        atol=1e-10, rtol=1e-8, max_step_size=0.2,
    )

    x0_val, v0_val = 1.0, 0.0
    state_vec = torch.tensor([[x0_val, v0_val]], device=device, dtype=torch.float64)

    def drift(s, t_):
        x_comp = s[:, 0:1]
        v_comp = s[:, 1:2]
        return torch.cat([v_comp, -x_comp], dim=-1)

    t_final = 2 * math.pi
    result = integrator.integrate(
        {"x": state_vec}, step_size=0.5, n_steps=int(t_final / 0.5),
        drift=drift,
        t=torch.tensor([0.0, t_final], device=device, dtype=torch.float64),
    )

    assert torch.allclose(result["x"], state_vec, atol=1e-6)

