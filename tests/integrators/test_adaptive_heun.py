"""Tests for AdaptiveHeunIntegrator."""

import math

import pytest
import torch

from torchebm.core import GaussianModel, DoubleWellModel
from torchebm.integrators import AdaptiveHeunIntegrator
from tests.conftest import requires_cuda


################################ Test Fixtures ###########################################


@pytest.fixture
def integrator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return AdaptiveHeunIntegrator(device=device, dtype=torch.float32)


@pytest.fixture
def integrator_f64():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return AdaptiveHeunIntegrator(device=device, dtype=torch.float64)


@pytest.fixture
def gaussian_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return GaussianModel(
        mean=torch.zeros(2, device=device), cov=torch.eye(2, device=device)
    ).to(device)


################################ Initialization Tests ###########################################


def test_initialization():
    integrator = AdaptiveHeunIntegrator()
    assert isinstance(integrator, AdaptiveHeunIntegrator)
    assert integrator.atol == 1e-6
    assert integrator.rtol == 1e-3
    assert integrator.max_steps == 10_000
    assert integrator.safety == 0.9
    assert integrator.min_factor == 0.2
    assert integrator.max_factor == 10.0
    assert integrator.max_step_size == float("inf")
    assert integrator._norm is None


def test_initialization_with_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(device=device, dtype=torch.float32)
    assert integrator.device == torch.device(device)
    assert integrator.dtype == torch.float32


def test_custom_tolerances():
    integrator = AdaptiveHeunIntegrator(
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
def test_cuda():
    integrator = AdaptiveHeunIntegrator(device="cuda", dtype=torch.float32)
    assert integrator.device == torch.device("cuda")


################################ Butcher Tableau Tests ###########################################


def test_tableau_a_values():
    integrator = AdaptiveHeunIntegrator()
    a = integrator.tableau_a

    assert len(a) == 2
    assert a[0] == ()
    assert len(a[1]) == 1
    assert a[1][0] == 1.0


def test_tableau_b_values():
    integrator = AdaptiveHeunIntegrator()
    b = integrator.tableau_b

    assert len(b) == 2
    assert b[0] == 0.5
    assert b[1] == 0.5


def test_tableau_c_values():
    integrator = AdaptiveHeunIntegrator()
    c = integrator.tableau_c

    assert len(c) == 2
    assert c[0] == 0.0
    assert c[1] == 1.0


def test_error_weights_values():
    r"""e_i = b_i - b_hat_i where b_hat = (1, 0) is Euler."""
    integrator = AdaptiveHeunIntegrator()
    e = integrator.error_weights

    assert len(e) == 2
    assert math.isclose(e[0], 0.5)
    assert math.isclose(e[1], -0.5)


def test_order():
    integrator = AdaptiveHeunIntegrator()
    assert integrator.order == 2


def test_fsal_property():
    integrator = AdaptiveHeunIntegrator()
    assert integrator.fsal is False


def test_n_stages():
    integrator = AdaptiveHeunIntegrator()
    assert integrator.n_stages == 2


def test_tableau_consistency_b_sum():
    integrator = AdaptiveHeunIntegrator()
    b = integrator.tableau_b
    assert math.isclose(sum(b), 1.0, abs_tol=1e-14)


def test_tableau_consistency_row_sums():
    integrator = AdaptiveHeunIntegrator()
    a = integrator.tableau_a
    c = integrator.tableau_c

    for i in range(len(a)):
        row_sum = sum(a[i]) if a[i] else 0.0
        assert math.isclose(row_sum, c[i], abs_tol=1e-14), (
            f"Row {i}: sum(a[{i}]) = {row_sum} != c[{i}] = {c[i]}"
        )


def test_error_weights_are_b_minus_bhat():
    r"""Verify e_i matches diffrax convention: b_error = [0.5, -0.5].

    The error_weights from AdaptiveHeun follow the diffrax convention
    where b_error = b_hat - b (note: reversed relative to some references).
    b_hat = (1, 0) (Euler), b = (0.5, 0.5) (Heun)
    b_error = (1-0.5, 0-0.5) = (0.5, -0.5)
    """
    integrator = AdaptiveHeunIntegrator()
    e = integrator.error_weights

    assert math.isclose(e[0], 0.5, abs_tol=1e-14)
    assert math.isclose(e[1], -0.5, abs_tol=1e-14)


################################ Step Method Tests ###########################################


def test_step_ode(integrator):
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -x_

    result = integrator.step({"x": x}, step_size=0.01, drift=drift)

    assert "x" in result
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_step_with_explicit_time(integrator):
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    t = torch.ones(10, device=device) * 0.5
    drift = lambda x_, t_: -x_ * (1 + t_[:, None])

    result = integrator.step({"x": x}, step_size=0.01, drift=drift, t=t)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_step_with_model_gradient(integrator, gaussian_model):
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -gaussian_model.gradient(x_)

    result = integrator.step({"x": x}, step_size=0.01, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_step_requires_drift(integrator):
    device = integrator.device
    x = torch.randn(10, 2, device=device)

    with pytest.raises(ValueError, match="drift must be provided explicitly"):
        integrator.step({"x": x}, step_size=0.01)


def test_step_default_time_is_zero(integrator):
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
    device = integrator.device
    x = torch.randn(10, 2, device=device)

    result = integrator.step({"x": x}, step_size=0.01, drift=lambda x_, t_: -x_)
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_step_tensor_step_size(integrator):
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    step_size = torch.tensor(0.01, device=device)

    result = integrator.step({"x": x}, step_size=step_size, drift=lambda x_, t_: -x_)
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


################################ Integrate Method Tests (Fixed-step) ##############################


def test_integrate_fixed_step(integrator):
    device = integrator.device
    x = torch.ones(10, 2, device=device)
    drift = lambda x_, t_: -x_

    result = integrator.integrate(
        {"x": x}, step_size=0.02, n_steps=50,
        drift=drift, adaptive=False,
    )

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, rtol=0.01)


def test_integrate_fixed_step_with_time_grid(integrator):
    device = integrator.device
    x = torch.ones(10, 2, device=device)
    drift = lambda x_, t_: -x_
    t = torch.linspace(0, 1, 51, device=device)

    result = integrator.integrate(
        {"x": x}, step_size=0.02, n_steps=50,
        drift=drift, t=t, adaptive=False,
    )

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, rtol=0.01)


def test_integrate_fixed_step_with_model(integrator, gaussian_model):
    device = integrator.device
    x = torch.randn(10, 2, device=device) * 3
    drift = lambda x_, t_: -gaussian_model.gradient(x_)

    result = integrator.integrate(
        {"x": x}, step_size=0.01, n_steps=100,
        drift=drift, adaptive=False,
    )

    initial_dist = torch.norm(x, dim=-1).mean()
    final_dist = torch.norm(result["x"], dim=-1).mean()
    assert final_dist < initial_dist


def test_integrate_invalid_n_steps(integrator):
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


################################ Integrate Method Tests (Adaptive) #################################


def test_integrate_adaptive_default():
    """Adaptive should be the default since error_weights is defined."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(device=device, dtype=torch.float64)

    x = torch.ones(10, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result = integrator.integrate(
        {"x": x}, step_size=0.1, n_steps=10, drift=drift,
    )

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, atol=5e-3)


def test_integrate_adaptive_explicit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(device=device, dtype=torch.float64)

    x = torch.ones(10, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result = integrator.integrate(
        {"x": x}, step_size=0.1, n_steps=10,
        drift=drift, adaptive=True,
    )

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, atol=5e-3)


def test_adaptive_with_custom_time_grid():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(device=device, dtype=torch.float64)

    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_
    t = torch.tensor([0.0, 2.0], device=device, dtype=torch.float64)

    result = integrator.integrate(
        {"x": x}, step_size=0.5, n_steps=4,
        drift=drift, t=t, adaptive=True,
    )

    expected = x * math.exp(-2.0)
    assert torch.allclose(result["x"], expected, atol=1e-3)


def test_adaptive_tight_tolerance():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(
        device=device, dtype=torch.float64, atol=1e-10, rtol=1e-8,
    )

    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result = integrator.integrate(
        {"x": x}, step_size=0.1, n_steps=10, drift=drift,
    )

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, atol=1e-6)


def test_adaptive_loose_tolerance():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(device=device, dtype=torch.float64)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(
        device=device, dtype=torch.float64, atol=1e-10, rtol=1e-8,
    )

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


def test_manual_heun_step_linear_decay():
    r"""Manual verification of a single Heun step for f(x,t)=-x.

    For f(x,t) = -x with step h:
    k1 = -x
    k2 = -(x + h*k1) = -(x - hx) = -x(1 - h)
    x_new = x + h/2*(k1 + k2)
          = x + h/2*(-x + -x(1-h))
          = x + h/2*(-x)(2 - h)
          = x - hx(2-h)/2
          = x(1 - h + h²/2)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(device=device, dtype=torch.float64)

    x = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_
    h = 0.1

    a = integrator.tableau_a
    b = integrator.tableau_b

    k1 = -x
    dx_stage = a[1][0] * k1
    x_stage2 = x + h * dx_stage
    k2 = -x_stage2

    x_expected = x + h * (b[0] * k1 + b[1] * k2)

    result = integrator.step({"x": x}, step_size=h, drift=drift)

    assert torch.allclose(result["x"], x_expected, atol=1e-14)

    # Also verify the analytical formula: x(1 - h + h²/2)
    analytical = x * (1 - h + h**2 / 2)
    assert torch.allclose(result["x"], analytical, atol=1e-14)


def test_manual_heun_step_constant_drift():
    """For constant drift f(x,t)=c, all stages equal c, x_new = x + h*c."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(device=device, dtype=torch.float64)

    x = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    c = torch.tensor([[3.0, -1.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: c
    h = 0.5

    result = integrator.step({"x": x}, step_size=h, drift=drift)

    expected = x + h * c
    assert torch.allclose(result["x"], expected, atol=1e-14)


def test_manual_heun_step_linear_time_drift():
    r"""For f(x,t)=t, x(0)=0: x(h)=h²/2. Heun integrates linear t exactly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(device=device, dtype=torch.float64)

    x = torch.tensor([[0.0, 0.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: t_.unsqueeze(-1).expand_as(x_)
    h = 1.0

    result = integrator.step(
        {"x": x}, step_size=h, drift=drift,
        t=torch.zeros(1, device=device, dtype=torch.float64),
    )

    expected = x + h**2 / 2
    assert torch.allclose(result["x"], expected, atol=1e-12)


def test_manual_heun_quadratic_time_not_exact():
    r"""For f(x,t)=t², x(0)=0, solution x(1)=1/3. Heun is order 2 so
    it should NOT integrate t² exactly in a single step.

    k1 = 0² = 0
    k2 = 1² = 1
    x_new = 0 + 1*(0.5*0 + 0.5*1) = 0.5
    Exact = 1/3 ≈ 0.3333
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(device=device, dtype=torch.float64)

    x0 = torch.zeros(1, 1, device=device, dtype=torch.float64)
    drift = lambda x_, t_: (t_**2).unsqueeze(-1).expand_as(x_)

    result = integrator.step(
        {"x": x0}, step_size=1.0, drift=drift,
        t=torch.zeros(1, device=device, dtype=torch.float64),
    )

    exact = torch.tensor([[1.0 / 3.0]], device=device, dtype=torch.float64)
    assert not torch.allclose(result["x"], exact, atol=1e-10)

    # Should get 0.5 from hand calculation
    hand = torch.tensor([[0.5]], device=device, dtype=torch.float64)
    assert torch.allclose(result["x"], hand, atol=1e-14)


def test_heun_exponential_decay_numerical():
    r"""Numerical check: Heun approximation to e^{-h}.

    For f(x,t)=-x, h=0.1, x(0)=1:
    k1 = -1
    k2 = -(1 + 0.1*(-1)) = -0.9
    x_new = 1 + 0.1/2*(-1 + (-0.9))
          = 1 + 0.05*(-1.9)
          = 1 - 0.095
          = 0.905
    True exp(-0.1) = 0.90483741803...
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(device=device, dtype=torch.float64)

    x = torch.tensor([[1.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_
    h = 0.1

    result = integrator.step({"x": x}, step_size=h, drift=drift)

    expected_hand = 1.0 + (h / 2) * (-1.0 + (-0.9))
    assert math.isclose(result["x"].item(), expected_hand, rel_tol=1e-12)
    assert math.isclose(expected_hand, 0.905, rel_tol=1e-12)

    # Close to exp(-0.1) but not exact (order 2)
    assert abs(result["x"].item() - math.exp(-0.1)) < 0.001


def test_manual_error_estimate():
    r"""Verify the error estimate for a single step.

    For f(x,t)=-x, h=0.1, x=1:
    k1 = -1, k2 = -0.9
    error = h * (e[0]*k1 + e[1]*k2) = 0.1 * (0.5*(-1) + (-0.5)*(-0.9))
          = 0.1 * (-0.5 + 0.45) = 0.1 * (-0.05) = -0.005
    This is the difference between Heun (0.905) and Euler (0.9): 0.005.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(device=device, dtype=torch.float64)

    x = torch.tensor([[1.0]], device=device, dtype=torch.float64)
    drift_fn = lambda x_, t_: -x_
    h = torch.tensor(0.1, device=device, dtype=torch.float64)
    t = torch.zeros(1, device=device, dtype=torch.float64)

    k = integrator._evaluate_stages(x, t, h, drift_fn)

    e = integrator.error_weights
    err_vec = h * (e[0] * k[0] + e[1] * k[1])

    expected_err = 0.1 * (0.5 * (-1.0) + (-0.5) * (-0.9))
    assert math.isclose(err_vec.item(), expected_err, abs_tol=1e-14)

    # Error should equal Heun - Euler
    heun_val = 1.0 + 0.1 * (0.5 * (-1.0) + 0.5 * (-0.9))
    euler_val = 1.0 + 0.1 * (-1.0)
    assert math.isclose(abs(err_vec.item()), abs(heun_val - euler_val), abs_tol=1e-14)


################################ Convergence and Accuracy Tests ####################################


def test_second_order_convergence():
    """Halving step size should reduce error by ~2² = 4x."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(device=device, dtype=torch.float64)

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
        assert ratio > 3.0, f"Expected ~4x error reduction for 2nd order, got {ratio:.1f}"


def test_adaptive_heun_vs_rk4_accuracy():
    """RK4 (order 4) should be more accurate than AdaptiveHeun (order 2) fixed-step."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    from torchebm.integrators import RK4Integrator

    heun = AdaptiveHeunIntegrator(device=device, dtype=dtype)
    rk4 = RK4Integrator(device=device, dtype=dtype)

    x0 = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
    drift = lambda x_, t_: -x_
    step_size = 0.1
    n_steps = 10

    heun_result = heun.integrate(
        {"x": x0.clone()}, step_size=step_size, n_steps=n_steps,
        drift=drift, adaptive=False,
    )
    rk4_result = rk4.integrate(
        {"x": x0.clone()}, step_size=step_size, n_steps=n_steps,
        drift=drift, adaptive=False,
    )

    expected = x0 * math.exp(-1.0)
    heun_error = torch.abs(heun_result["x"] - expected).max().item()
    rk4_error = torch.abs(rk4_result["x"] - expected).max().item()

    assert rk4_error < heun_error


def test_exponential_decay_convergence():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(device=device, dtype=torch.float64)

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
        assert errors[i + 1] < errors[i]


def test_nonlinear_drift_cubic():
    r"""dx/dt = -x³, x(0)=0.5, solution x(t)=1/√(2t+1/x₀²). At t=1: x=1/√6."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(device=device, dtype=torch.float64)

    x = torch.ones(10, 2, device=device, dtype=torch.float64) * 0.5
    drift = lambda x_, t_: -(x_**3)

    result = integrator.integrate(
        {"x": x}, step_size=0.001, n_steps=1000,
        drift=drift, adaptive=False,
    )

    x0_val = 0.5
    expected_val = 1.0 / math.sqrt(2 * 1.0 + 1.0 / x0_val**2)
    expected = torch.full_like(x, expected_val)

    assert torch.allclose(result["x"], expected, atol=1e-4)


def test_harmonic_oscillator():
    """dx/dt=v, dv/dt=-x. After 2π the state returns to start."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(
        device=device, dtype=torch.float64, atol=1e-8, rtol=1e-6,
        max_steps=50_000,
    )

    state_vec = torch.tensor([[1.0, 0.0]], device=device, dtype=torch.float64)

    def drift(s, t_):
        return torch.cat([s[:, 1:2], -s[:, 0:1]], dim=-1)

    t_final = 2 * math.pi
    result = integrator.integrate(
        {"x": state_vec}, step_size=0.01, n_steps=int(t_final / 0.01),
        drift=drift,
        t=torch.tensor([0.0, t_final], device=device, dtype=torch.float64),
    )

    assert torch.allclose(result["x"], state_vec, atol=1e-4)


################################ Reproducibility Tests #############################################


def test_reproducibility_fixed(integrator):
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(device=device, dtype=torch.float64)

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


################################ Edge Cases ########################################################


def test_large_batch_size(integrator):
    device = integrator.device
    x = torch.randn(1000, 10, device=device)
    drift = lambda x_, t_: -x_

    result = integrator.step({"x": x}, step_size=0.01, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_high_dimension(integrator):
    device = integrator.device
    x = torch.randn(10, 100, device=device)
    drift = lambda x_, t_: -x_

    result = integrator.step({"x": x}, step_size=0.01, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_small_step_size(integrator):
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -x_

    result = integrator.step({"x": x}, step_size=1e-8, drift=drift)

    assert torch.all(torch.isfinite(result["x"]))
    assert torch.allclose(result["x"], x, atol=1e-5)


def test_large_step_size(integrator):
    device = integrator.device
    x = torch.ones(10, 2, device=device) * 0.1
    drift = lambda x_, t_: -x_

    result = integrator.step({"x": x}, step_size=1.0, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_large_values(integrator):
    device = integrator.device
    x = torch.randn(10, 2, device=device) * 1000
    drift = lambda x_, t_: -x_ * 0.001

    result = integrator.step({"x": x}, step_size=0.01, drift=drift)

    assert torch.all(torch.isfinite(result["x"]))


def test_zero_drift(integrator):
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: torch.zeros_like(x_)

    result = integrator.step({"x": x}, step_size=0.1, drift=drift)

    assert torch.allclose(result["x"], x, atol=1e-7)


def test_zero_step_size(integrator):
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    drift = lambda x_, t_: -x_

    result = integrator.step({"x": x}, step_size=0.0, drift=drift)

    assert torch.allclose(result["x"], x, atol=1e-7)


def test_time_dependent_drift(integrator):
    device = integrator.device
    x = torch.ones(10, 2, device=device)
    t = torch.ones(10, device=device) * 0.5
    drift = lambda x_, t_: -x_ * (1 + t_[:, None])

    result = integrator.step({"x": x}, step_size=0.01, drift=drift, t=t)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_multidimensional_batch(integrator):
    device = integrator.device
    x = torch.randn(8, 4, 3, device=device)
    drift = lambda x_, t_: -x_

    result = integrator.step({"x": x}, step_size=0.01, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_single_sample(integrator):
    device = integrator.device
    x = torch.randn(1, 2, device=device)
    drift = lambda x_, t_: -x_

    result = integrator.step({"x": x}, step_size=0.01, drift=drift)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_dtype_float32(integrator):
    device = integrator.device
    x = torch.randn(5, 2, device=device, dtype=torch.float32)
    drift = lambda x_, t_: -x_

    result = integrator.step({"x": x}, step_size=0.01, drift=drift)
    assert result["x"].dtype == torch.float32


def test_dtype_float64(integrator_f64):
    device = integrator_f64.device
    x = torch.randn(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result = integrator_f64.step({"x": x}, step_size=0.01, drift=drift)
    assert result["x"].dtype == torch.float64


def test_double_well_convergence():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(device=device, dtype=torch.float64)
    model = DoubleWellModel(barrier_height=2.0).to(device)

    x = torch.ones(10, 1, device=device, dtype=torch.float64) * 1.5
    drift = lambda x_, t_: -model.gradient(x_.float()).double()

    result = integrator.integrate(
        {"x": x}, step_size=0.01, n_steps=200,
        drift=drift, adaptive=False,
    )

    assert torch.all(torch.isfinite(result["x"]))
    assert torch.all(torch.abs(result["x"]) > 0.5)


def test_stiff_ode():
    """Moderately stiff: dx/dt = -50*(x - cos(t))."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(
        device=device, dtype=torch.float64, atol=1e-6, rtol=1e-4,
    )

    x0 = torch.tensor([[1.0]], device=device, dtype=torch.float64)
    lam = 50.0

    def drift(x_, t_):
        cos_t = torch.cos(t_).unsqueeze(-1)
        return -lam * (x_ - cos_t)

    result = integrator.integrate(
        {"x": x0}, step_size=0.001, n_steps=1000, drift=drift,
    )

    expected_approx = math.cos(1.0)
    assert torch.abs(result["x"].squeeze() - expected_approx).item() < 0.1


def test_adaptive_zero_error_growth():
    """Constant drift: error estimate = 0, step should grow to max."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(device=device, dtype=torch.float64)

    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: torch.ones_like(x_)

    result = integrator.integrate(
        {"x": x}, step_size=0.1, n_steps=10, drift=drift,
    )

    expected = x + 1.0
    assert torch.allclose(result["x"], expected, atol=1e-10)


################################ norm & max_step_size Tests ########################################


def test_default_norm_is_rms():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(device=device, dtype=torch.float64)
    assert integrator._norm is None

    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result = integrator.integrate(
        {"x": x}, step_size=0.1, n_steps=10, drift=drift,
    )

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, atol=5e-3)


def test_custom_norm_max():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_norm = lambda x: torch.max(torch.abs(x))

    integrator = AdaptiveHeunIntegrator(
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
    assert torch.allclose(result["x"], expected, atol=1e-4)


def test_default_max_step_size_is_inf():
    integrator = AdaptiveHeunIntegrator()
    assert integrator.max_step_size == float("inf")


def test_custom_max_step_size():
    integrator = AdaptiveHeunIntegrator(max_step_size=0.5)
    assert integrator.max_step_size == 0.5


def test_max_step_size_limits_step():
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

    integrator_unlimited = AdaptiveHeunIntegrator(
        device=device, dtype=torch.float64, atol=1e-6, rtol=1e-4,
    )
    integrator_limited = AdaptiveHeunIntegrator(
        device=device, dtype=torch.float64, atol=1e-6, rtol=1e-4,
        max_step_size=0.01,
    )

    integrator_unlimited.integrate(
        {"x": x.clone()}, step_size=0.5, n_steps=10, drift=counting_drift_a,
    )
    integrator_limited.integrate(
        {"x": x.clone()}, step_size=0.5, n_steps=10, drift=counting_drift_b,
    )

    assert calls_limited[0] > calls_unlimited[0]


def test_max_step_size_still_accurate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = AdaptiveHeunIntegrator(
        device=device, dtype=torch.float64,
        atol=1e-8, rtol=1e-6, max_step_size=0.1,
    )

    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result = integrator.integrate(
        {"x": x}, step_size=0.5, n_steps=2, drift=drift,
    )

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, atol=1e-4)
