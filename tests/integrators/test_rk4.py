"""Tests for RK4Integrator."""

import math

import pytest
import torch

from torchebm.core import GaussianModel, DoubleWellModel
from torchebm.integrators import RK4Integrator
from tests.conftest import requires_cuda


################################ Test Fixtures ###########################################


@pytest.fixture
def integrator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return RK4Integrator(device=device, dtype=torch.float32)


@pytest.fixture
def integrator_f64():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return RK4Integrator(device=device, dtype=torch.float64)


@pytest.fixture
def gaussian_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return GaussianModel(
        mean=torch.zeros(2, device=device), cov=torch.eye(2, device=device)
    ).to(device)


################################ Initialization Tests ###########################################


def test_rk4_initialization():
    integrator = RK4Integrator()
    assert isinstance(integrator, RK4Integrator)


def test_rk4_initialization_with_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = RK4Integrator(device=device, dtype=torch.float32)
    assert integrator.device == torch.device(device)
    assert integrator.dtype == torch.float32


def test_rk4_no_adaptive_params():
    """RK4 is fixed-step only — error_weights and order should be None."""
    integrator = RK4Integrator()
    assert integrator.error_weights is None
    assert integrator.order is None
    assert integrator.fsal is False


@requires_cuda
def test_rk4_cuda():
    integrator = RK4Integrator(device="cuda", dtype=torch.float32)
    assert integrator.device == torch.device("cuda")


################################ Butcher Tableau Tests ###########################################


def test_tableau_a_values():
    integrator = RK4Integrator()
    a = integrator.tableau_a

    assert len(a) == 4
    assert a[0] == ()
    assert len(a[1]) == 1
    assert math.isclose(a[1][0], 0.5)
    assert len(a[2]) == 2
    assert a[2][0] == 0.0
    assert math.isclose(a[2][1], 0.5)
    assert len(a[3]) == 3
    assert a[3][0] == 0.0
    assert a[3][1] == 0.0
    assert math.isclose(a[3][2], 1.0)


def test_tableau_b_values():
    integrator = RK4Integrator()
    b = integrator.tableau_b

    assert len(b) == 4
    assert math.isclose(b[0], 1 / 6)
    assert math.isclose(b[1], 1 / 3)
    assert math.isclose(b[2], 1 / 3)
    assert math.isclose(b[3], 1 / 6)


def test_tableau_c_values():
    integrator = RK4Integrator()
    c = integrator.tableau_c

    assert len(c) == 4
    assert c[0] == 0.0
    assert math.isclose(c[1], 0.5)
    assert math.isclose(c[2], 0.5)
    assert math.isclose(c[3], 1.0)


def test_n_stages():
    integrator = RK4Integrator()
    assert integrator.n_stages == 4


def test_tableau_consistency_b_sum():
    integrator = RK4Integrator()
    b = integrator.tableau_b
    assert math.isclose(sum(b), 1.0, abs_tol=1e-14)


def test_tableau_consistency_row_sums():
    integrator = RK4Integrator()
    a = integrator.tableau_a
    c = integrator.tableau_c

    for i in range(len(a)):
        row_sum = sum(a[i]) if a[i] else 0.0
        assert math.isclose(row_sum, c[i], abs_tol=1e-14), (
            f"Row {i}: sum(a[{i}]) = {row_sum} != c[{i}] = {c[i]}"
        )


################################ Step Method Tests ###########################################


def test_step_ode(integrator):
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    state = {"x": x}
    drift = lambda x_, t_: -x_

    result = integrator.step(state, step_size=0.01, drift=drift)

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


################################ Integrate Method Tests ###########################################


def test_integrate_ode(integrator):
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
    assert torch.allclose(result["x"], expected, rtol=0.001)


def test_integrate_with_time_grid(integrator):
    device = integrator.device
    x = torch.ones(10, 2, device=device)
    drift = lambda x_, t_: -x_
    t = torch.linspace(0, 1, 51, device=device)

    result = integrator.integrate(
        {"x": x}, step_size=0.02, n_steps=50,
        drift=drift, t=t, adaptive=False,
    )

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, rtol=0.001)


def test_integrate_with_model(integrator, gaussian_model):
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


def test_single_step_integration_matches_step(integrator_f64):
    """1-step integrate should match a single step() call."""
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


def test_rk4_no_adaptive_mode():
    """RK4 has no error_weights, so adaptive=True should fall back or raise."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = RK4Integrator(device=device, dtype=torch.float64)
    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    # Without error_weights, adaptive mode is not available.
    # Depending on the base class behaviour it may raise or just use fixed-step.
    # Test that fixed-step is the reliable path.
    result = integrator.integrate(
        {"x": x}, step_size=0.02, n_steps=50,
        drift=drift, adaptive=False,
    )
    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, rtol=0.001)


################################ Manual Verification Tests ###########################################


def test_manual_rk4_step_linear_decay():
    r"""Manual verification of a single RK4 step for f(x,t)=-x.

    For f(x,t) = -x with step h:
    k1 = -x
    k2 = -(x + h/2 * k1) = -(x - hx/2) = -x(1 - h/2)
    k3 = -(x + h/2 * k2) = -(x - hx(1-h/2)/2) = -x(1 - h/2 + h²/4)
    k4 = -(x + h * k3) = -(x - hx(1-h/2+h²/4)) = -x(1 - h + h²/2 - h³/4)
    x_new = x + h/6(k1 + 2k2 + 2k3 + k4)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = RK4Integrator(device=device, dtype=torch.float64)

    x = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_
    h = 0.1

    a = integrator.tableau_a
    b = integrator.tableau_b
    c = integrator.tableau_c

    # Manually compute stages
    k = []
    for i in range(4):
        if i == 0:
            x_stage = x
        else:
            dx = torch.zeros_like(x)
            for j in range(i):
                if a[i][j] != 0:
                    dx = dx + a[i][j] * k[j]
            x_stage = x + h * dx
        k.append(-x_stage)

    x_expected = x.clone()
    for i in range(4):
        x_expected = x_expected + h * b[i] * k[i]

    result = integrator.step({"x": x}, step_size=h, drift=drift)

    assert torch.allclose(result["x"], x_expected, atol=1e-14)


def test_manual_rk4_step_constant_drift():
    """For constant drift f(x,t)=c, all stages equal c, x_new = x + h*c."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = RK4Integrator(device=device, dtype=torch.float64)

    x = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    c = torch.tensor([[3.0, -1.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: c
    h = 0.5

    result = integrator.step({"x": x}, step_size=h, drift=drift)

    expected = x + h * c
    assert torch.allclose(result["x"], expected, atol=1e-14)


def test_manual_rk4_step_linear_time_drift():
    r"""For f(x,t)=t, x(0)=0, RK4 should integrate x(h)=h²/2 exactly.

    RK4 is 4th order, so it integrates polynomials up to degree 3 exactly.
    f(x,t)=t ⇒ x(h) = h²/2.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = RK4Integrator(device=device, dtype=torch.float64)

    x = torch.tensor([[0.0, 0.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: t_.unsqueeze(-1).expand_as(x_)
    h = 1.0

    result = integrator.step(
        {"x": x}, step_size=h, drift=drift,
        t=torch.zeros(1, device=device, dtype=torch.float64),
    )

    expected = x + h**2 / 2
    assert torch.allclose(result["x"], expected, atol=1e-12)


def test_manual_rk4_cubic_time_drift():
    r"""For f(x,t)=t³, x(0)=0, solution is x(h)=h⁴/4.

    RK4 should integrate degree-3 polynomial ODE exactly:
    dx/dt = t³ ⇒ x(1) = 1/4.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = RK4Integrator(device=device, dtype=torch.float64)

    x0 = torch.zeros(1, 1, device=device, dtype=torch.float64)
    drift = lambda x_, t_: (t_**3).unsqueeze(-1).expand_as(x_)

    result = integrator.step(
        {"x": x0}, step_size=1.0, drift=drift,
        t=torch.zeros(1, device=device, dtype=torch.float64),
    )

    expected = torch.tensor([[0.25]], device=device, dtype=torch.float64)
    assert torch.allclose(result["x"], expected, atol=1e-12)


def test_manual_rk4_quadratic_drift():
    """For f(x,t)=t², x(0)=0, solution x(1)=1/3. RK4 integrates exactly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = RK4Integrator(device=device, dtype=torch.float64)

    x0 = torch.zeros(1, 1, device=device, dtype=torch.float64)
    drift = lambda x_, t_: (t_**2).unsqueeze(-1).expand_as(x_)

    result = integrator.step(
        {"x": x0}, step_size=1.0, drift=drift,
        t=torch.zeros(1, device=device, dtype=torch.float64),
    )

    expected = torch.tensor([[1.0 / 3.0]], device=device, dtype=torch.float64)
    assert torch.allclose(result["x"], expected, atol=1e-14)


def test_rk4_exponential_decay_numerical():
    r"""Numerical check: RK4 approximation to e^{-h}.

    For f(x,t)=-x, h=0.1, x(0)=1:
    k1 = -1
    k2 = -(1 + 0.05*(-1)) = -0.95
    k3 = -(1 + 0.05*(-0.95)) = -0.9525
    k4 = -(1 + 0.1*(-0.9525)) = -0.90475
    x_new = 1 + 0.1/6*(-1 + 2*(-0.95) + 2*(-0.9525) + (-0.90475))
          = 1 + 0.1/6 * (-5.70975)
          = 1 - 0.0951625
          = 0.9048375
    True exp(-0.1) = 0.90483741803...
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = RK4Integrator(device=device, dtype=torch.float64)

    x = torch.tensor([[1.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_
    h = 0.1

    result = integrator.step({"x": x}, step_size=h, drift=drift)

    expected_hand = 1.0 + (h / 6) * (-1.0 + 2 * (-0.95) + 2 * (-0.9525) + (-0.90475))
    assert math.isclose(result["x"].item(), expected_hand, rel_tol=1e-12)

    # Should also be very close to exp(-0.1)
    assert math.isclose(result["x"].item(), math.exp(-0.1), rel_tol=1e-5)


################################ Convergence and Accuracy Tests ###########################################


def test_fourth_order_convergence():
    """Halving step size should reduce error by ~2^4 = 16x."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = RK4Integrator(device=device, dtype=torch.float64)

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
        assert ratio > 12, f"Expected ~16x error reduction for 4th order, got {ratio:.1f}"


def test_rk4_vs_heun_accuracy():
    """RK4 (order 4) should be more accurate than Heun (order 2)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    from torchebm.integrators import HeunIntegrator

    rk4 = RK4Integrator(device=device, dtype=dtype)
    heun = HeunIntegrator(device=device, dtype=dtype)

    x0 = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
    drift = lambda x_, t_: -x_
    step_size = 0.1
    n_steps = 10

    rk4_result = rk4.integrate(
        {"x": x0.clone()}, step_size=step_size, n_steps=n_steps,
        drift=drift, adaptive=False,
    )
    heun_result = heun.integrate(
        {"x": x0.clone()}, step_size=step_size, n_steps=n_steps,
        drift=drift, adaptive=False,
    )

    expected = x0 * math.exp(-1.0)
    rk4_error = torch.abs(rk4_result["x"] - expected).max().item()
    heun_error = torch.abs(heun_result["x"] - expected).max().item()

    assert rk4_error < heun_error, (
        f"RK4 error {rk4_error} should be less than Heun error {heun_error}"
    )


def test_exponential_decay_convergence():
    """Errors should decrease monotonically as step size shrinks."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = RK4Integrator(device=device, dtype=torch.float64)

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
    integrator = RK4Integrator(device=device, dtype=torch.float64)

    x = torch.ones(10, 2, device=device, dtype=torch.float64) * 0.5
    drift = lambda x_, t_: -(x_**3)

    result = integrator.integrate(
        {"x": x}, step_size=0.01, n_steps=100,
        drift=drift, adaptive=False,
    )

    x0_val = 0.5
    expected_val = 1.0 / math.sqrt(2 * 1.0 + 1.0 / x0_val**2)
    expected = torch.full_like(x, expected_val)

    assert torch.allclose(result["x"], expected, atol=1e-8)


def test_harmonic_oscillator():
    """dx/dt=v, dv/dt=-x. After 2π the state returns to start."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = RK4Integrator(device=device, dtype=torch.float64)

    state_vec = torch.tensor([[1.0, 0.0]], device=device, dtype=torch.float64)

    def drift(s, t_):
        return torch.cat([s[:, 1:2], -s[:, 0:1]], dim=-1)

    t_final = 2 * math.pi
    n_steps = 1000
    step_size = t_final / n_steps
    result = integrator.integrate(
        {"x": state_vec}, step_size=step_size, n_steps=n_steps,
        drift=drift, adaptive=False,
    )

    assert torch.allclose(result["x"], state_vec, atol=1e-5)


def test_polynomial_degree4_not_exact_single_step():
    r"""dx/dt = t⁴, x(0)=0 ⇒ x(1)=1/5. RK4 should NOT integrate t⁴ exactly
    in a single step (only exact up to degree 3)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = RK4Integrator(device=device, dtype=torch.float64)

    x0 = torch.zeros(1, 1, device=device, dtype=torch.float64)
    drift = lambda x_, t_: (t_**4).unsqueeze(-1).expand_as(x_)

    result = integrator.step(
        {"x": x0}, step_size=1.0, drift=drift,
        t=torch.zeros(1, device=device, dtype=torch.float64),
    )

    exact = torch.tensor([[0.2]], device=device, dtype=torch.float64)
    # Should NOT be exact — there should be a nonzero error
    assert not torch.allclose(result["x"], exact, atol=1e-10)
    # But should be a reasonable approximation
    assert torch.allclose(result["x"], exact, atol=0.02)


################################ Reproducibility Tests ###########################################


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


################################ Edge Cases ###########################################


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
    integrator = RK4Integrator(device=device, dtype=torch.float64)
    model = DoubleWellModel(barrier_height=2.0).to(device)

    x = torch.ones(10, 1, device=device, dtype=torch.float64) * 1.5
    drift = lambda x_, t_: -model.gradient(x_.float()).double()

    result = integrator.integrate(
        {"x": x}, step_size=0.01, n_steps=200,
        drift=drift, adaptive=False,
    )

    assert torch.all(torch.isfinite(result["x"]))
    assert torch.all(torch.abs(result["x"]) > 0.5)


def test_quadratic_drift():
    """f(x)=-x², starting from 1 all values should decrease."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = RK4Integrator(device=device, dtype=torch.float32)

    x = torch.ones(10, 2, device=device)
    drift = lambda x_, t_: -(x_**2)
    n_steps = 50
    t = torch.linspace(0, 0.5, n_steps, device=device)

    result = integrator.integrate(
        {"x": x}, step_size=0.01, n_steps=n_steps,
        drift=drift, t=t, adaptive=False,
    )

    assert torch.all(torch.isfinite(result["x"]))
    assert torch.all(result["x"] < x)


def test_stiff_ode():
    """Moderately stiff: dx/dt = -50*(x - cos(t))."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = RK4Integrator(device=device, dtype=torch.float64)

    x0 = torch.tensor([[1.0]], device=device, dtype=torch.float64)
    lam = 50.0

    def drift(x_, t_):
        cos_t = torch.cos(t_).unsqueeze(-1)
        return -lam * (x_ - cos_t)

    result = integrator.integrate(
        {"x": x0}, step_size=0.001, n_steps=1000,
        drift=drift, adaptive=False,
    )

    expected_approx = math.cos(1.0)
    assert torch.abs(result["x"].squeeze() - expected_approx).item() < 0.1
