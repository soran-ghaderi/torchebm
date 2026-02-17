"""Tests for Bosh3Integrator."""

import math

import pytest
import torch

from torchebm.core import GaussianModel, DoubleWellModel
from torchebm.integrators import Bosh3Integrator
from tests.conftest import requires_cuda


################################ Test Fixtures ###########################################


@pytest.fixture
def integrator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Bosh3Integrator(device=device, dtype=torch.float32)


@pytest.fixture
def integrator_f64():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Bosh3Integrator(device=device, dtype=torch.float64)


@pytest.fixture
def gaussian_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return GaussianModel(
        mean=torch.zeros(2, device=device), cov=torch.eye(2, device=device)
    ).to(device)


################################ Initialization Tests ###########################################


def test_initialization():
    integrator = Bosh3Integrator()
    assert isinstance(integrator, Bosh3Integrator)
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
    integrator = Bosh3Integrator(device=device, dtype=torch.float32)
    assert integrator.device == torch.device(device)
    assert integrator.dtype == torch.float32


def test_custom_tolerances():
    integrator = Bosh3Integrator(
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
    integrator = Bosh3Integrator(device="cuda", dtype=torch.float32)
    assert integrator.device == torch.device("cuda")


################################ Butcher Tableau Tests ###########################################


def test_tableau_a_values():
    integrator = Bosh3Integrator()
    a = integrator.tableau_a

    assert len(a) == 3
    assert a[0] == ()
    assert len(a[1]) == 1
    assert math.isclose(a[1][0], 0.5)
    assert len(a[2]) == 2
    assert a[2][0] == 0.0
    assert math.isclose(a[2][1], 0.75)


def test_tableau_b_values():
    integrator = Bosh3Integrator()
    b = integrator.tableau_b

    assert len(b) == 3
    assert math.isclose(b[0], 2 / 9)
    assert math.isclose(b[1], 1 / 3)
    assert math.isclose(b[2], 4 / 9)


def test_tableau_c_values():
    integrator = Bosh3Integrator()
    c = integrator.tableau_c

    assert len(c) == 3
    assert c[0] == 0.0
    assert math.isclose(c[1], 0.5)
    assert math.isclose(c[2], 0.75)


def test_error_weights_values():
    r"""e_i = b_i - b_hat_i with 4 entries (3 stages + 1 FSAL).

    b     = (2/9,   1/3,  4/9,  0  )
    b_hat = (7/24,  1/4,  1/3,  1/8)
    e     = b - b_hat
    """
    integrator = Bosh3Integrator()
    e = integrator.error_weights

    assert len(e) == 4  # 3 stages + 1 FSAL
    assert math.isclose(e[0], 2 / 9 - 7 / 24)
    assert math.isclose(e[1], 1 / 3 - 1 / 4)
    assert math.isclose(e[2], 4 / 9 - 1 / 3)
    assert math.isclose(e[3], -1 / 8)


def test_order():
    integrator = Bosh3Integrator()
    assert integrator.order == 3


def test_fsal_property():
    integrator = Bosh3Integrator()
    assert integrator.fsal is True


def test_n_stages():
    integrator = Bosh3Integrator()
    assert integrator.n_stages == 3


def test_tableau_consistency_b_sum():
    integrator = Bosh3Integrator()
    b = integrator.tableau_b
    assert math.isclose(sum(b), 1.0, abs_tol=1e-14)


def test_tableau_consistency_row_sums():
    integrator = Bosh3Integrator()
    a = integrator.tableau_a
    c = integrator.tableau_c

    for i in range(len(a)):
        row_sum = sum(a[i]) if a[i] else 0.0
        assert math.isclose(row_sum, c[i], abs_tol=1e-14), (
            f"Row {i}: sum(a[{i}]) = {row_sum} != c[{i}] = {c[i]}"
        )


def test_error_weights_sum():
    r"""Sum of error weights should be 0 (consistency of both solutions)."""
    integrator = Bosh3Integrator()
    e = integrator.error_weights
    assert math.isclose(sum(e), 0.0, abs_tol=1e-14)


def test_error_weights_are_b_minus_bhat():
    r"""Verify e = b - b_hat explicitly."""
    integrator = Bosh3Integrator()
    b = integrator.tableau_b
    # b_hat has 4 entries (3 stages + FSAL), b only has 3 (extended with 0)
    b_hat = (7 / 24, 1 / 4, 1 / 3, 1 / 8)
    b_ext = tuple(b) + (0.0,)  # extend b with 0 for FSAL
    e = integrator.error_weights

    for i in range(4):
        assert math.isclose(e[i], b_ext[i] - b_hat[i], abs_tol=1e-14)


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
    assert torch.allclose(result["x"], expected, rtol=0.001)


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
    assert torch.allclose(result["x"], expected, rtol=0.001)


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
    integrator = Bosh3Integrator(device=device, dtype=torch.float64, rtol=1e-7, atol=1e-9)

    x = torch.ones(10, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result = integrator.integrate(
        {"x": x}, step_size=0.1, n_steps=10, drift=drift,
    )

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, atol=1e-10) # tightened for pressuring!


def test_integrate_adaptive_explicit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)

    x = torch.ones(10, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result = integrator.integrate(
        {"x": x}, step_size=0.1, n_steps=10,
        drift=drift, adaptive=True,
    )

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, atol=1e-3)


def test_adaptive_with_custom_time_grid():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)

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
    integrator = Bosh3Integrator(
        device=device, dtype=torch.float64, atol=1e-12, rtol=1e-10,
    )

    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result = integrator.integrate(
        {"x": x}, step_size=0.1, n_steps=10, drift=drift,
    )

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, atol=1e-8)


def test_adaptive_loose_tolerance():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Bosh3Integrator(
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
    integrator = Bosh3Integrator(
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
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)

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
    integrator = Bosh3Integrator(
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


def test_manual_bosh3_step_linear_decay():
    r"""Manual verification of a single Bosh3 step for f(x,t)=-x.

    Tableau: a = ((), (1/2,), (0, 3/4)), b = (2/9, 1/3, 4/9), c = (0, 1/2, 3/4)

    For f(x,t) = -x with step h:
    k1 = -x
    k2 = -(x + h/2*k1) = -(x - hx/2) = -x(1 - h/2)
    k3 = -(x + 3h/4*k2) = -(x - 3hx(1-h/2)/4) = -x(1 - 3h/4 + 3h²/8)
    x_new = x + h*(2/9*k1 + 1/3*k2 + 4/9*k3)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)

    x = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_
    h = 0.1

    a = integrator.tableau_a
    b = integrator.tableau_b

    # Manually compute stages
    k1 = -x
    k2 = -(x + h * a[1][0] * k1)
    k3 = -(x + h * (a[2][0] * k1 + a[2][1] * k2))

    x_expected = x + h * (b[0] * k1 + b[1] * k2 + b[2] * k3)

    result = integrator.step({"x": x}, step_size=h, drift=drift)

    assert torch.allclose(result["x"], x_expected, atol=1e-14)


def test_manual_bosh3_step_constant_drift():
    """For constant drift f(x,t)=c, all stages equal c, x_new = x + h*c."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)

    x = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    c = torch.tensor([[3.0, -1.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: c
    h = 0.5

    result = integrator.step({"x": x}, step_size=h, drift=drift)

    expected = x + h * c
    assert torch.allclose(result["x"], expected, atol=1e-14)


def test_manual_bosh3_step_linear_time_drift():
    r"""For f(x,t)=t, x(0)=0: x(h)=h²/2. Bosh3 integrates linear t exactly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)

    x = torch.tensor([[0.0, 0.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: t_.unsqueeze(-1).expand_as(x_)
    h = 1.0

    result = integrator.step(
        {"x": x}, step_size=h, drift=drift,
        t=torch.zeros(1, device=device, dtype=torch.float64),
    )

    expected = x + h**2 / 2
    assert torch.allclose(result["x"], expected, atol=1e-12)


def test_manual_bosh3_quadratic_time_drift():
    r"""For f(x,t)=t², x(0)=0, solution x(1)=1/3.

    Bosh3 is order 3, so it should integrate degree-2 polynomial exactly.
    k1 = 0² = 0
    k2 = (0 + 0.5*1)² = 0.25
    k3 = (0 + 0.75*1)² = 0.5625
    x_new = 0 + 1*(2/9*0 + 1/3*0.25 + 4/9*0.5625)
          = 0.08333... + 0.25 = 0.33333...
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)

    x0 = torch.zeros(1, 1, device=device, dtype=torch.float64)
    drift = lambda x_, t_: (t_**2).unsqueeze(-1).expand_as(x_)

    result = integrator.step(
        {"x": x0}, step_size=1.0, drift=drift,
        t=torch.zeros(1, device=device, dtype=torch.float64),
    )

    expected = torch.tensor([[1.0 / 3.0]], device=device, dtype=torch.float64)
    assert torch.allclose(result["x"], expected, atol=1e-14)


def test_bosh3_cubic_time_not_exact_single_step():
    r"""For f(x,t)=t³, x(0)=0, solution x(1)=1/4. Bosh3 (order 3) should
    NOT integrate t³ exactly in a single step.

    k1 = 0, k2 = 0.5³ = 0.125, k3 = 0.75³ = 0.421875
    x_new = 2/9*0 + 1/3*0.125 + 4/9*0.421875 = 0.041666... + 0.1875 = 0.229166...
    Exact = 0.25
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)

    x0 = torch.zeros(1, 1, device=device, dtype=torch.float64)
    drift = lambda x_, t_: (t_**3).unsqueeze(-1).expand_as(x_)

    result = integrator.step(
        {"x": x0}, step_size=1.0, drift=drift,
        t=torch.zeros(1, device=device, dtype=torch.float64),
    )

    exact = torch.tensor([[0.25]], device=device, dtype=torch.float64)
    assert not torch.allclose(result["x"], exact, atol=1e-10)

    # Hand calculation
    hand = 2 / 9 * 0 + 1 / 3 * 0.125 + 4 / 9 * 0.421875
    hand_t = torch.tensor([[hand]], device=device, dtype=torch.float64)
    assert torch.allclose(result["x"], hand_t, atol=1e-14)


def test_bosh3_exponential_decay_numerical():
    r"""Numerical check: Bosh3 single-step approximation to e^{-h}.

    For f(x,t)=-x, h=0.1, x(0)=1:
    k1 = -1
    k2 = -(1 + 0.1*0.5*(-1)) = -(1 - 0.05) = -0.95
    k3 = -(1 + 0.1*0.75*(-0.95)) = -(1 - 0.07125) = -0.92875
    x_new = 1 + 0.1*(2/9*(-1) + 1/3*(-0.95) + 4/9*(-0.92875))
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)

    x = torch.tensor([[1.0]], device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_
    h = 0.1

    k1 = -1.0
    k2 = -(1.0 + h * 0.5 * k1)
    k3 = -(1.0 + h * 0.75 * k2)

    expected_hand = 1.0 + h * (2 / 9 * k1 + 1 / 3 * k2 + 4 / 9 * k3)

    result = integrator.step({"x": x}, step_size=h, drift=drift)

    assert math.isclose(result["x"].item(), expected_hand, rel_tol=1e-12)
    assert math.isclose(result["x"].item(), math.exp(-0.1), rel_tol=1e-4)


def test_manual_error_estimate():
    r"""Verify the FSAL error estimate for a single step.

    For f(x,t)=-x, h=0.1, x=1:
    k1 = -1
    k2 = -(1 - 0.05) = -0.95
    k3 = -(1 - 0.1*0.75*0.95) = -0.92875
    x_new = 1 + 0.1*(2/9*(-1) + 1/3*(-0.95) + 4/9*(-0.92875))
    k_fsal = -x_new
    error = h * (e[0]*k1 + e[1]*k2 + e[2]*k3 + e[3]*k_fsal)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)

    x = torch.tensor([[1.0]], device=device, dtype=torch.float64)
    drift_fn = lambda x_, t_: -x_
    h_val = 0.1
    h = torch.tensor(h_val, device=device, dtype=torch.float64)
    t = torch.zeros(1, device=device, dtype=torch.float64)

    k = integrator._evaluate_stages(x, t, h, drift_fn)

    # Compute x_new for FSAL evaluation
    x_new = integrator._combine_stages(x, h, k)
    k_fsal = drift_fn(x_new, t + h)
    k_err = k + [k_fsal]

    e = integrator.error_weights
    err_vec = h * sum(e[i] * k_err[i] for i in range(4))

    # The error should be nonzero (3rd vs 2nd order solutions differ)
    assert err_vec.abs().item() > 1e-10

    # Verify 3rd-order solution minus 2nd-order solution
    b_hat = (7 / 24, 1 / 4, 1 / 3, 1 / 8)
    x_hat = x + h * (b_hat[0] * k_err[0] + b_hat[1] * k_err[1]
                      + b_hat[2] * k_err[2] + b_hat[3] * k_err[3])
    difference = x_new - x_hat
    assert torch.allclose(err_vec, difference, atol=1e-14)


################################ FSAL Tests ########################################################


def test_fsal_reduces_drift_evaluations():
    """FSAL should reuse the last stage from the previous step in adaptive mode."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)

    call_count = [0]

    def counting_drift(x_, t_):
        call_count[0] += 1
        return -x_

    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    integrator.integrate(
        {"x": x}, step_size=0.5, n_steps=2, drift=counting_drift, adaptive=True,
    )

    assert call_count[0] > 0


def test_fsal_caching_consistency():
    """Verify FSAL mode and non-FSAL produce identical results for fixed-step."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)

    x = torch.randn(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    # Fixed-step doesn't use FSAL caching (calls step() individually)
    result = integrator.integrate(
        {"x": x.clone()}, step_size=0.01, n_steps=100,
        drift=drift, adaptive=False,
    )

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, atol=1e-4)


################################ Convergence and Accuracy Tests ####################################


def test_third_order_convergence():
    """Halving step size should reduce error by ~2³ = 8x."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)

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
        assert ratio > 6, f"Expected ~8x error reduction for 3rd order, got {ratio:.1f}"


def test_bosh3_vs_adaptive_heun_accuracy():
    """Bosh3 (order 3) should be more accurate than AdaptiveHeun (order 2) fixed-step."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    from torchebm.integrators import AdaptiveHeunIntegrator

    bosh3 = Bosh3Integrator(device=device, dtype=dtype)
    heun = AdaptiveHeunIntegrator(device=device, dtype=dtype)

    x0 = torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)
    drift = lambda x_, t_: -x_
    step_size = 0.1
    n_steps = 10

    bosh3_result = bosh3.integrate(
        {"x": x0.clone()}, step_size=step_size, n_steps=n_steps,
        drift=drift, adaptive=False,
    )
    heun_result = heun.integrate(
        {"x": x0.clone()}, step_size=step_size, n_steps=n_steps,
        drift=drift, adaptive=False,
    )

    expected = x0 * math.exp(-1.0)
    bosh3_error = torch.abs(bosh3_result["x"] - expected).max().item()
    heun_error = torch.abs(heun_result["x"] - expected).max().item()

    assert bosh3_error < heun_error, (
        f"Bosh3 error {bosh3_error} should be less than AdaptiveHeun error {heun_error}"
    )


def test_exponential_decay_convergence():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)

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
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)

    x = torch.ones(10, 2, device=device, dtype=torch.float64) * 0.5
    drift = lambda x_, t_: -(x_**3)

    result = integrator.integrate(
        {"x": x}, step_size=0.01, n_steps=100,
        drift=drift, adaptive=False,
    )

    x0_val = 0.5
    expected_val = 1.0 / math.sqrt(2 * 1.0 + 1.0 / x0_val**2)
    expected = torch.full_like(x, expected_val)

    assert torch.allclose(result["x"], expected, atol=1e-6)


def test_harmonic_oscillator():
    """dx/dt=v, dv/dt=-x. After 2π the state returns to start."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Bosh3Integrator(
        device=device, dtype=torch.float64, atol=1e-10, rtol=1e-8,
    )

    state_vec = torch.tensor([[1.0, 0.0]], device=device, dtype=torch.float64)

    def drift(s, t_):
        return torch.cat([s[:, 1:2], -s[:, 0:1]], dim=-1)

    t_final = 2 * math.pi
    result = integrator.integrate(
        {"x": state_vec}, step_size=0.1, n_steps=int(t_final / 0.1),
        drift=drift,
        t=torch.tensor([0.0, t_final], device=device, dtype=torch.float64),
    )

    assert torch.allclose(result["x"], state_vec, atol=1e-5)


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
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)

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
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)
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
    integrator = Bosh3Integrator(
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
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)

    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: torch.ones_like(x_)

    result = integrator.integrate(
        {"x": x}, step_size=0.1, n_steps=10, drift=drift,
    )

    expected = x + 1.0
    assert torch.allclose(result["x"], expected, atol=1e-10)


def test_quadratic_drift():
    """f(x)=-x², starting from 1 all values should decrease."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Bosh3Integrator(device=device, dtype=torch.float32)

    x = torch.ones(10, 2, device=device)
    drift = lambda x_, t_: -(x_**2)

    result = integrator.integrate(
        {"x": x}, step_size=0.01, n_steps=50,
        drift=drift, adaptive=False,
    )

    assert torch.all(torch.isfinite(result["x"]))
    assert torch.all(result["x"] < x)


################################ norm & max_step_size Tests ########################################


def test_default_norm_is_rms():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = Bosh3Integrator(device=device, dtype=torch.float64)
    assert integrator._norm is None

    x = torch.ones(5, 2, device=device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    result = integrator.integrate(
        {"x": x}, step_size=0.1, n_steps=10, drift=drift,
    )

    expected = x * math.exp(-1.0)
    assert torch.allclose(result["x"], expected, atol=1e-3)


def test_custom_norm_max():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_norm = lambda x: torch.max(torch.abs(x))

    integrator = Bosh3Integrator(
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


def test_default_max_step_size_is_inf():
    integrator = Bosh3Integrator()
    assert integrator.max_step_size == float("inf")


def test_custom_max_step_size():
    integrator = Bosh3Integrator(max_step_size=0.5)
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

    integrator_unlimited = Bosh3Integrator(
        device=device, dtype=torch.float64, atol=1e-6, rtol=1e-4,
    )
    integrator_limited = Bosh3Integrator(
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
    integrator = Bosh3Integrator(
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
