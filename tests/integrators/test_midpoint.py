"""Tests for the explicit midpoint Runge-Kutta integrator."""

import math

import pytest
import torch

from tests.conftest import requires_cuda
from torchebm.integrators import MidpointIntegrator, get_integrator


@pytest.fixture
def integrator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MidpointIntegrator(device=device, dtype=torch.float64)


def test_midpoint_initialization_with_device(integrator):
    assert integrator.device == torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    assert integrator.dtype == torch.float64


@requires_cuda
def test_midpoint_cuda():
    integrator = MidpointIntegrator(device="cuda", dtype=torch.float32)
    assert integrator.device == torch.device("cuda")


def test_midpoint_tableau():
    integrator = MidpointIntegrator()

    assert integrator.tableau_a == ((), (0.5,))
    assert integrator.tableau_b == (0.0, 1.0)
    assert integrator.tableau_c == (0.0, 0.5)
    assert integrator.n_stages == 2


def test_midpoint_tableau_consistency():
    integrator = MidpointIntegrator()

    assert math.isclose(sum(integrator.tableau_b), 1.0, abs_tol=1e-14)
    for row, node in zip(integrator.tableau_a, integrator.tableau_c):
        assert math.isclose(sum(row), node, abs_tol=1e-14)


def test_midpoint_no_adaptive_params():
    integrator = MidpointIntegrator()

    assert integrator.error_weights is None
    assert integrator.order is None
    assert integrator.fsal is False


def test_midpoint_registry_name():
    assert isinstance(get_integrator("midpoint"), MidpointIntegrator)


def test_midpoint_single_step_linear_growth(integrator):
    x = torch.ones(3, 2, device=integrator.device, dtype=torch.float64)

    result = integrator.step({"x": x}, step_size=0.1, drift=lambda x_, t_: x_)

    expected = x * (1 + 0.1 + 0.1**2 / 2)
    assert torch.allclose(result["x"], expected)


def test_midpoint_uses_midpoint_time(integrator):
    x = torch.zeros(2, 1, device=integrator.device, dtype=torch.float64)
    t = torch.full((2,), 0.25, device=integrator.device, dtype=torch.float64)

    result = integrator.step(
        {"x": x}, step_size=0.2, drift=lambda x_, t_: t_[:, None], t=t
    )

    expected = x + 0.2 * (0.25 + 0.2 / 2)
    assert torch.allclose(result["x"], expected)


def test_midpoint_requires_drift(integrator):
    x = torch.ones(2, 1, device=integrator.device, dtype=torch.float64)

    with pytest.raises(ValueError, match="drift must be provided explicitly"):
        integrator.step({"x": x}, step_size=0.1)


def test_midpoint_default_time_is_zero(integrator):
    x = torch.ones(2, 1, device=integrator.device, dtype=torch.float64)
    times = []

    def recording_drift(x_, t_):
        times.append(t_.clone())
        return -x_

    integrator.step({"x": x}, step_size=0.1, drift=recording_drift)

    assert torch.allclose(
        times[0], torch.zeros(2, device=integrator.device, dtype=torch.float64)
    )


def test_midpoint_tensor_step_size(integrator):
    x = torch.ones(2, 1, device=integrator.device, dtype=torch.float64)
    step_size = torch.tensor(0.1, device=integrator.device, dtype=torch.float64)

    result = integrator.step({"x": x}, step_size=step_size, drift=lambda x_, t_: -x_)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_midpoint_integrate_linear_decay(integrator):
    x = torch.ones(3, 2, device=integrator.device, dtype=torch.float64)

    result = integrator.integrate(
        {"x": x},
        step_size=0.01,
        n_steps=100,
        drift=lambda x_, t_: -x_,
        adaptive=False,
    )

    assert torch.allclose(result["x"], x * math.exp(-1.0), rtol=5e-5, atol=1e-7)


def test_midpoint_integrate_matches_single_step(integrator):
    x = torch.randn(3, 2, device=integrator.device, dtype=torch.float64)
    drift = lambda x_, t_: -x_

    step_result = integrator.step({"x": x.clone()}, step_size=0.1, drift=drift)
    integrate_result = integrator.integrate(
        {"x": x.clone()},
        step_size=0.1,
        n_steps=1,
        drift=drift,
        adaptive=False,
    )

    assert torch.allclose(step_result["x"], integrate_result["x"])


def test_midpoint_integrate_rejects_invalid_step_count(integrator):
    x = torch.ones(2, 1, device=integrator.device, dtype=torch.float64)

    with pytest.raises(ValueError, match="n_steps must be positive"):
        integrator.integrate(
            {"x": x},
            step_size=0.1,
            n_steps=0,
            drift=lambda x_, t_: -x_,
            adaptive=False,
        )


def test_midpoint_has_second_order_convergence(integrator):

    def solve(step_size):
        state = {"x": torch.ones(1, 1, device=integrator.device, dtype=torch.float64)}
        for _ in range(round(1 / step_size)):
            state = integrator.step(state, step_size=step_size, drift=lambda x_, t_: x_)
        return state["x"].item()

    coarse_error = abs(solve(0.1) - math.e)
    fine_error = abs(solve(0.05) - math.e)

    assert coarse_error / fine_error > 3.5
