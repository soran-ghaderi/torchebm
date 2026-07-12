"""Tests for the explicit midpoint Runge-Kutta integrator."""

import math

import torch

from torchebm.integrators import MidpointIntegrator, get_integrator


def test_midpoint_tableau():
    integrator = MidpointIntegrator()

    assert integrator.tableau_a == ((), (0.5,))
    assert integrator.tableau_b == (0.0, 1.0)
    assert integrator.tableau_c == (0.0, 0.5)
    assert integrator.n_stages == 2
    assert integrator.error_weights is None


def test_midpoint_registry_name():
    assert isinstance(get_integrator("midpoint"), MidpointIntegrator)


def test_midpoint_single_step_linear_growth():
    integrator = MidpointIntegrator(dtype=torch.float64)
    x = torch.ones(3, 2, dtype=torch.float64)

    result = integrator.step({"x": x}, step_size=0.1, drift=lambda x_, t_: x_)

    expected = x * (1 + 0.1 + 0.1**2 / 2)
    assert torch.allclose(result["x"], expected)


def test_midpoint_uses_midpoint_time():
    integrator = MidpointIntegrator(dtype=torch.float64)
    x = torch.zeros(2, 1, dtype=torch.float64)
    t = torch.full((2,), 0.25, dtype=torch.float64)

    result = integrator.step(
        {"x": x}, step_size=0.2, drift=lambda x_, t_: t_[:, None], t=t
    )

    expected = x + 0.2 * (0.25 + 0.2 / 2)
    assert torch.allclose(result["x"], expected)


def test_midpoint_has_second_order_convergence():
    integrator = MidpointIntegrator(dtype=torch.float64)

    def solve(step_size):
        state = {"x": torch.ones(1, 1, dtype=torch.float64)}
        for _ in range(round(1 / step_size)):
            state = integrator.step(state, step_size=step_size, drift=lambda x_, t_: x_)
        return state["x"].item()

    coarse_error = abs(solve(0.1) - math.e)
    fine_error = abs(solve(0.05) - math.e)

    assert coarse_error / fine_error > 3.5
