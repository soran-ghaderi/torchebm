"""Tests for the BaseSymplecticIntegrator shared contract."""

import pytest
import torch

from torchebm.core import BaseSymplecticIntegrator
from torchebm.integrators import (
    GeneralisedLeapfrogIntegrator,
    LeapfrogIntegrator,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def _drift(x, t):
    return -x  # harmonic force


def _force(x, p, t):
    return -x


def _velocity(x, p, t):
    return p


def _make_state(batch=8, dim=2, dtype=torch.float32, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return {
        "x": torch.randn(batch, dim, generator=g, dtype=dtype).to(device),
        "p": torch.randn(batch, dim, generator=g, dtype=dtype).to(device),
    }


def _integrate(integ, state, **kwargs):
    if isinstance(integ, GeneralisedLeapfrogIntegrator):
        return integ.integrate(
            state, step_size=0.1, force=_force, velocity=_velocity, **kwargs
        )
    return integ.integrate(state, step_size=0.1, drift=_drift, **kwargs)


@pytest.fixture(params=["leapfrog", "generalised"])
def integrator(request):
    if request.param == "leapfrog":
        return LeapfrogIntegrator(device=device)
    return GeneralisedLeapfrogIntegrator(device=device)


def test_hierarchy_and_separable_flags():
    assert issubclass(LeapfrogIntegrator, BaseSymplecticIntegrator)
    assert issubclass(GeneralisedLeapfrogIntegrator, BaseSymplecticIntegrator)
    assert LeapfrogIntegrator.separable is True
    assert GeneralisedLeapfrogIntegrator.separable is False


@pytest.mark.parametrize("n_steps", [0, -1, -10])
def test_integrate_rejects_nonpositive_n_steps(integrator, n_steps):
    state = _make_state()
    with pytest.raises(ValueError, match="n_steps must be positive"):
        _integrate(integrator, state, n_steps=n_steps)


def test_integrate_returns_x_p_and_does_not_mutate_input(integrator):
    state = _make_state()
    x0, p0 = state["x"].clone(), state["p"].clone()
    result = _integrate(integrator, state, n_steps=5)
    assert set(result.keys()) == {"x", "p"}
    assert torch.equal(state["x"], x0)
    assert torch.equal(state["p"], p0)
    assert result["x"].shape == x0.shape
    assert result["p"].shape == p0.shape


def test_inference_mode_matches_normal_mode(integrator):
    state = _make_state()
    out = _integrate(integrator, state, n_steps=5)
    out_inf = _integrate(integrator, state, n_steps=5, inference_mode=True)
    assert torch.allclose(out["x"], out_inf["x"])
    assert torch.allclose(out["p"], out_inf["p"])


def test_safe_mode_zeroes_nans(integrator):
    state = _make_state()
    state["x"][0, 0] = float("nan")
    result = _integrate(integrator, state, n_steps=3, safe=True)
    assert torch.isfinite(result["x"]).all()
    assert torch.isfinite(result["p"]).all()


def test_safe_mode_clamps_extreme_forces():
    integ = LeapfrogIntegrator(device=device)
    state = _make_state()
    huge_drift = lambda x, t: torch.full_like(x, 1e12)
    result = integ.integrate(
        state, step_size=0.1, n_steps=1, drift=huge_drift, safe=True
    )
    # Force is clamped to +-1e6, so a single step moves by at most
    # ~step_size * (p + 0.5 * step_size * 1e6); it must stay finite and
    # far below the unclamped 1e12 scale.
    assert torch.isfinite(result["x"]).all()
    assert result["p"].abs().max() < 1e9


def test_safe_clamp_constant():
    assert BaseSymplecticIntegrator._SAFE_CLAMP == 1e6
