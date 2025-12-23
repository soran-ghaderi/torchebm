"""Tests for LeapfrogIntegrator."""

import math

import pytest
import torch
import numpy as np

from torchebm.core import BaseModel, GaussianModel, DoubleWellModel, HarmonicModel
from torchebm.integrators import LeapfrogIntegrator
from tests.conftest import requires_cuda


################################ Test Fixtures ###########################################


@pytest.fixture
def integrator():
    """Create a default LeapfrogIntegrator."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return LeapfrogIntegrator(device=device, dtype=torch.float32)


@pytest.fixture
def gaussian_model():
    """Create a Gaussian energy model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return GaussianModel(
        mean=torch.zeros(2, device=device), cov=torch.eye(2, device=device)
    ).to(device)


@pytest.fixture
def harmonic_model():
    """Create a harmonic oscillator potential."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HarmonicModel(center=torch.zeros(2, device=device), k=1.0).to(device)


@pytest.fixture
def double_well_model():
    """Create a double-well energy model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return DoubleWellModel(barrier_height=2.0).to(device)


################################ Initialization Tests ###########################################


def test_leapfrog_initialization():
    """Test basic initialization."""
    integrator = LeapfrogIntegrator()
    assert isinstance(integrator, LeapfrogIntegrator)


def test_leapfrog_initialization_with_device():
    """Test initialization with specific device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = LeapfrogIntegrator(device=device, dtype=torch.float32)
    assert integrator.device == torch.device(device)
    assert integrator.dtype == torch.float32


@requires_cuda
def test_leapfrog_cuda():
    """Test CUDA initialization."""
    integrator = LeapfrogIntegrator(device="cuda", dtype=torch.float32)
    assert integrator.device == torch.device("cuda")


################################ Step Method Tests ###########################################


def test_step_basic(integrator, gaussian_model):
    """Test basic step with model."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    p = torch.randn(10, 2, device=device)
    state = {"x": x, "p": p}
    step_size = 0.01

    result = integrator.step(state, model=gaussian_model, step_size=step_size)

    assert "x" in result
    assert "p" in result
    assert result["x"].shape == x.shape
    assert result["p"].shape == p.shape
    assert torch.all(torch.isfinite(result["x"]))
    assert torch.all(torch.isfinite(result["p"]))


def test_step_with_potential_grad(integrator):
    """Test step with custom potential gradient."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    p = torch.randn(10, 2, device=device)
    state = {"x": x, "p": p}
    step_size = 0.01
    potential_grad = lambda x_: x_  # Harmonic potential: U(x) = 0.5 * x^2

    result = integrator.step(
        state, model=None, step_size=step_size, potential_grad=potential_grad
    )

    assert "x" in result
    assert "p" in result
    assert result["x"].shape == x.shape
    assert result["p"].shape == p.shape


def test_step_requires_potential(integrator):
    """Test that step raises error when neither model nor potential_grad is provided."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    p = torch.randn(10, 2, device=device)
    state = {"x": x, "p": p}

    with pytest.raises(ValueError, match="Either `model` must be provided"):
        integrator.step(state, model=None, step_size=0.01)


def test_step_with_scalar_mass(integrator, gaussian_model):
    """Test step with scalar mass."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    p = torch.randn(10, 2, device=device)
    state = {"x": x, "p": p}
    step_size = 0.01
    mass = 2.0

    result = integrator.step(state, model=gaussian_model, step_size=step_size, mass=mass)

    assert result["x"].shape == x.shape
    assert result["p"].shape == p.shape
    assert torch.all(torch.isfinite(result["x"]))
    assert torch.all(torch.isfinite(result["p"]))


def test_step_with_tensor_mass(integrator, gaussian_model):
    """Test step with tensor mass (per-dimension mass)."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    p = torch.randn(10, 2, device=device)
    state = {"x": x, "p": p}
    step_size = 0.01
    mass = torch.tensor([1.0, 2.0], device=device)

    result = integrator.step(state, model=gaussian_model, step_size=step_size, mass=mass)

    assert result["x"].shape == x.shape
    assert result["p"].shape == p.shape
    assert torch.all(torch.isfinite(result["x"]))
    assert torch.all(torch.isfinite(result["p"]))


################################ Integrate Method Tests ###########################################


def test_integrate_basic(integrator, gaussian_model):
    """Test basic integration."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    p = torch.randn(10, 2, device=device)
    state = {"x": x, "p": p}
    step_size = 0.01
    n_steps = 100

    result = integrator.integrate(
        state, model=gaussian_model, step_size=step_size, n_steps=n_steps
    )

    assert "x" in result
    assert "p" in result
    assert result["x"].shape == x.shape
    assert result["p"].shape == p.shape
    assert torch.all(torch.isfinite(result["x"]))
    assert torch.all(torch.isfinite(result["p"]))


def test_integrate_invalid_n_steps(integrator, gaussian_model):
    """Test that integrate raises error for invalid n_steps."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    p = torch.randn(10, 2, device=device)
    state = {"x": x, "p": p}

    with pytest.raises(ValueError, match="n_steps must be positive"):
        integrator.integrate(state, model=gaussian_model, step_size=0.01, n_steps=0)

    with pytest.raises(ValueError, match="n_steps must be positive"):
        integrator.integrate(state, model=gaussian_model, step_size=0.01, n_steps=-5)


def test_integrate_with_potential_grad(integrator):
    """Test integration with custom potential gradient."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    p = torch.randn(10, 2, device=device)
    state = {"x": x, "p": p}
    step_size = 0.01
    n_steps = 100
    potential_grad = lambda x_: x_

    result = integrator.integrate(
        state,
        model=None,
        step_size=step_size,
        n_steps=n_steps,
        potential_grad=potential_grad,
    )

    assert result["x"].shape == x.shape
    assert result["p"].shape == p.shape


################################ Manual Verification Tests ###########################################


def test_manual_leapfrog_step():
    """Manual verification of single leapfrog step.

    Leapfrog (Störmer-Verlet) algorithm:
    p_{1/2} = p - (ε/2) * ∇U(x)
    x' = x + ε * p_{1/2}
    p' = p_{1/2} - (ε/2) * ∇U(x')

    For U(x) = 0.5 * x^2, ∇U(x) = x
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = LeapfrogIntegrator(device=device, dtype=torch.float64)

    x = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    p = torch.tensor([[0.5, -0.5]], device=device, dtype=torch.float64)
    state = {"x": x, "p": p}
    step_size = 0.1
    potential_grad = lambda x_: x_  # For U(x) = 0.5 * x^2

    result = integrator.step(
        state, model=None, step_size=step_size, potential_grad=potential_grad
    )

    # Manual calculation:
    # p_half = [0.5, -0.5] - 0.05 * [1.0, 2.0] = [0.45, -0.6]
    # x_new = [1.0, 2.0] + 0.1 * [0.45, -0.6] = [1.045, 1.94]
    # p_new = [0.45, -0.6] - 0.05 * [1.045, 1.94] = [0.39775, -0.697]
    p_half = p - 0.5 * step_size * x
    x_new = x + step_size * p_half
    p_new = p_half - 0.5 * step_size * x_new

    assert torch.allclose(result["x"], x_new, atol=1e-10)
    assert torch.allclose(result["p"], p_new, atol=1e-10)


def test_manual_leapfrog_step_with_mass():
    """Manual verification of leapfrog step with mass.

    For Hamiltonian H = p²/(2m) + U(x):
    Velocity v = p/m
    
    Leapfrog with mass:
    p_{1/2} = p - (ε/2) * ∇U(x)
    x' = x + ε * p_{1/2} / m
    p' = p_{1/2} - (ε/2) * ∇U(x')

    For U(x) = 0.5 * x^2, ∇U(x) = x
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = LeapfrogIntegrator(device=device, dtype=torch.float64)

    x = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    p = torch.tensor([[0.5, -0.5]], device=device, dtype=torch.float64)
    state = {"x": x, "p": p}
    step_size = 0.1
    mass = 2.0
    potential_grad = lambda x_: x_  # For U(x) = 0.5 * x^2

    result = integrator.step(
        state, model=None, step_size=step_size, mass=mass, potential_grad=potential_grad
    )

    # Manual calculation with mass:
    # p_half = p - (ε/2) * ∇U(x) = [0.5, -0.5] - 0.05 * [1.0, 2.0] = [0.45, -0.6]
    # x_new = x + ε * p_half / m = [1.0, 2.0] + 0.1 * [0.45, -0.6] / 2.0 = [1.0225, 1.97]
    # p_new = p_half - (ε/2) * ∇U(x_new) = [0.45, -0.6] - 0.05 * [1.0225, 1.97] = [0.398875, -0.6985]
    p_half = p - 0.5 * step_size * x
    x_new = x + step_size * p_half / mass
    p_new = p_half - 0.5 * step_size * x_new

    assert torch.allclose(result["x"], x_new, atol=1e-10)
    assert torch.allclose(result["p"], p_new, atol=1e-10)


def test_energy_conservation_with_mass():
    """Test energy conservation with non-unit mass.

    For Hamiltonian H = p²/(2m) + 0.5*x², leapfrog should conserve energy.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = LeapfrogIntegrator(device=device, dtype=torch.float64)

    x0 = torch.tensor([[1.0, 0.0]], device=device, dtype=torch.float64)
    p0 = torch.tensor([[0.0, 2.0]], device=device, dtype=torch.float64)
    state = {"x": x0, "p": p0}
    mass = 2.0
    potential_grad = lambda x_: x_

    def compute_energy(x, p, m):
        kinetic = 0.5 * torch.sum(p**2) / m
        potential = 0.5 * torch.sum(x**2)
        return kinetic + potential

    initial_energy = compute_energy(x0, p0, mass)

    step_size = 0.05
    n_steps = 500
    result = integrator.integrate(
        state,
        model=None,
        step_size=step_size,
        n_steps=n_steps,
        mass=mass,
        potential_grad=potential_grad,
    )

    final_energy = compute_energy(result["x"], result["p"], mass)

    relative_error = torch.abs(final_energy - initial_energy) / initial_energy
    assert relative_error < 0.02, f"Energy not conserved: {relative_error}"


def test_energy_conservation_harmonic():
    """Test energy conservation in harmonic oscillator.

    For Hamiltonian H = 0.5 * p^2 + 0.5 * x^2, leapfrog should
    approximately conserve energy.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = LeapfrogIntegrator(device=device, dtype=torch.float64)

    x0 = torch.tensor([[1.0, 0.0]], device=device, dtype=torch.float64)
    p0 = torch.tensor([[0.0, 1.0]], device=device, dtype=torch.float64)
    state = {"x": x0, "p": p0}
    potential_grad = lambda x_: x_

    def compute_energy(x, p):
        kinetic = 0.5 * torch.sum(p**2)
        potential = 0.5 * torch.sum(x**2)
        return kinetic + potential

    initial_energy = compute_energy(x0, p0)

    # Integrate for many steps
    step_size = 0.1
    n_steps = 1000
    result = integrator.integrate(
        state,
        model=None,
        step_size=step_size,
        n_steps=n_steps,
        potential_grad=potential_grad,
    )

    final_energy = compute_energy(result["x"], result["p"])

    # Energy should be conserved to high precision for symplectic integrator
    relative_error = torch.abs(final_energy - initial_energy) / initial_energy
    assert relative_error < 0.01, f"Energy not conserved: {relative_error}"


def test_symplecticity_phase_space_volume():
    """Test that leapfrog preserves phase space volume (symplecticity).

    The Jacobian determinant of a symplectic map should be 1.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = LeapfrogIntegrator(device=device, dtype=torch.float64)

    # Create a small perturbation around initial state
    x0 = torch.tensor([[1.0]], device=device, dtype=torch.float64)
    p0 = torch.tensor([[0.5]], device=device, dtype=torch.float64)
    potential_grad = lambda x_: x_
    step_size = 0.1
    eps = 1e-5

    # Compute the Jacobian numerically
    # For 1D: J = [[dx'/dx, dx'/dp], [dp'/dx, dp'/dp]]

    # Perturb x
    state_x_plus = {"x": x0 + eps, "p": p0.clone()}
    state_x_minus = {"x": x0 - eps, "p": p0.clone()}
    result_x_plus = integrator.step(state_x_plus, model=None, step_size=step_size, potential_grad=potential_grad)
    result_x_minus = integrator.step(state_x_minus, model=None, step_size=step_size, potential_grad=potential_grad)

    dx_dx = (result_x_plus["x"] - result_x_minus["x"]) / (2 * eps)
    dp_dx = (result_x_plus["p"] - result_x_minus["p"]) / (2 * eps)

    # Perturb p
    state_p_plus = {"x": x0.clone(), "p": p0 + eps}
    state_p_minus = {"x": x0.clone(), "p": p0 - eps}
    result_p_plus = integrator.step(state_p_plus, model=None, step_size=step_size, potential_grad=potential_grad)
    result_p_minus = integrator.step(state_p_minus, model=None, step_size=step_size, potential_grad=potential_grad)

    dx_dp = (result_p_plus["x"] - result_p_minus["x"]) / (2 * eps)
    dp_dp = (result_p_plus["p"] - result_p_minus["p"]) / (2 * eps)

    # Jacobian determinant
    det_J = dx_dx * dp_dp - dx_dp * dp_dx

    # Should be 1 for symplectic map
    assert torch.allclose(det_J, torch.ones_like(det_J), atol=1e-6), f"Jacobian det = {det_J.item()}"


def test_reversibility():
    """Test time-reversibility of leapfrog.

    If we integrate forward n steps and then backward n steps (with negated momentum),
    we should return to the starting point.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = LeapfrogIntegrator(device=device, dtype=torch.float64)

    x0 = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    p0 = torch.tensor([[0.5, -0.5]], device=device, dtype=torch.float64)
    potential_grad = lambda x_: x_
    step_size = 0.05
    n_steps = 100

    # Forward integration
    state = {"x": x0, "p": p0}
    forward_result = integrator.integrate(
        state,
        model=None,
        step_size=step_size,
        n_steps=n_steps,
        potential_grad=potential_grad,
    )

    # Negate momentum and integrate again (backwards in time)
    state_reversed = {"x": forward_result["x"], "p": -forward_result["p"]}
    backward_result = integrator.integrate(
        state_reversed,
        model=None,
        step_size=step_size,
        n_steps=n_steps,
        potential_grad=potential_grad,
    )

    # Should return to original state (with negated momentum)
    assert torch.allclose(backward_result["x"], x0, atol=1e-10)
    assert torch.allclose(backward_result["p"], -p0, atol=1e-10)


def test_harmonic_oscillator_trajectory():
    """Test that harmonic oscillator follows expected circular trajectory."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = LeapfrogIntegrator(device=device, dtype=torch.float64)

    # Initial conditions: (x, p) = (1, 0)
    # For harmonic oscillator, trajectory should be a circle in phase space
    x0 = torch.tensor([[1.0]], device=device, dtype=torch.float64)
    p0 = torch.tensor([[0.0]], device=device, dtype=torch.float64)
    potential_grad = lambda x_: x_

    # After time t, analytical solution: x(t) = cos(t), p(t) = -sin(t)
    # Full period is 2π
    step_size = 0.01
    n_steps = int(2 * math.pi / step_size)

    state = {"x": x0, "p": p0}
    result = integrator.integrate(
        state,
        model=None,
        step_size=step_size,
        n_steps=n_steps,
        potential_grad=potential_grad,
    )

    # After one full period, should return near starting point
    assert torch.allclose(result["x"], x0, atol=0.05)
    assert torch.allclose(result["p"], p0, atol=0.05)


################################ Tests with Energy Models ###########################################


def test_with_double_well_model(integrator, double_well_model):
    """Test integration with double-well potential."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    p = torch.randn(10, 2, device=device)
    state = {"x": x, "p": p}
    step_size = 0.01
    n_steps = 100

    result = integrator.integrate(
        state, model=double_well_model, step_size=step_size, n_steps=n_steps
    )

    assert result["x"].shape == x.shape
    assert result["p"].shape == p.shape
    assert torch.all(torch.isfinite(result["x"]))
    assert torch.all(torch.isfinite(result["p"]))


################################ NaN Handling Tests ###########################################


def test_nan_handling():
    """Test that NaN values are replaced."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = LeapfrogIntegrator(device=device, dtype=torch.float32)

    # Create potential that produces NaN gradients for large x
    class NanInducingPotential(BaseModel):
        def forward(self, x):
            return 0.5 * torch.sum(x**2, dim=-1)

        def gradient(self, x):
            grad = x.clone()
            mask = torch.norm(x, dim=-1, keepdim=True) > 5.0
            grad[mask.expand_as(grad)] = float('nan')
            return grad

    model = NanInducingPotential().to(device)

    # Start near the NaN threshold
    x = torch.ones(5, 2, device=device) * 4.9
    p = torch.randn(5, 2, device=device)
    state = {"x": x, "p": p}

    result = integrator.step(state, model=model, step_size=0.1)

    # Result should not contain NaN due to nan_to_num handling
    assert torch.all(torch.isfinite(result["x"]))
    assert torch.all(torch.isfinite(result["p"]))


################################ Reproducibility Tests ###########################################


def test_reproducibility(integrator, gaussian_model):
    """Test that same inputs produce same outputs."""
    device = integrator.device

    torch.manual_seed(42)
    x1 = torch.randn(10, 2, device=device)
    p1 = torch.randn(10, 2, device=device)

    torch.manual_seed(42)
    x2 = torch.randn(10, 2, device=device)
    p2 = torch.randn(10, 2, device=device)

    result1 = integrator.step({"x": x1, "p": p1}, model=gaussian_model, step_size=0.01)
    result2 = integrator.step({"x": x2, "p": p2}, model=gaussian_model, step_size=0.01)

    assert torch.allclose(result1["x"], result2["x"])
    assert torch.allclose(result1["p"], result2["p"])


################################ Edge Cases ###########################################


def test_large_batch_size(integrator):
    """Test with large batch size."""
    device = integrator.device
    x = torch.randn(1000, 10, device=device)
    p = torch.randn(1000, 10, device=device)
    state = {"x": x, "p": p}
    potential_grad = lambda x_: x_  # Harmonic potential

    result = integrator.step(state, model=None, step_size=0.01, potential_grad=potential_grad)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_high_dimension(integrator):
    """Test with high-dimensional input."""
    device = integrator.device
    dim = 100
    x = torch.randn(10, dim, device=device)
    p = torch.randn(10, dim, device=device)
    state = {"x": x, "p": p}
    potential_grad = lambda x_: x_

    result = integrator.step(
        state, model=None, step_size=0.01, potential_grad=potential_grad
    )

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_small_step_size(integrator, gaussian_model):
    """Test with very small step size."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    p = torch.randn(10, 2, device=device)
    state = {"x": x, "p": p}

    result = integrator.step(state, model=gaussian_model, step_size=1e-8)

    # With tiny step, state should barely change
    assert torch.allclose(result["x"], x, atol=1e-5)


def test_large_step_size(integrator, gaussian_model):
    """Test stability with large step size."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    p = torch.randn(10, 2, device=device)
    state = {"x": x, "p": p}

    result = integrator.step(state, model=gaussian_model, step_size=1.0)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))
    