"""Tests for LeapfrogIntegrator."""

import math

import pytest
import torch
import numpy as np

from torchebm.core import BaseModel, GaussianModel, DoubleWellModel, HarmonicModel
from torchebm.integrators import GeneralisedLeapfrogIntegrator, LeapfrogIntegrator
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

    result = integrator.step(state, drift=lambda x_, t_: -gaussian_model.gradient(x_), step_size=step_size)

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
        state, step_size=step_size, drift=lambda x_, t_: -potential_grad(x_)
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

    with pytest.raises(ValueError, match="drift must be provided explicitly"):
        integrator.step(state, step_size=0.01)


def test_step_with_scalar_mass(integrator, gaussian_model):
    """Test step with scalar mass."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    p = torch.randn(10, 2, device=device)
    state = {"x": x, "p": p}
    step_size = 0.01
    mass = 2.0

    result = integrator.step(state, drift=lambda x_, t_: -gaussian_model.gradient(x_), step_size=step_size, mass=mass)

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

    result = integrator.step(state, drift=lambda x_, t_: -gaussian_model.gradient(x_), step_size=step_size, mass=mass)

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
        state, drift=lambda x_, t_: -gaussian_model.gradient(x_), step_size=step_size, n_steps=n_steps
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
        integrator.integrate(state, drift=lambda x_, t_: -gaussian_model.gradient(x_), step_size=0.01, n_steps=0)

    with pytest.raises(ValueError, match="n_steps must be positive"):
        integrator.integrate(state, drift=lambda x_, t_: -gaussian_model.gradient(x_), step_size=0.01, n_steps=-5)


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
        step_size=step_size,
        n_steps=n_steps,
        drift=lambda x_, t_: -potential_grad(x_),
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
        state, step_size=step_size, drift=lambda x_, t_: -potential_grad(x_)
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
        state, step_size=step_size, mass=mass, drift=lambda x_, t_: -potential_grad(x_)
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
        step_size=step_size,
        n_steps=n_steps,
        mass=mass,
        drift=lambda x_, t_: -potential_grad(x_),
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
        step_size=step_size,
        n_steps=n_steps,
        drift=lambda x_, t_: -potential_grad(x_),
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
    result_x_plus = integrator.step(state_x_plus, step_size=step_size, drift=lambda x_, t_: -potential_grad(x_))
    result_x_minus = integrator.step(state_x_minus, step_size=step_size, drift=lambda x_, t_: -potential_grad(x_))

    dx_dx = (result_x_plus["x"] - result_x_minus["x"]) / (2 * eps)
    dp_dx = (result_x_plus["p"] - result_x_minus["p"]) / (2 * eps)

    # Perturb p
    state_p_plus = {"x": x0.clone(), "p": p0 + eps}
    state_p_minus = {"x": x0.clone(), "p": p0 - eps}
    result_p_plus = integrator.step(state_p_plus, step_size=step_size, drift=lambda x_, t_: -potential_grad(x_))
    result_p_minus = integrator.step(state_p_minus, step_size=step_size, drift=lambda x_, t_: -potential_grad(x_))

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
        step_size=step_size,
        n_steps=n_steps,
        drift=lambda x_, t_: -potential_grad(x_),
    )

    # Negate momentum and integrate again (backwards in time)
    state_reversed = {"x": forward_result["x"], "p": -forward_result["p"]}
    backward_result = integrator.integrate(
        state_reversed,
        step_size=step_size,
        n_steps=n_steps,
        drift=lambda x_, t_: -potential_grad(x_),
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
        step_size=step_size,
        n_steps=n_steps,
        drift=lambda x_, t_: -potential_grad(x_),
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
        state, drift=lambda x_, t_: -double_well_model.gradient(x_), step_size=step_size, n_steps=n_steps
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

    result = integrator.step(state, drift=lambda x_, t_: -model.gradient(x_), step_size=0.1, safe=True)

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

    result1 = integrator.step({"x": x1, "p": p1}, drift=lambda x_, t_: -gaussian_model.gradient(x_), step_size=0.01)
    result2 = integrator.step({"x": x2, "p": p2}, drift=lambda x_, t_: -gaussian_model.gradient(x_), step_size=0.01)

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

    result = integrator.step(state, step_size=0.01, drift=lambda x_, t_: -potential_grad(x_))

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
        state, step_size=0.01, drift=lambda x_, t_: -potential_grad(x_)
    )

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_small_step_size(integrator, gaussian_model):
    """Test with very small step size."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    p = torch.randn(10, 2, device=device)
    state = {"x": x, "p": p}

    result = integrator.step(state, drift=lambda x_, t_: -gaussian_model.gradient(x_), step_size=1e-8)

    # With tiny step, state should barely change
    assert torch.allclose(result["x"], x, atol=1e-5)


def test_large_step_size(integrator, gaussian_model):
    """Test stability with large step size."""
    device = integrator.device
    x = torch.randn(10, 2, device=device)
    p = torch.randn(10, 2, device=device)
    state = {"x": x, "p": p}

    result = integrator.step(state, drift=lambda x_, t_: -gaussian_model.gradient(x_), step_size=1.0)

    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


################################ GeneralisedLeapfrogIntegrator Tests ###########################################
#
# Non-separable Hamiltonian used by several tests below:
#     M(x)   = 1 + x^2        (per-coordinate, treated diagonally)
#     U(x)   = 1/2 * x^2
#     H(x,p) = U(x) + 1/2 * p^2 / M(x) + 1/2 * log M(x)
#
# Hamilton's equations:
#     velocity = ∂H/∂p = p / M(x)
#     force    = -∂H/∂x = -[x + 1/2 * p^2 * dM^{-1}/dx + 1/2 * d log M / dx]


def _nonsep_force(x, p, t):
    inv_M = 1.0 / (1.0 + x ** 2)
    dU_dx = x
    dKinv_dx = -2.0 * x * inv_M ** 2
    dlogdet_dx = 2.0 * x * inv_M
    return -(dU_dx + 0.5 * p ** 2 * dKinv_dx + 0.5 * dlogdet_dx)


def _nonsep_velocity(x, p, t):
    return p / (1.0 + x ** 2)


def _nonsep_hamiltonian(x, p):
    M = 1.0 + x ** 2
    return 0.5 * (x ** 2).sum() + 0.5 * (p ** 2 / M).sum() + 0.5 * torch.log(M).sum()


@pytest.fixture
def gli_integrator():
    """Default GeneralisedLeapfrogIntegrator."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return GeneralisedLeapfrogIntegrator(device=device, dtype=torch.float32)


# ----- Initialization -----


def test_gli_initialization():
    integrator = GeneralisedLeapfrogIntegrator()
    assert isinstance(integrator, GeneralisedLeapfrogIntegrator)
    assert integrator.solver_max_iter == 8
    assert integrator.solver_check_every == 0


def test_gli_initialization_with_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    integrator = GeneralisedLeapfrogIntegrator(device=device, dtype=torch.float32)
    assert integrator.device == torch.device(device)
    assert integrator.dtype == torch.float32


def test_gli_initialization_invalid_solver_max_iter():
    with pytest.raises(ValueError, match="solver_max_iter must be >= 1"):
        GeneralisedLeapfrogIntegrator(solver_max_iter=0)


@requires_cuda
def test_gli_cuda():
    integrator = GeneralisedLeapfrogIntegrator(device="cuda", dtype=torch.float32)
    assert integrator.device == torch.device("cuda")


# ----- Validation -----


def test_gli_step_requires_force_and_velocity(gli_integrator):
    device = gli_integrator.device
    state = {"x": torch.randn(4, 2, device=device), "p": torch.randn(4, 2, device=device)}
    with pytest.raises(ValueError, match="Both"):
        gli_integrator.step(state, step_size=0.01,
                            force=lambda x, p, t: -x)
    with pytest.raises(ValueError, match="Both"):
        gli_integrator.step(state, step_size=0.01,
                            velocity=lambda x, p, t: p)


def test_gli_integrate_invalid_n_steps(gli_integrator):
    device = gli_integrator.device
    state = {"x": torch.randn(4, 2, device=device), "p": torch.randn(4, 2, device=device)}
    with pytest.raises(ValueError, match="n_steps must be positive"):
        gli_integrator.integrate(
            state, step_size=0.01, n_steps=0,
            force=lambda x, p, t: -x, velocity=lambda x, p, t: p,
        )
    with pytest.raises(ValueError, match="n_steps must be positive"):
        gli_integrator.integrate(
            state, step_size=0.01, n_steps=-3,
            force=lambda x, p, t: -x, velocity=lambda x, p, t: p,
        )


# ----- Equivalence to standard leapfrog (separable case) -----


def test_gli_separable_matches_standard_leapfrog_step():
    """For H = U(x) + |p|^2/2 the GLI Picard solves converge in one
    iteration, so a single GLI step must equal a single LeapfrogIntegrator
    step to machine precision."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    std = LeapfrogIntegrator(device=device, dtype=torch.float64)
    gli = GeneralisedLeapfrogIntegrator(device=device, dtype=torch.float64,
                                        solver_max_iter=4)
    torch.manual_seed(0)
    x0 = torch.randn(8, 3, dtype=torch.float64, device=device)
    p0 = torch.randn(8, 3, dtype=torch.float64, device=device)

    r_std = std.step({"x": x0.clone(), "p": p0.clone()},
                     step_size=0.05, drift=lambda x, t: -x)
    r_gli = gli.step({"x": x0.clone(), "p": p0.clone()},
                     step_size=0.05,
                     force=lambda x, p, t: -x,
                     velocity=lambda x, p, t: p)

    assert torch.allclose(r_std["x"], r_gli["x"], atol=1e-12)
    assert torch.allclose(r_std["p"], r_gli["p"], atol=1e-12)


def test_gli_separable_matches_standard_leapfrog_integrate():
    """Same as above over many steps."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    std = LeapfrogIntegrator(device=device, dtype=torch.float64)
    gli = GeneralisedLeapfrogIntegrator(device=device, dtype=torch.float64,
                                        solver_max_iter=4)
    torch.manual_seed(0)
    x0 = torch.randn(8, 3, dtype=torch.float64, device=device)
    p0 = torch.randn(8, 3, dtype=torch.float64, device=device)

    r_std = std.integrate({"x": x0.clone(), "p": p0.clone()},
                          step_size=0.05, n_steps=50,
                          drift=lambda x, t: -x)
    r_gli = gli.integrate({"x": x0.clone(), "p": p0.clone()},
                          step_size=0.05, n_steps=50,
                          force=lambda x, p, t: -x,
                          velocity=lambda x, p, t: p)

    assert torch.allclose(r_std["x"], r_gli["x"], atol=1e-12)
    assert torch.allclose(r_std["p"], r_gli["p"], atol=1e-12)


# ----- Manual single-step trace -----


def test_gli_manual_step_separable():
    """GLI on H = 0.5*x^2 + 0.5*p^2 reduces to standard leapfrog;
    trace the algebra explicitly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gli = GeneralisedLeapfrogIntegrator(device=device, dtype=torch.float64,
                                        solver_max_iter=4)
    x = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float64)
    p = torch.tensor([[0.5, -0.5]], device=device, dtype=torch.float64)
    eps = 0.1

    result = gli.step({"x": x.clone(), "p": p.clone()}, step_size=eps,
                      force=lambda x_, p_, t_: -x_,
                      velocity=lambda x_, p_, t_: p_)

    # Manual leapfrog:
    p_half = p - 0.5 * eps * x
    x_new = x + eps * p_half
    p_new = p_half - 0.5 * eps * x_new

    assert torch.allclose(result["x"], x_new, atol=1e-12)
    assert torch.allclose(result["p"], p_new, atol=1e-12)


# ----- Non-separable: energy conservation -----


def test_gli_non_separable_energy_conservation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gli = GeneralisedLeapfrogIntegrator(device=device, dtype=torch.float64,
                                        solver_max_iter=8)
    x0 = torch.tensor([[1.0, -0.5]], dtype=torch.float64, device=device)
    p0 = torch.tensor([[0.3, 0.7]], dtype=torch.float64, device=device)
    E0 = _nonsep_hamiltonian(x0, p0)

    res = gli.integrate({"x": x0.clone(), "p": p0.clone()},
                        step_size=0.02, n_steps=200,
                        force=_nonsep_force, velocity=_nonsep_velocity)
    E1 = _nonsep_hamiltonian(res["x"], res["p"])

    relative_error = (E1 - E0).abs() / E0.abs()
    assert relative_error < 1e-3, f"GLI energy drift = {relative_error.item()}"


# ----- Non-separable: reversibility -----


def test_gli_non_separable_reversibility():
    """Forward then (negate p) forward should recover the initial state."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gli = GeneralisedLeapfrogIntegrator(device=device, dtype=torch.float64,
                                        solver_max_iter=12,
                                        solver_check_every=2,
                                        solver_tol=1e-12)
    x0 = torch.tensor([[1.0, -0.5]], dtype=torch.float64, device=device)
    p0 = torch.tensor([[0.3, 0.7]], dtype=torch.float64, device=device)

    fwd = gli.integrate({"x": x0.clone(), "p": p0.clone()},
                        step_size=0.02, n_steps=100,
                        force=_nonsep_force, velocity=_nonsep_velocity)
    bwd = gli.integrate({"x": fwd["x"].clone(), "p": -fwd["p"].clone()},
                        step_size=0.02, n_steps=100,
                        force=_nonsep_force, velocity=_nonsep_velocity)

    assert torch.allclose(bwd["x"], x0, atol=1e-8)
    assert torch.allclose(bwd["p"], -p0, atol=1e-8)


# ----- Symplecticity (phase-space volume preservation) -----


def test_gli_symplecticity_phase_space_volume():
    """1-D check that the GLI step has Jacobian determinant ≈ 1.

    Uses the separable harmonic oscillator so the Picard solves are exact
    (one iteration is a fixed point), isolating the symplecticity of the
    GLI scheme itself.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gli = GeneralisedLeapfrogIntegrator(device=device, dtype=torch.float64,
                                        solver_max_iter=8)
    force = lambda x_, p_, t_: -x_
    velocity = lambda x_, p_, t_: p_

    x0 = torch.tensor([[1.0]], device=device, dtype=torch.float64)
    p0 = torch.tensor([[0.5]], device=device, dtype=torch.float64)
    step_size = 0.1
    eps = 1e-5

    def step(x, p):
        out = gli.step({"x": x, "p": p}, step_size=step_size,
                       force=force, velocity=velocity)
        return out["x"], out["p"]

    xp, pp = step(x0 + eps, p0.clone())
    xm, pm = step(x0 - eps, p0.clone())
    dx_dx = (xp - xm) / (2 * eps)
    dp_dx = (pp - pm) / (2 * eps)

    xp, pp = step(x0.clone(), p0 + eps)
    xm, pm = step(x0.clone(), p0 - eps)
    dx_dp = (xp - xm) / (2 * eps)
    dp_dp = (pp - pm) / (2 * eps)

    det_J = dx_dx * dp_dp - dx_dp * dp_dx
    assert torch.allclose(det_J, torch.ones_like(det_J), atol=1e-6), \
        f"GLI Jacobian det = {det_J.item()}"


# ----- Solver options -----


def test_gli_solver_check_every_early_exit_matches_fixed_iter():
    """With a sufficiently tight tolerance and large solver_max_iter, the
    early-exit Picard path must agree with the fixed-iteration path to
    high precision."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fixed = GeneralisedLeapfrogIntegrator(device=device, dtype=torch.float64,
                                          solver_max_iter=20)
    early = GeneralisedLeapfrogIntegrator(device=device, dtype=torch.float64,
                                          solver_max_iter=50,
                                          solver_check_every=1,
                                          solver_tol=1e-12)
    x0 = torch.tensor([[0.5, -0.3]], dtype=torch.float64, device=device)
    p0 = torch.tensor([[0.2, 0.1]], dtype=torch.float64, device=device)

    r_fixed = fixed.integrate({"x": x0.clone(), "p": p0.clone()},
                              step_size=0.02, n_steps=20,
                              force=_nonsep_force, velocity=_nonsep_velocity)
    r_early = early.integrate({"x": x0.clone(), "p": p0.clone()},
                              step_size=0.02, n_steps=20,
                              force=_nonsep_force, velocity=_nonsep_velocity)

    assert torch.allclose(r_fixed["x"], r_early["x"], atol=1e-10)
    assert torch.allclose(r_fixed["p"], r_early["p"], atol=1e-10)


# ----- Inference mode -----


def test_gli_inference_mode_matches_normal():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gli = GeneralisedLeapfrogIntegrator(device=device, dtype=torch.float64)

    torch.manual_seed(0)
    x0 = torch.randn(4, 2, dtype=torch.float64, device=device)
    p0 = torch.randn(4, 2, dtype=torch.float64, device=device)

    r_normal = gli.integrate({"x": x0.clone(), "p": p0.clone()},
                             step_size=0.05, n_steps=10,
                             force=_nonsep_force, velocity=_nonsep_velocity,
                             inference_mode=False)
    r_inf = gli.integrate({"x": x0.clone(), "p": p0.clone()},
                          step_size=0.05, n_steps=10,
                          force=_nonsep_force, velocity=_nonsep_velocity,
                          inference_mode=True)
    assert torch.allclose(r_normal["x"], r_inf["x"])
    assert torch.allclose(r_normal["p"], r_inf["p"])


# ----- Safe mode -----


def test_gli_safe_mode_handles_nan():
    """safe=True should sanitise NaN forces and clamp wild proposals."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gli = GeneralisedLeapfrogIntegrator(device=device, dtype=torch.float32)

    def bad_force(x, p, t):
        f = -x
        mask = torch.norm(x, dim=-1, keepdim=True) > 5.0
        return torch.where(mask.expand_as(f), torch.full_like(f, float("nan")), f)

    x = torch.ones(5, 2, device=device) * 4.9
    p = torch.randn(5, 2, device=device)
    result = gli.step({"x": x, "p": p}, step_size=0.1,
                      force=bad_force, velocity=lambda x_, p_, t_: p_,
                      safe=True)
    assert torch.all(torch.isfinite(result["x"]))
    assert torch.all(torch.isfinite(result["p"]))


# ----- Edge cases -----


def test_gli_large_batch_size(gli_integrator):
    device = gli_integrator.device
    x = torch.randn(1000, 10, device=device)
    p = torch.randn(1000, 10, device=device)
    result = gli_integrator.step({"x": x, "p": p}, step_size=0.01,
                                 force=lambda x_, p_, t_: -x_,
                                 velocity=lambda x_, p_, t_: p_)
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_gli_high_dimension(gli_integrator):
    device = gli_integrator.device
    dim = 100
    x = torch.randn(10, dim, device=device)
    p = torch.randn(10, dim, device=device)
    result = gli_integrator.step({"x": x, "p": p}, step_size=0.01,
                                 force=lambda x_, p_, t_: -x_,
                                 velocity=lambda x_, p_, t_: p_)
    assert result["x"].shape == x.shape
    assert torch.all(torch.isfinite(result["x"]))


def test_gli_reproducibility(gli_integrator):
    """Deterministic given identical inputs."""
    device = gli_integrator.device
    torch.manual_seed(42)
    x1 = torch.randn(10, 2, device=device)
    p1 = torch.randn(10, 2, device=device)
    torch.manual_seed(42)
    x2 = torch.randn(10, 2, device=device)
    p2 = torch.randn(10, 2, device=device)

    r1 = gli_integrator.step({"x": x1, "p": p1}, step_size=0.01,
                             force=lambda x_, p_, t_: -x_,
                             velocity=lambda x_, p_, t_: p_)
    r2 = gli_integrator.step({"x": x2, "p": p2}, step_size=0.01,
                             force=lambda x_, p_, t_: -x_,
                             velocity=lambda x_, p_, t_: p_)
    assert torch.allclose(r1["x"], r2["x"])
    assert torch.allclose(r1["p"], r2["p"])
