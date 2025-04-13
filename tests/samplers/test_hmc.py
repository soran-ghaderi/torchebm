import pytest
import torch
import numpy as np
from torchebm.core.base_energy_function import (
    BaseEnergyFunction,
    GaussianEnergy,
    DoubleWellEnergy,
)
from torchebm.samplers.hmc import HamiltonianMonteCarlo
from tests.conftest import requires_cuda


@pytest.fixture
def energy_function(request):
    """Fixture to create various energy functions for testing."""
    if not hasattr(request, "param"):
        return GaussianEnergy(mean=torch.zeros(10), cov=torch.eye(10))

    # Use parameters when provided through parametrize
    params = request.param
    if params.get("type") == "gaussian":
        mean = params.get("mean", torch.zeros(10))
        cov = params.get("cov", torch.eye(10))
        return GaussianEnergy(mean=mean, cov=cov)
    elif params.get("type") == "double_well":
        barrier_height = params.get("barrier_height", 2.0)
        return DoubleWellEnergy(barrier_height=barrier_height)
    else:
        # Default to Gaussian
        return GaussianEnergy(mean=torch.zeros(10), cov=torch.eye(10))


@pytest.fixture
def hmc_sampler(request, energy_function):
    """Fixture to create an HMC sampler with various configurations."""
    if not hasattr(request, "param"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HamiltonianMonteCarlo(
            energy_function=energy_function,
            step_size=0.1,
            n_leapfrog_steps=10,
            device=device,
        )

    # Use parameters when provided
    params = request.param
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Configure step_size and n_leapfrog_steps
    step_size = params.get("step_size", 0.1)
    n_leapfrog_steps = params.get("n_leapfrog_steps", 10)

    # Configure mass (optional)
    mass = params.get("mass", None)

    # Ensure energy function is on the correct device
    energy_function = energy_function.to(device)

    return HamiltonianMonteCarlo(
        energy_function=energy_function,
        step_size=step_size,
        n_leapfrog_steps=n_leapfrog_steps,
        mass=mass,
        device=device,
    )


def test_hmc_initialization(hmc_sampler):
    """Test basic initialization of the HMC sampler."""
    assert isinstance(hmc_sampler, HamiltonianMonteCarlo)
    assert hmc_sampler.step_size == 0.1
    assert hmc_sampler.n_leapfrog_steps == 10
    assert hmc_sampler.mass is None


def test_hmc_initialization_with_mass():
    """Test HMC initialization with different mass configurations."""
    # With scalar mass
    energy_fn = GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2))
    scalar_mass = 2.0
    hmc = HamiltonianMonteCarlo(
        energy_function=energy_fn, step_size=0.1, n_leapfrog_steps=10, mass=scalar_mass
    )
    assert hmc.mass == scalar_mass

    # With tensor mass
    tensor_mass = torch.ones(2) * 3.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor_mass = tensor_mass.to(device)
    energy_fn = energy_fn.to(device)
    hmc = HamiltonianMonteCarlo(
        energy_function=energy_fn,
        step_size=0.1,
        n_leapfrog_steps=10,
        mass=tensor_mass,
        device=device,
    )
    assert torch.all(hmc.mass == tensor_mass)
    assert hmc.mass.device.type == device


def test_hmc_initialization_invalid_params(energy_function):
    """Test that invalid parameters raise appropriate exceptions."""
    with pytest.raises(ValueError):
        HamiltonianMonteCarlo(energy_function, step_size=-0.1, n_leapfrog_steps=10)
    with pytest.raises(ValueError):
        HamiltonianMonteCarlo(energy_function, step_size=0.1, n_leapfrog_steps=0)


@pytest.mark.parametrize(
    "energy_function, hmc_sampler",
    [
        (
            {"type": "gaussian", "mean": torch.zeros(2), "cov": torch.eye(2)},
            {"device": "cuda" if torch.cuda.is_available() else "cpu"},
        ),
        (
            {"type": "gaussian", "mean": torch.zeros(2), "cov": 2 * torch.eye(2)},
            {"device": "cuda" if torch.cuda.is_available() else "cpu"},
        ),
    ],
    indirect=True,  # This tells pytest to apply parameters to fixtures
)
def test_hmc_sample_chain_basic(hmc_sampler):
    """Test basic sampling functionality."""
    dim = 2
    n_steps = 50
    final_state = hmc_sampler.sample_chain(dim=dim, n_steps=n_steps)

    # Check output shape and validity
    assert final_state.shape == (1, dim)  # (n_samples, dim)
    assert torch.all(torch.isfinite(final_state))


@pytest.mark.parametrize(
    "energy_function, hmc_sampler",
    [
        (
            {"type": "gaussian", "mean": torch.zeros(2), "cov": torch.eye(2)},
            {"device": "cuda" if torch.cuda.is_available() else "cpu"},
        ),
    ],
    indirect=True,
)
def test_hmc_sample_chain_with_trajectory(hmc_sampler):
    """Test sampling with trajectory return."""
    dim = 2
    n_steps = 50
    trajectory = hmc_sampler.sample_chain(
        dim=dim, n_steps=n_steps, return_trajectory=True
    )

    # Check trajectory shape and validity
    assert trajectory.shape == (1, n_steps, dim)  # (n_samples, n_steps, dim)
    assert torch.all(torch.isfinite(trajectory))


@pytest.mark.parametrize(
    "energy_function, hmc_sampler",
    [
        (
            {"type": "gaussian", "mean": torch.zeros(2), "cov": torch.eye(2)},
            {"device": "cuda" if torch.cuda.is_available() else "cpu"},
        ),
    ],
    indirect=True,
)
def test_hmc_sample_chain_with_diagnostics(hmc_sampler):
    """Test sampling with diagnostics return."""
    dim = 2
    n_steps = 50
    final_state, diagnostics = hmc_sampler.sample_chain(
        dim=dim, n_steps=n_steps, return_diagnostics=True
    )

    # Check diagnostics shape and contents
    assert final_state.shape == (1, dim)
    assert diagnostics.shape == (
        n_steps,
        4,
        1,
        dim,
    )  # (n_steps, n_diagnostics, n_samples, dim)
    assert torch.all(torch.isfinite(diagnostics))


@pytest.mark.parametrize(
    "energy_function, hmc_sampler",
    [
        (
            {"type": "gaussian", "mean": torch.zeros(2), "cov": torch.eye(2)},
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "n_leapfrog_steps": 5,
            },
        ),
    ],
    indirect=True,
)
def test_hmc_sample_chain_multiple_samples(hmc_sampler):
    """Test sampling with multiple parallel chains."""
    dim = 2
    n_steps = 50
    n_samples = 10
    samples = hmc_sampler.sample_chain(dim=dim, n_steps=n_steps, n_samples=n_samples)

    # Check output shape and validity for multiple samples
    assert samples.shape == (n_samples, dim)
    assert torch.all(torch.isfinite(samples))


@pytest.mark.parametrize(
    "energy_function, hmc_sampler",
    [
        (
            {"type": "gaussian", "mean": torch.zeros(2), "cov": torch.eye(2)},
            {"device": "cuda" if torch.cuda.is_available() else "cpu"},
        ),
    ],
    indirect=True,
)
def test_hmc_reproducibility(hmc_sampler):
    """Test that sampling with the same seed produces the same results."""
    torch.manual_seed(42)
    dim = 2
    n_steps = 50
    result1 = hmc_sampler.sample_chain(dim=dim, n_steps=n_steps)

    torch.manual_seed(42)
    result2 = hmc_sampler.sample_chain(dim=dim, n_steps=n_steps)

    # Should get identical results with the same seed
    assert torch.allclose(result1, result2)


@pytest.mark.parametrize(
    "hmc_sampler",
    [
        ({"device": "cpu"}),
        ({"device": "cuda" if torch.cuda.is_available() else "cpu"}),
    ],
    indirect=True,
)
def test_hmc_device_consistency(hmc_sampler):
    """Test that tensors are consistently on the correct device."""
    n_steps = 10
    device = hmc_sampler.device

    # Sample and check device - don't specify dim to use the inferred dimension
    samples = hmc_sampler.sample_chain(n_steps=n_steps)
    assert samples.device.type == device.type

    # Test with custom input and check device consistency
    if hasattr(hmc_sampler.energy_function, "mean"):
        dim = hmc_sampler.energy_function.mean.shape[0]
    else:
        dim = 2  # Fallback dimension for other energy functions
    x_init = torch.randn(5, dim, device=device)
    samples = hmc_sampler.sample_chain(x=x_init, n_steps=n_steps)
    assert samples.device.type == device.type


@pytest.mark.parametrize(
    "energy_function, hmc_sampler",
    [
        (
            {"type": "double_well", "barrier_height": 2.0},
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "step_size": 0.05,
                "n_leapfrog_steps": 20,
            },
        ),
    ],
    indirect=True,
)
def test_hmc_with_double_well(hmc_sampler):
    """Test HMC on a more complex energy landscape (double well potential)."""
    dim = 2
    n_steps = 100
    n_samples = 50
    samples = hmc_sampler.sample_chain(dim=dim, n_steps=n_steps, n_samples=n_samples)

    # Check basic properties
    assert samples.shape == (n_samples, dim)
    assert torch.all(torch.isfinite(samples))

    # For double well, samples should cluster around (-1,-1) and (1,1)
    # This is a probabilistic test, so it might occasionally fail even with correct implementation
    # We check that there are samples near both modes
    # May need to adjust the threshold for reliable detection
    near_neg1 = torch.sum(torch.all(torch.abs(samples - (-1)) < 0.5, dim=1))
    near_pos1 = torch.sum(torch.all(torch.abs(samples - 1) < 0.5, dim=1))

    # There should be at least some samples near each mode
    assert near_neg1 > 0, "No samples found near the (-1,-1) mode"
    assert near_pos1 > 0, "No samples found near the (1,1) mode"


@pytest.mark.parametrize(
    "energy_function, hmc_sampler",
    [
        (
            {"type": "gaussian", "mean": torch.zeros(2), "cov": torch.eye(2)},
            {"device": "cuda" if torch.cuda.is_available() else "cpu", "mass": 2.0},
        ),
    ],
    indirect=True,
)
def test_hmc_with_scalar_mass(hmc_sampler):
    """Test HMC with scalar mass parameter."""
    dim = 2
    n_steps = 50
    samples = hmc_sampler.sample_chain(dim=dim, n_steps=n_steps)

    assert samples.shape == (1, dim)
    assert torch.all(torch.isfinite(samples))
    assert hmc_sampler.mass == 2.0


@pytest.mark.parametrize(
    "energy_function, hmc_sampler",
    [
        (
            {"type": "gaussian", "mean": torch.zeros(2), "cov": torch.eye(2)},
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "mass": torch.tensor(
                    [2.0, 3.0], device="cuda" if torch.cuda.is_available() else "cpu"
                ),
            },
        ),
    ],
    indirect=True,
)
def test_hmc_with_tensor_mass(hmc_sampler):
    """Test HMC with tensor mass parameter."""
    dim = 2
    n_steps = 50
    samples = hmc_sampler.sample_chain(dim=dim, n_steps=n_steps)

    assert samples.shape == (1, dim)
    assert torch.all(torch.isfinite(samples))
    assert torch.allclose(
        hmc_sampler.mass, torch.tensor([2.0, 3.0], device=hmc_sampler.device)
    )


def test_hmc_step_internals():
    """Test internal HMC step components."""
    # Create a simple energy function
    energy_fn = GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2))

    # Initialize HMC sampler with fixed parameters for deterministic testing
    hmc = HamiltonianMonteCarlo(
        energy_function=energy_fn, step_size=0.1, n_leapfrog_steps=5
    )

    # Test momentum initialization
    torch.manual_seed(123)
    shape = torch.Size([10, 2])
    momentum = hmc._initialize_momentum(shape)
    assert momentum.shape == shape

    # Test kinetic energy calculation
    kinetic = hmc._compute_kinetic_energy(momentum)
    assert kinetic.shape == torch.Size([10])
    assert torch.all(kinetic >= 0)  # Kinetic energy should be non-negative

    # Test leapfrog step
    position = torch.zeros(10, 2)
    new_position, new_momentum = hmc._leapfrog_integration(position, momentum)
    assert new_position.shape == position.shape
    assert new_momentum.shape == momentum.shape

    # Test HMC step (single iteration)
    torch.manual_seed(123)
    position = torch.zeros(10, 2)
    new_position, acceptance_prob, accepted = hmc.hmc_step(position)
    assert new_position.shape == position.shape
    assert acceptance_prob.shape == torch.Size([10])
    assert accepted.shape == torch.Size([10])

    # Acceptance probabilities should be between 0 and 1
    assert torch.all(acceptance_prob >= 0) and torch.all(acceptance_prob <= 1)


def test_hmc_gaussian_sampling_statistics():
    """Test statistical properties of HMC samples from a Gaussian distribution."""
    # Skip if no GPU to avoid slow CPU tests
    if not torch.cuda.is_available():
        pytest.skip("Skipping statistical test on CPU for speed")

    # Create a 2D Gaussian with known parameters
    mean = torch.tensor([1.0, -1.0], device="cuda")
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]], device="cuda")
    energy_fn = GaussianEnergy(mean=mean, cov=cov)

    # Initialize HMC sampler
    hmc = HamiltonianMonteCarlo(
        energy_function=energy_fn, step_size=0.05, n_leapfrog_steps=10, device="cuda"
    )

    # Run HMC for many steps to collect statistics
    n_samples = 1000
    n_steps = 200  # Run for more steps to ensure good mixing
    samples = hmc.sample_chain(n_samples=n_samples, n_steps=n_steps, dim=2)

    # Compute sample mean and covariance
    sample_mean = samples.mean(dim=0)
    centered = samples - sample_mean
    sample_cov = torch.matmul(centered.t(), centered) / (n_samples - 1)

    # Check that sample statistics are close to population parameters
    # Use relatively wide tolerance since this is a statistical test
    assert torch.allclose(sample_mean, mean, rtol=0.2, atol=0.2)
    assert torch.allclose(sample_cov, cov, rtol=0.3, atol=0.3)


def test_hmc_small_step_size():
    """Test HMC with a very small step size."""
    energy_fn = GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2))

    # Use a very small step size
    tiny_step = 1e-5
    hmc = HamiltonianMonteCarlo(
        energy_function=energy_fn, step_size=tiny_step, n_leapfrog_steps=10
    )

    # Sample and ensure we get valid results
    samples = hmc.sample_chain(dim=2, n_steps=20, n_samples=5)
    assert samples.shape == (5, 2)
    assert torch.all(torch.isfinite(samples))


def test_hmc_large_leapfrog_steps():
    """Test HMC with a large number of leapfrog steps."""
    energy_fn = GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2))

    # Use many leapfrog steps
    many_steps = 100
    hmc = HamiltonianMonteCarlo(
        energy_function=energy_fn,
        step_size=0.01,  # Small step size to avoid numerical issues
        n_leapfrog_steps=many_steps,
    )

    # Sample and ensure we get valid results
    samples = hmc.sample_chain(dim=2, n_steps=20, n_samples=5)
    assert samples.shape == (5, 2)
    assert torch.all(torch.isfinite(samples))


@requires_cuda
def test_hmc_high_dimensions():
    """Test HMC in high-dimensional spaces."""
    # Create a high-dimensional Gaussian
    dim = 100
    energy_fn = GaussianEnergy(mean=torch.zeros(dim), cov=torch.eye(dim))

    # Initialize HMC sampler with suitable parameters for high dimensions
    hmc = HamiltonianMonteCarlo(
        energy_function=energy_fn,
        step_size=0.01,
        n_leapfrog_steps=20,
        device="cuda",  # Use GPU for this test
    )

    # Sample and ensure we get valid results
    samples = hmc.sample_chain(dim=dim, n_steps=50, n_samples=10)
    assert samples.shape == (10, dim)
    assert torch.all(torch.isfinite(samples))

    # Simple test that samples are distributed around the mean (0)
    # In high dimensions with few samples, some dimensions may have larger deviations
    # Check that the overall mean is close to zero, and most dimensions are within tolerance
    sample_means = samples.mean(dim=0)

    # Check overall mean is close to zero
    assert torch.abs(sample_means.mean()) < 0.1, "Overall mean should be close to zero"

    # Check that at least 90% of dimensions have means within ±1.0
    within_tolerance = (torch.abs(sample_means) < 1.0).float().mean()
    assert (
        within_tolerance >= 0.9
    ), f"Only {within_tolerance:.2f} of dimensions within tolerance"


def test_hmc_custom_initial_state():
    """Test HMC with specific custom initial state."""
    energy_fn = GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2))
    hmc = HamiltonianMonteCarlo(
        energy_function=energy_fn, step_size=0.1, n_leapfrog_steps=10
    )

    # Create a specific initial state far from mean
    initial_state = torch.tensor([[10.0, -10.0]], device=hmc.device)

    # Sample and check that we move toward the mean
    samples = hmc.sample_chain(x=initial_state, n_steps=200)

    # After many steps, should be closer to mean than starting point
    final_dist_to_mean = torch.norm(samples)
    initial_dist_to_mean = torch.norm(initial_state)
    assert final_dist_to_mean < initial_dist_to_mean


def test_hmc_numerical_stability():
    """Test HMC numerical stability with extreme values."""
    energy_fn = GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2))
    hmc = HamiltonianMonteCarlo(
        energy_function=energy_fn, step_size=0.1, n_leapfrog_steps=10
    )

    # Extreme initial position
    extreme_position = torch.tensor([[1e5, 1e5]], device=hmc.device)

    # This should not produce NaN or infinite values
    result = hmc.sample_chain(x=extreme_position, n_steps=10)
    assert torch.all(torch.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__])
