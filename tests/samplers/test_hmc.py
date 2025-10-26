import pytest
import torch
import numpy as np
from functools import partial

from torchebm.core import (
    BaseModel,
    GaussianModel,
    DoubleWellModel,
    BaseScheduler,
    ConstantScheduler,
    LinearScheduler,
)
from torchebm.samplers import HamiltonianMonteCarlo
from tests.conftest import requires_cuda as rc

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


###############################################################################
# Test fixtures
###############################################################################


class NanInducingEnergy(BaseModel):
    """Energy function designed to produce NaNs during gradient calculation."""

    def __init__(self, dim=2, nan_threshold=5.0, device="cpu"):
        super().__init__()
        self.dim = dim
        self.nan_threshold = nan_threshold
        self._device = torch.device(device)
        # Move a parameter to the device to satisfy BaseSampler's potential device checks
        self.dummy_param = torch.nn.Parameter(torch.zeros(1, device=self._device))

    @property
    def device(self):
        return self._device

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        # Simple quadratic energy, but gradient will cause issues
        return 0.5 * torch.sum(x**2, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward method to satisfy BaseEnergyFunction contract
        return self.energy(x)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        # Induce NaN if norm is above threshold
        norm = torch.norm(x, dim=-1, keepdim=True)
        grad = x.clone()
        # Create NaN values where the condition is met
        mask = norm > self.nan_threshold
        grad[mask.expand_as(grad)] = torch.nan
        return grad

    def to(self, device):
        self._device = torch.device(device)
        self.dummy_param = self.dummy_param.to(self._device)
        return self


@pytest.fixture
def energy_function(request):
    """Fixture to create various energy functions for testing."""
    params = getattr(request, "param", {})  # Use getattr for default case
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    if params.get("type") == "gaussian":
        dim = params.get("dim", 10)
        mean = params.get("mean", torch.zeros(dim, device=device))
        cov = params.get("cov", torch.eye(dim, device=device))
        return GaussianModel(mean=mean, cov=cov).to(device)
    elif params.get("type") == "double_well":
        dim = params.get("dim", 2)  # DoubleWell usually tested in 2D
        barrier_height = params.get("barrier_height", 2.0)
        return DoubleWellModel(barrier_height=barrier_height).to(device)
    elif params.get("type") == "nan_inducing":
        dim = params.get("dim", 2)
        nan_threshold = params.get("nan_threshold", 5.0)
        return NanInducingEnergy(
            dim=dim, nan_threshold=nan_threshold, device=device
        ).to(device)
    else:
        # Default to Gaussian
        dim = params.get("dim", 10)
        return GaussianModel(mean=torch.zeros(dim), cov=torch.eye(dim)).to(device)


@pytest.fixture
def hmc_sampler(request, energy_function):
    """Fixture to create an HMC sampler with various configurations."""
    params = getattr(request, "param", {})
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Configure step_size (can be float or scheduler)
    step_size_config = params.get("step_size", 0.1)
    if isinstance(step_size_config, dict) and step_size_config.get("type") == "linear":
        # Example: Use LinearScheduler if specified
        start_value = step_size_config.get("start", 0.1)
        end_value = step_size_config.get("end", 0.01)
        num_steps = step_size_config.get("steps", 100)
        step_size = LinearScheduler(
            start_value=start_value, end_value=end_value, n_steps=num_steps
        )
    else:
        step_size = step_size_config  # Assume float otherwise

    n_leapfrog_steps = params.get("n_leapfrog_steps", 10)
    mass = params.get("mass", None)
    dtype_str = params.get("dtype", None)  # Allow specifying dtype for testing

    # Determine default dtype based on device if not specified
    if dtype_str is None:
        dtype = torch.float32
    elif dtype_str == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Handle potential CUDA-only float16 request
    if dtype == torch.float16 and device == "cpu":
        pytest.skip("float16 is only tested on CUDA devices.")

    # Move tensor mass to device if provided
    if isinstance(mass, torch.Tensor):
        mass = mass.to(device)

    return HamiltonianMonteCarlo(
        model=energy_function,
        step_size=step_size,
        n_leapfrog_steps=n_leapfrog_steps,
        mass=mass,
        dtype=dtype,
        device=device,
    )


###############################################################################
# Basic Initialization and Validation Tests
###############################################################################


def test_hmc_initialization(hmc_sampler):
    """Test basic initialization of the HMC sampler."""
    assert isinstance(hmc_sampler, HamiltonianMonteCarlo)
    # Check scheduler registration
    assert hmc_sampler.schedulers["step_size"] is not None
    assert isinstance(hmc_sampler.schedulers["step_size"], BaseScheduler)
    assert hmc_sampler.n_leapfrog_steps > 0


def test_hmc_initialization_with_mass():
    """Test HMC initialization with different mass configurations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_fn = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2)).to(device)

    # With scalar mass
    scalar_mass = 2.0
    hmc_scalar = HamiltonianMonteCarlo(
        model=energy_fn,
        step_size=0.1,
        n_leapfrog_steps=10,
        mass=scalar_mass,
        device=device,
    )
    assert hmc_scalar.mass == scalar_mass
    assert isinstance(hmc_scalar.mass, float)

    # With tensor mass
    tensor_mass = torch.tensor([2.0, 3.0], device=device)
    hmc_tensor = HamiltonianMonteCarlo(
        model=energy_fn,
        step_size=0.1,
        n_leapfrog_steps=10,
        mass=tensor_mass,
        device=device,
    )
    assert torch.all(hmc_tensor.mass == tensor_mass)
    assert isinstance(hmc_tensor.mass, torch.Tensor)
    assert hmc_tensor.mass.device.type == device


def test_hmc_initialization_invalid_params(energy_function):
    """Test that invalid parameters raise appropriate exceptions."""
    with pytest.raises(ValueError, match="step_size must be positive"):
        HamiltonianMonteCarlo(energy_function, step_size=-0.1, n_leapfrog_steps=10)

    with pytest.raises(ValueError, match="step_size must be positive"):
        HamiltonianMonteCarlo(energy_function, step_size=0.0, n_leapfrog_steps=10)

    with pytest.raises(ValueError, match="n_leapfrog_steps must be positive"):
        HamiltonianMonteCarlo(energy_function, step_size=0.1, n_leapfrog_steps=0)

    with pytest.raises(ValueError, match="n_leapfrog_steps must be positive"):
        HamiltonianMonteCarlo(energy_function, step_size=0.1, n_leapfrog_steps=-5)


###############################################################################
# Basic Sampling Tests
###############################################################################


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
    indirect=True,
)
def test_hmc_sample_basic(hmc_sampler):
    """Test basic sampling functionality."""
    dim = hmc_sampler.model.mean.shape[0]
    n_steps = 50
    final_state = hmc_sampler.sample(dim=dim, n_steps=n_steps)
    assert final_state.shape == (1, dim)
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
def test_hmc_sample_with_trajectory(hmc_sampler):
    """Test sampling with trajectory return."""
    dim = hmc_sampler.model.mean.shape[0]
    n_steps = 50
    trajectory = hmc_sampler.sample(dim=dim, n_steps=n_steps, return_trajectory=True)
    assert trajectory.shape == (1, n_steps, dim)
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
def test_hmc_sample_with_diagnostics(hmc_sampler):
    """Test sampling with diagnostics return."""
    dim = hmc_sampler.model.mean.shape[0]
    n_steps = 50
    final_state, diagnostics = hmc_sampler.sample(
        dim=dim, n_steps=n_steps, return_diagnostics=True
    )
    assert final_state.shape == (1, dim)
    # Shape: (n_steps, n_diagnostics=4, n_samples=1, dim)
    assert diagnostics.shape == (n_steps, 4, 1, dim)
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
def test_hmc_sample_multiple_samples(hmc_sampler):
    """Test sampling with multiple parallel chains."""
    dim = hmc_sampler.model.mean.shape[0]
    n_steps = 50
    n_samples = 10
    samples = hmc_sampler.sample(dim=dim, n_steps=n_steps, n_samples=n_samples)
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
    np.random.seed(42)
    dim = hmc_sampler.model.mean.shape[0]
    n_steps = 50
    result1 = hmc_sampler.sample(dim=dim, n_steps=n_steps)

    torch.manual_seed(42)
    np.random.seed(42)
    result2 = hmc_sampler.sample(dim=dim, n_steps=n_steps)

    assert torch.allclose(result1, result2, atol=1e-6)


@pytest.mark.parametrize(
    "energy_function, hmc_sampler",
    [
        (
            {"type": "double_well", "barrier_height": 2.0, "dim": 2},
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
    samples = hmc_sampler.sample(dim=dim, n_steps=n_steps, n_samples=n_samples)

    assert samples.shape == (n_samples, dim)
    assert torch.all(torch.isfinite(samples))

    # Probabilistic check for modes around (-1, -1) and (1, 1)
    center1 = -torch.ones(dim, device=samples.device)
    center2 = torch.ones(dim, device=samples.device)
    dist_to_center1 = torch.norm(samples - center1, dim=1)
    dist_to_center2 = torch.norm(samples - center2, dim=1)

    near_neg1 = torch.sum(dist_to_center1 < 0.7 * np.sqrt(dim))
    near_pos1 = torch.sum(dist_to_center2 < 0.7 * np.sqrt(dim))

    assert near_neg1 > 0, "No samples found near the (-1,...,-1) mode"
    assert near_pos1 > 0, "No samples found near the (1,...,1) mode"


###############################################################################
# Advanced Feature Tests (Schedulers, Device Support, Numeric Stability)
###############################################################################


@pytest.mark.parametrize(
    "energy_function, hmc_sampler",
    [
        (
            {"type": "gaussian", "dim": 2},
            # Configure sampler to use a LinearScheduler
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "step_size": {"type": "linear", "start": 0.1, "end": 0.01, "steps": 50},
                "n_leapfrog_steps": 5,
            },
        ),
    ],
    indirect=["energy_function", "hmc_sampler"],
)
def test_hmc_with_scheduler(hmc_sampler):
    """Test HMC initialization and sampling with a step size scheduler."""
    assert isinstance(hmc_sampler.schedulers["step_size"], LinearScheduler)

    # Check initial step size
    initial_step_size = hmc_sampler.get_scheduled_value("step_size")
    assert np.isclose(initial_step_size, 0.1)  # Start value

    # Run sampling
    dim = hmc_sampler.model.mean.shape[0]
    n_steps = 50
    final_state = hmc_sampler.sample(dim=dim, n_steps=n_steps)
    assert final_state.shape == (1, dim)
    assert torch.all(torch.isfinite(final_state))

    # Check step size after running
    final_step_size = hmc_sampler.get_scheduled_value("step_size")
    assert np.isclose(final_step_size, 0.01)  # Should reach end value
    assert final_step_size < initial_step_size


@pytest.mark.parametrize(
    "hmc_sampler",
    [
        (
            {
                "device": "cpu",
                "energy_function": {"type": "gaussian", "dim": 3, "device": "cpu"},
            }
        ),
        pytest.param(
            {
                "device": "cuda",
                "energy_function": {"type": "gaussian", "dim": 3, "device": "cuda"},
            },
            marks=requires_cuda,
        ),
    ],
    indirect=True,
)
def test_hmc_device_consistency(hmc_sampler):
    """Test that tensors are consistently on the correct device."""
    n_steps = 10
    device = hmc_sampler.device
    dim = hmc_sampler.model.mean.shape[0]

    # Sample and check device
    samples = hmc_sampler.sample(dim=dim, n_steps=n_steps)
    assert samples.device.type == device.type

    # Test with custom input and check device consistency
    x_init = torch.randn(5, dim, device=device)
    samples_custom = hmc_sampler.sample(x=x_init, n_steps=n_steps)
    assert samples_custom.device.type == device.type


@rc
@pytest.mark.parametrize(
    "energy_function, hmc_sampler",
    [
        (
            {"type": "gaussian", "dim": 4, "device": "cuda"},
            {
                "device": "cuda",
                "dtype": "float16",
                "step_size": 0.05,
                "n_leapfrog_steps": 8,
            },
        ),
    ],
    indirect=["energy_function", "hmc_sampler"],
)
def test_hmc_float16_cuda(hmc_sampler):
    """Test HMC operation with float16 dtype on CUDA."""
    assert hmc_sampler.device.type == "cuda"
    assert hmc_sampler.dtype == torch.float16

    dim = hmc_sampler.model.mean.shape[0]
    n_steps = 20
    n_samples = 4

    # Test momentum initialization dtype
    momentum = hmc_sampler._initialize_momentum(torch.Size([n_samples, dim]))
    assert momentum.dtype == torch.float16

    # Test sampling output dtype
    final_state = hmc_sampler.sample(dim=dim, n_steps=n_steps, n_samples=n_samples)
    assert final_state.dtype == torch.float16
    assert final_state.shape == (n_samples, dim)
    assert torch.all(torch.isfinite(final_state))

    # Test trajectory dtype
    trajectory = hmc_sampler.sample(
        dim=dim, n_steps=n_steps, n_samples=n_samples, return_trajectory=True
    )
    assert trajectory.dtype == torch.float16
    assert trajectory.shape == (n_samples, n_steps, dim)
    assert torch.all(torch.isfinite(trajectory))

    # Test diagnostics dtype
    _, diagnostics = hmc_sampler.sample(
        dim=dim, n_steps=n_steps, n_samples=n_samples, return_diagnostics=True
    )
    assert diagnostics.dtype == torch.float16
    assert diagnostics.shape == (n_steps, 4, n_samples, dim)
    assert torch.all(torch.isfinite(diagnostics))


###############################################################################
# Numerical Stability and Edge Case Tests
###############################################################################


@pytest.mark.parametrize(
    "energy_function, hmc_sampler",
    [
        (
            {"type": "nan_inducing", "dim": 2, "nan_threshold": 1.0},
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "step_size": 0.1,
                "n_leapfrog_steps": 10,
            },
        ),
    ],
    indirect=["energy_function", "hmc_sampler"],
)
def test_leapfrog_nan_handling(hmc_sampler):
    """Test _leapfrog_integration handles NaNs occurring mid-integration."""
    dim = hmc_sampler.model.dim
    shape = torch.Size([5, dim])
    # Start near the threshold to potentially cross it quickly
    position = (
        torch.ones(shape, device=hmc_sampler.device, dtype=hmc_sampler.dtype) * 0.9
    )
    momentum = (
        torch.randn(shape, device=hmc_sampler.device, dtype=hmc_sampler.dtype) * 0.5
    )

    # Run leapfrog integration
    state = {"x": position, "p": momentum}
    result = hmc_sampler.integrator.integrate(
        state,
        hmc_sampler.model,
        hmc_sampler.get_scheduled_value("step_size"),
        hmc_sampler.n_leapfrog_steps,
        hmc_sampler.mass,
    )
    new_pos, new_mom = result["x"], result["p"]

    # Check output shape and finiteness (due to nan_to_num replacement)
    assert new_pos.shape == shape
    assert new_mom.shape == shape
    assert torch.all(torch.isfinite(new_pos))
    assert torch.all(torch.isfinite(new_mom))

    # Check that the values are not excessively large
    assert torch.all(torch.abs(new_pos) < 1e6)
    assert torch.all(torch.abs(new_mom) < 1e6)


def test_hmc_small_step_size():
    """Test HMC with a very small step size."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_fn = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2)).to(device)
    tiny_step = 1e-5
    hmc = HamiltonianMonteCarlo(
        model=energy_fn,
        step_size=tiny_step,
        n_leapfrog_steps=10,
        device=device,
    )
    samples = hmc.sample(dim=2, n_steps=20, n_samples=5)
    assert samples.shape == (5, 2)
    assert torch.all(torch.isfinite(samples))


def test_hmc_large_leapfrog_steps():
    """Test HMC with a large number of leapfrog steps."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_fn = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2)).to(device)
    many_steps = 100
    hmc = HamiltonianMonteCarlo(
        model=energy_fn,
        step_size=0.01,
        n_leapfrog_steps=many_steps,
        device=device,
    )
    samples = hmc.sample(dim=2, n_steps=20, n_samples=5)
    assert samples.shape == (5, 2)
    assert torch.all(torch.isfinite(samples))


@rc
def test_hmc_high_dimensions():
    """Test HMC in high-dimensional spaces."""
    dim = 100
    device = "cuda"
    energy_fn = GaussianModel(mean=torch.zeros(dim), cov=torch.eye(dim)).to(device)
    hmc = HamiltonianMonteCarlo(
        model=energy_fn,
        step_size=0.01,
        n_leapfrog_steps=20,
        device=device,
        dtype=torch.float32,
    )

    samples = hmc.sample(dim=dim, n_steps=50, n_samples=10)
    assert samples.shape == (10, dim)
    assert torch.all(torch.isfinite(samples))

    sample_means = samples.mean(dim=0)
    assert torch.abs(sample_means.mean()) < 0.15
    within_tolerance = (torch.abs(sample_means) < 1.5).float().mean()
    assert within_tolerance >= 0.85


def test_hmc_custom_initial_state():
    """Test HMC with specific custom initial state."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_fn = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2)).to(device)
    hmc = HamiltonianMonteCarlo(
        model=energy_fn, step_size=0.1, n_leapfrog_steps=10, device=device
    )
    initial_state = torch.tensor([[10.0, -10.0]], device=hmc.device, dtype=hmc.dtype)
    samples = hmc.sample(x=initial_state, n_steps=200)

    final_dist_to_mean = torch.norm(samples - hmc.model.mean.to(hmc.device))
    initial_dist_to_mean = torch.norm(initial_state - hmc.model.mean.to(hmc.device))
    assert final_dist_to_mean < initial_dist_to_mean


@pytest.mark.parametrize(
    "energy_function, hmc_sampler",
    [
        (
            {"type": "gaussian", "dim": 2},
            {"device": "cuda" if torch.cuda.is_available() else "cpu"},
        ),
    ],
    indirect=True,
)
def test_hmc_dim_inference(hmc_sampler):
    """Test automatic dimension inference when x=None, dim=None."""
    if not hasattr(hmc_sampler.model, "mean"):
        pytest.skip(
            "Dimension inference test requires energy_function with 'mean' attribute."
        )

    dim = hmc_sampler.model.mean.shape[0]
    n_steps = 10

    # Call sample without specifying dim or x
    final_state = hmc_sampler.sample(n_steps=n_steps)

    assert final_state.shape == (1, dim)
    assert torch.all(torch.isfinite(final_state))


@pytest.mark.parametrize(
    "energy_function, hmc_sampler",
    [
        # Use DoubleWell which doesn't have a 'mean' attribute by default
        ({"type": "double_well", "dim": 2}, {"device": "cpu"}),
    ],
    indirect=True,
)
def test_hmc_dim_inference_failure(hmc_sampler):
    """Test ValueError when dim cannot be inferred."""
    if hasattr(hmc_sampler.model, "mean"):
        pytest.skip("Skipping failure test as energy function unexpectedly has 'mean'.")

    n_steps = 10
    with pytest.raises(ValueError, match="dim must be provided when x is None"):
        hmc_sampler.sample(n_steps=n_steps)


###############################################################################
# Statistical Property Tests
###############################################################################


@rc
def test_hmc_gaussian_sampling_statistics():
    """Test statistical properties of HMC samples from a Gaussian distribution."""
    device = "cuda"
    # Create a 2D Gaussian with known parameters
    mean = torch.tensor([1.0, -1.0], device=device)
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]], device=device)
    energy_fn = GaussianModel(mean=mean, cov=cov).to(device)

    hmc = HamiltonianMonteCarlo(
        model=energy_fn,
        step_size=0.05,
        n_leapfrog_steps=15,
        device=device,
        dtype=torch.float32,
    )

    n_samples = 1500
    n_steps = 250
    burn_in = 50

    # Generate samples with trajectory to discard burn-in
    trajectory = hmc.sample(
        n_samples=n_samples, n_steps=n_steps, dim=2, return_trajectory=True
    )
    samples = trajectory[:, burn_in:, :].reshape(-1, 2)  # Collect samples after burn-in

    actual_n_samples = samples.shape[0]

    sample_mean = samples.mean(dim=0)
    centered = samples - sample_mean
    sample_cov = torch.matmul(centered.t(), centered) / (actual_n_samples - 1)

    # Check statistics with reasonable tolerance
    assert torch.allclose(sample_mean, mean, rtol=0.15, atol=0.15)
    assert torch.allclose(sample_cov, cov, rtol=0.25, atol=0.25)


###############################################################################
# Internal Component Tests
###############################################################################


def test_hmc_step_internals():
    """Test internal HMC step components, including stability aspects."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_fn = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2)).to(device)
    hmc = HamiltonianMonteCarlo(
        model=energy_fn, step_size=0.1, n_leapfrog_steps=5, device=device
    )
    dtype = hmc.dtype
    batch_size = 10
    dim = 2
    shape = torch.Size([batch_size, dim])

    # Test momentum initialization
    torch.manual_seed(123)
    momentum = hmc._initialize_momentum(shape)
    assert momentum.shape == shape
    assert momentum.dtype == dtype
    assert momentum.device.type == device

    # Test kinetic energy calculation
    kinetic = hmc._compute_kinetic_energy(momentum)
    assert kinetic.shape == torch.Size([batch_size])
    assert torch.all(kinetic >= 0)
    assert torch.all(torch.isfinite(kinetic))

    # Test leapfrog step with normal values
    position = torch.zeros(shape, device=device, dtype=dtype)
    state = {"x": position, "p": momentum}
    result = hmc.integrator.integrate(
        state,
        hmc.model,
        hmc.get_scheduled_value("step_size"),
        hmc.n_leapfrog_steps,
        hmc.mass,
    )
    new_position, new_momentum = result["x"], result["p"]
    assert new_position.shape == position.shape
    assert new_momentum.shape == momentum.shape
    assert torch.all(torch.isfinite(new_position))
    assert torch.all(torch.isfinite(new_momentum))

    # Test leapfrog step with potentially large gradients (testing clamping)
    far_position = torch.ones(shape, device=device, dtype=dtype) * 100.0
    far_momentum = torch.randn(shape, device=device, dtype=dtype)
    state = {"x": far_position, "p": far_momentum}
    result = hmc.integrator.integrate(
        state,
        hmc.model,
        hmc.get_scheduled_value("step_size"),
        hmc.n_leapfrog_steps,
        hmc.mass,
    )
    new_pos_far, new_mom_far = result["x"], result["p"]
    assert torch.all(torch.isfinite(new_pos_far))
    assert torch.all(torch.isfinite(new_mom_far))

    # Test single leapfrog step (integrator step)
    torch.manual_seed(123)
    position = torch.zeros(shape, device=device, dtype=dtype)
    momentum = hmc._initialize_momentum(shape)
    state = {"x": position, "p": momentum}
    result = hmc.integrator.step(
        state, hmc.model, hmc.get_scheduled_value("step_size"), hmc.mass
    )
    new_position, new_momentum = result["x"], result["p"]
    assert new_position.shape == position.shape
    assert new_momentum.shape == momentum.shape
    assert torch.all(torch.isfinite(new_position))
    assert torch.all(torch.isfinite(new_momentum))


@pytest.mark.parametrize(
    "energy_function, hmc_sampler",
    [
        (
            {"type": "gaussian", "dim": 2},
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "n_leapfrog_steps": 5,
            },
        ),
    ],
    indirect=True,
)
def test_hmc_diagnostics_stability(hmc_sampler):
    """Test stability of diagnostics calculation, especially variance clamping."""
    dim = hmc_sampler.model.mean.shape[0]
    n_steps = 20

    # Case 1: Single sample (variance should be zero, then clamped)
    n_samples_1 = 1
    final_state_1, diagnostics_1 = hmc_sampler.sample(
        dim=dim, n_steps=n_steps, n_samples=n_samples_1, return_diagnostics=True
    )
    assert final_state_1.shape == (n_samples_1, dim)
    assert diagnostics_1.shape == (n_steps, 4, n_samples_1, dim)
    assert torch.all(torch.isfinite(diagnostics_1))
    # Check variance component (index 1) - should be clamped non-negative
    assert torch.all(diagnostics_1[:, 1, :, :] >= 0)

    # Case 2: Multiple identical samples (variance is zero, should be clamped)
    n_samples_multi = 5
    # Start all samples at the same point
    initial_state = torch.zeros(
        (n_samples_multi, dim), device=hmc_sampler.device, dtype=hmc_sampler.dtype
    )
    final_state_multi, diagnostics_multi = hmc_sampler.sample(
        x=initial_state,
        n_steps=1,
        n_samples=n_samples_multi,
        return_diagnostics=True,
    )

    assert final_state_multi.shape == (n_samples_multi, dim)
    assert diagnostics_multi.shape == (1, 4, n_samples_multi, dim)
    assert torch.all(torch.isfinite(diagnostics_multi))
    # Check variance component (index 1) - should be clamped non-negative
    variance_diag = diagnostics_multi[0, 1, :, :]
    assert torch.all(variance_diag >= 0)


@pytest.mark.parametrize("start_val", [1e4, 1e6])
def test_hmc_numerical_stability_extreme_values(start_val):
    """Test HMC numerical stability with extreme values and check internal clamping."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use an energy function where energy grows quickly (e.g., quartic)
    class QuarticEnergy(BaseModel):
        def __init__(self, dim=2, device="cpu"):
            super().__init__()
            self.dim = dim
            self._device = torch.device(device)
            self.dummy_param = torch.nn.Parameter(torch.zeros(1, device=self._device))

        @property
        def device(self):
            return self._device

        def energy(self, x: torch.Tensor) -> torch.Tensor:
            return 0.1 * torch.sum(x**4, dim=-1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Forward method to satisfy BaseEnergyFunction contract
            return self.energy(x)

        def gradient(self, x: torch.Tensor) -> torch.Tensor:
            return 0.4 * x**3

        def to(self, device):
            self._device = torch.device(device)
            self.dummy_param = self.dummy_param.to(self._device)
            return self

    energy_fn = QuarticEnergy(dim=2, device=device).to(device)

    # Use adjusted HMC params for stability
    hmc = HamiltonianMonteCarlo(
        model=energy_fn, step_size=1e-3, n_leapfrog_steps=5, device=device
    )

    # Extreme initial position
    extreme_position = (
        torch.ones((3, 2), device=hmc.device, dtype=hmc.dtype) * start_val
    )

    # Test internal leapfrog step with extreme input
    momentum = hmc._initialize_momentum(extreme_position.shape)
    state = {"x": extreme_position, "p": momentum}
    result = hmc.integrator.integrate(
        state,
        hmc.model,
        hmc.get_scheduled_value("step_size"),
        hmc.n_leapfrog_steps,
        hmc.mass,
    )
    new_pos_leap, new_mom_leap = result["x"], result["p"]
    assert torch.all(torch.isfinite(new_pos_leap))
    assert torch.all(torch.isfinite(new_mom_leap))

    # Test full sampling for finiteness
    result = hmc.sample(x=extreme_position.clone(), n_steps=10)
    assert torch.all(torch.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__])
