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
from torchebm.samplers import HamiltonianMonteCarlo, RiemannianManifoldHMC
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

    energy_function = energy_function.to(device)

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
    assert isinstance(diagnostics, dict)
    assert set(diagnostics) == {"mean", "var", "energy", "acceptance_rate"}
    assert diagnostics["mean"].shape == (n_steps, dim)
    assert diagnostics["var"].shape == (n_steps, dim)
    assert diagnostics["energy"].shape == (n_steps,)
    assert diagnostics["acceptance_rate"].shape == (n_steps,)
    for v in diagnostics.values():
        assert torch.all(torch.isfinite(v))


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
    assert isinstance(diagnostics, dict)
    for v in diagnostics.values():
        assert v.dtype == torch.float16
        assert torch.all(torch.isfinite(v))
    assert diagnostics["mean"].shape == (n_steps, dim)


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
        step_size=hmc_sampler.get_scheduled_value("step_size"),
        n_steps=hmc_sampler.n_leapfrog_steps,
        mass=hmc_sampler.mass,
        drift=lambda x_, t_: -hmc_sampler.model.gradient(x_),
        safe=True,
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
        step_size=hmc.get_scheduled_value("step_size"),
        n_steps=hmc.n_leapfrog_steps,
        mass=hmc.mass,
        drift=lambda x_, t_: -hmc.model.gradient(x_),
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
        step_size=hmc.get_scheduled_value("step_size"),
        n_steps=hmc.n_leapfrog_steps,
        mass=hmc.mass,
        drift=lambda x_, t_: -hmc.model.gradient(x_),
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
        state,
        step_size=hmc.get_scheduled_value("step_size"),
        mass=hmc.mass,
        drift=lambda x_, t_: -hmc.model.gradient(x_),
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
    assert diagnostics_1["mean"].shape == (n_steps, dim)
    assert diagnostics_1["var"].shape == (n_steps, dim)
    for v in diagnostics_1.values():
        assert torch.all(torch.isfinite(v))
    # Variance non-negative
    assert torch.all(diagnostics_1["var"] >= 0)

    # Case 2: Multiple identical samples (variance is zero, should be clamped)
    n_samples_multi = 5
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
    assert diagnostics_multi["var"].shape == (1, dim)
    for v in diagnostics_multi.values():
        assert torch.all(torch.isfinite(v))
    assert torch.all(diagnostics_multi["var"] >= 0)


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
        step_size=hmc.get_scheduled_value("step_size"),
        n_steps=hmc.n_leapfrog_steps,
        mass=hmc.mass,
        drift=lambda x_, t_: -hmc.model.gradient(x_),
        safe=True,
    )
    new_pos_leap, new_mom_leap = result["x"], result["p"]
    assert torch.all(torch.isfinite(new_pos_leap))
    assert torch.all(torch.isfinite(new_mom_leap))

    # Test full sampling for finiteness
    result = hmc.sample(x=extreme_position.clone(), n_steps=10)
    assert torch.all(torch.isfinite(result))


###############################################################################
# RiemannianManifoldHMC Tests
###############################################################################
#
# Shared metric helpers
# ---------------------
#   _identity_metric : G(x) = I — degenerate case where RMHMC must behave
#                      like ordinary HMC on the same target.
#   _outer_metric    : G(x) = I + alpha * x x^T — a position-dependent SPD
#                      metric that exercises the non-separable code path
#                      (∂H/∂p and ∂H/∂x both depend on x and p).
#


def _identity_metric(dim):
    def metric_fn(x):
        eye = torch.eye(dim, dtype=x.dtype, device=x.device)
        return eye.expand(x.shape[0], dim, dim).contiguous()
    return metric_fn


def _outer_metric(dim, alpha=0.5):
    def metric_fn(x):
        eye = torch.eye(dim, dtype=x.dtype, device=x.device).expand(
            x.shape[0], dim, dim
        )
        outer = x.unsqueeze(-1) * x.unsqueeze(-2)
        return eye + alpha * outer
    return metric_fn


def _stiff_metric(dim, alpha=4.0, beta=3.0):
    """Strongly anisotropic, position-dependent SPD metric.

    ``G(x) = diag(1 + beta*i) + alpha * x x^T`` has a high (and
    position-dependent) condition number, which stresses the GLI Picard
    solve far more than the mild ``_outer_metric``.
    """
    def metric_fn(x):
        d = 1.0 + beta * torch.arange(dim, device=x.device, dtype=x.dtype)
        base = torch.diag(d).expand(x.shape[0], dim, dim)
        outer = x.unsqueeze(-1) * x.unsqueeze(-2)
        return base + alpha * outer
    return metric_fn


def _gaussian_target(dim, device):
    return GaussianModel(
        mean=torch.zeros(dim, device=device),
        cov=torch.eye(dim, device=device),
    ).to(device)


# ----- Initialization & validation -----


def test_rmhmc_initialization():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _gaussian_target(2, device)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=_identity_metric(2),
        step_size=0.1, n_leapfrog_steps=10, device=device,
    )
    assert isinstance(sampler, RiemannianManifoldHMC)
    assert sampler.schedulers["step_size"] is not None
    assert isinstance(sampler.schedulers["step_size"], BaseScheduler)
    assert sampler.n_leapfrog_steps == 10
    # The default integrator should be the Generalised Leapfrog.
    assert type(sampler.integrator).__name__ == "GeneralisedLeapfrogIntegrator"


def test_rmhmc_initialization_invalid_metric_fn():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _gaussian_target(2, device)
    with pytest.raises(TypeError, match="metric_fn must be callable"):
        RiemannianManifoldHMC(model, metric_fn=None)
    with pytest.raises(TypeError, match="metric_fn must be callable"):
        RiemannianManifoldHMC(model, metric_fn=42)


def test_rmhmc_initialization_invalid_n_leapfrog_steps():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _gaussian_target(2, device)
    metric_fn = _identity_metric(2)
    with pytest.raises(ValueError, match="n_leapfrog_steps must be positive"):
        RiemannianManifoldHMC(model, metric_fn=metric_fn, n_leapfrog_steps=0)
    with pytest.raises(ValueError, match="n_leapfrog_steps must be positive"):
        RiemannianManifoldHMC(model, metric_fn=metric_fn, n_leapfrog_steps=-3)


def test_rmhmc_initialization_invalid_step_size():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _gaussian_target(2, device)
    metric_fn = _identity_metric(2)
    with pytest.raises(ValueError, match="step_size must be positive"):
        RiemannianManifoldHMC(model, metric_fn=metric_fn, step_size=-0.1)
    with pytest.raises(ValueError, match="step_size must be positive"):
        RiemannianManifoldHMC(model, metric_fn=metric_fn, step_size=0.0)


# ----- Sampling interface -----


def test_rmhmc_sample_shape():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 2
    model = _gaussian_target(dim, device)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=_identity_metric(dim),
        step_size=0.1, n_leapfrog_steps=8, device=device,
    )
    samples = sampler.sample(n_samples=8, dim=dim, n_steps=20)
    assert samples.shape == (8, dim)
    assert torch.all(torch.isfinite(samples))


def test_rmhmc_sample_with_trajectory():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 2
    model = _gaussian_target(dim, device)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=_identity_metric(dim),
        step_size=0.1, n_leapfrog_steps=6, device=device,
    )
    traj = sampler.sample(n_samples=4, dim=dim, n_steps=15, return_trajectory=True)
    assert traj.shape == (4, 15, dim)
    assert torch.all(torch.isfinite(traj))


def test_rmhmc_sample_with_diagnostics():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 2
    n_steps = 15
    model = _gaussian_target(dim, device)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=_identity_metric(dim),
        step_size=0.1, n_leapfrog_steps=6, device=device,
    )
    final_state, diag = sampler.sample(
        n_samples=8, dim=dim, n_steps=n_steps, return_diagnostics=True
    )
    assert final_state.shape == (8, dim)
    assert isinstance(diag, dict)
    assert set(diag) == {"mean", "var", "energy", "acceptance_rate"}
    assert diag["mean"].shape == (n_steps, dim)
    assert diag["var"].shape == (n_steps, dim)
    assert diag["energy"].shape == (n_steps,)
    assert diag["acceptance_rate"].shape == (n_steps,)
    for v in diag.values():
        assert torch.all(torch.isfinite(v))
    assert torch.all(diag["acceptance_rate"] >= 0)
    assert torch.all(diag["acceptance_rate"] <= 1)
    assert torch.all(diag["var"] >= 0)


def test_rmhmc_thinning():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 2
    model = _gaussian_target(dim, device)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=_identity_metric(dim),
        step_size=0.1, n_leapfrog_steps=5, device=device,
    )
    thin = 5
    n_steps = 30
    traj = sampler.sample(
        n_samples=4, dim=dim, n_steps=n_steps, thin=thin, return_trajectory=True
    )
    assert traj.shape == (4, n_steps // thin, dim)


def test_rmhmc_thin_validation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _gaussian_target(2, device)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=_identity_metric(2),
        step_size=0.1, n_leapfrog_steps=5, device=device,
    )
    with pytest.raises(ValueError, match="thin must be >= 1"):
        sampler.sample(n_samples=4, dim=2, n_steps=10, thin=0)


def test_rmhmc_requires_2d_state():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _gaussian_target(2, device)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=_identity_metric(2),
        step_size=0.1, n_leapfrog_steps=5, device=device,
    )
    bad_x = torch.randn(4, 2, 3, device=device)
    with pytest.raises(ValueError, match="2-D state tensors"):
        sampler.sample(x=bad_x, n_steps=5)


def test_rmhmc_dim_inference():
    """dim auto-inferred from model.mean when x and dim are both None."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 3
    model = _gaussian_target(dim, device)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=_identity_metric(dim),
        step_size=0.1, n_leapfrog_steps=5, device=device,
    )
    final_state = sampler.sample(n_steps=5)
    assert final_state.shape == (1, dim)


def test_rmhmc_dim_inference_failure():
    """Model without `mean` attribute and no dim/x argument must raise."""
    device = "cpu"
    model = DoubleWellModel(barrier_height=2.0).to(device)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=_identity_metric(2),
        step_size=0.05, n_leapfrog_steps=4, device=device,
    )
    with pytest.raises(ValueError, match="dim must be provided when x is None"):
        sampler.sample(n_steps=5)


def test_rmhmc_custom_initial_state():
    """Starting far from the mode, the chain must move toward it."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 2
    model = _gaussian_target(dim, device)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=_identity_metric(dim),
        step_size=0.1, n_leapfrog_steps=10, device=device,
    )
    initial_state = torch.tensor(
        [[5.0, -5.0]], device=device, dtype=sampler.dtype
    )
    samples = sampler.sample(x=initial_state, n_steps=80)
    final_dist = torch.norm(samples - model.mean.to(device))
    initial_dist = torch.norm(initial_state - model.mean.to(device))
    assert final_dist < initial_dist


# ----- Reproducibility -----


def test_rmhmc_reproducibility():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 2
    model = _gaussian_target(dim, device)
    metric_fn = _identity_metric(dim)

    def make_sampler():
        return RiemannianManifoldHMC(
            model, metric_fn=metric_fn,
            step_size=0.1, n_leapfrog_steps=8, device=device,
        )

    torch.manual_seed(7)
    r1 = make_sampler().sample(n_samples=4, dim=dim, n_steps=20)
    torch.manual_seed(7)
    r2 = make_sampler().sample(n_samples=4, dim=dim, n_steps=20)
    assert torch.allclose(r1, r2)


# ----- Internal helpers (identity-metric closed forms) -----


def test_rmhmc_internal_components_identity_metric():
    """Under an identity metric on N(0, I):

        K(x, p) = 1/2 * |p|^2 + 1/2 log|I| = 1/2 |p|^2
        velocity = G^{-1} p = p
        force    = -∂(U + K)/∂x = -x  (since ∂K/∂x = 0)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 3
    model = _gaussian_target(dim, device)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=_identity_metric(dim),
        step_size=0.05, n_leapfrog_steps=4,
        dtype=torch.float64, device=device,
    )

    torch.manual_seed(0)
    x = torch.randn(5, dim, dtype=torch.float64, device=device)
    p = torch.randn(5, dim, dtype=torch.float64, device=device)
    t = torch.zeros(5, dtype=torch.float64, device=device)

    K = sampler._kinetic_energy(x, p)
    K_expected = 0.5 * (p ** 2).sum(dim=-1)
    assert torch.allclose(K, K_expected, atol=1e-10)

    v = sampler._velocity(x, p, t)
    assert torch.allclose(v, p, atol=1e-10)

    f = sampler._force(x, p, t)
    assert torch.allclose(f, -x, atol=1e-6)

    # Momentum sampling under identity metric: roughly N(0, I).
    torch.manual_seed(1)
    big_batch = torch.zeros(2000, dim, dtype=torch.float64, device=device)
    p_samples = sampler._initialize_momentum(big_batch)
    assert p_samples.shape == big_batch.shape
    assert torch.all(torch.isfinite(p_samples))
    assert p_samples.std(dim=0).mean().item() == pytest.approx(1.0, abs=0.1)


def test_rmhmc_momentum_covariance_matches_metric():
    """Empirically, momenta sampled by the helper have covariance ≈ G(x).

    Uses a constant non-identity metric so the empirical covariance has a
    known target without depending on x.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 2
    G_const = torch.tensor([[2.0, 0.5], [0.5, 1.5]], dtype=torch.float64, device=device)

    def metric_const(x):
        return G_const.expand(x.shape[0], dim, dim).contiguous()

    model = _gaussian_target(dim, device)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=metric_const,
        step_size=0.1, n_leapfrog_steps=4,
        dtype=torch.float64, device=device,
    )

    torch.manual_seed(0)
    x = torch.zeros(20_000, dim, dtype=torch.float64, device=device)
    p = sampler._initialize_momentum(x)
    emp_cov = (p.t() @ p) / p.shape[0]
    assert torch.allclose(emp_cov, G_const, atol=0.06)


# ----- Statistical target recovery -----


def test_rmhmc_identity_metric_recovers_gaussian():
    """Identity metric -- empirical moments must match N(0, I)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 2
    model = _gaussian_target(dim, device)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=_identity_metric(dim),
        step_size=0.2, n_leapfrog_steps=8,
        dtype=torch.float64, device=device,
    )
    torch.manual_seed(0)
    samples = sampler.sample(n_samples=200, dim=dim, n_steps=100)
    emp_mean = samples.mean(dim=0)
    emp_std = samples.std(dim=0)
    assert torch.allclose(
        emp_mean, torch.zeros(dim, dtype=torch.float64, device=device), atol=0.3
    )
    assert torch.allclose(
        emp_std, torch.ones(dim, dtype=torch.float64, device=device), atol=0.3
    )


def test_rmhmc_position_dependent_metric_recovers_gaussian():
    """With G(x) = I + 0.5 * x x^T the Hamiltonian is non-separable, so the
    GLI Picard solves and the MH correction are exercised together; the
    invariant distribution must still be N(0, I).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 2
    model = _gaussian_target(dim, device)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=_outer_metric(dim, alpha=0.5),
        step_size=0.1, n_leapfrog_steps=6,
        dtype=torch.float64, device=device,
    )
    torch.manual_seed(1)
    samples = sampler.sample(n_samples=200, dim=dim, n_steps=120)
    emp_mean = samples.mean(dim=0)
    emp_std = samples.std(dim=0)
    assert torch.allclose(
        emp_mean, torch.zeros(dim, dtype=torch.float64, device=device), atol=0.35
    )
    assert torch.allclose(
        emp_std, torch.ones(dim, dtype=torch.float64, device=device), atol=0.35
    )


def test_rmhmc_stiff_metric_recovers_gaussian():
    """Ill-conditioned, position-dependent metric: with enough Picard
    iterations the GLI still converges and RMHMC recovers N(0, I).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 2
    model = _gaussian_target(dim, device)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=_stiff_metric(dim, alpha=4.0, beta=3.0),
        step_size=0.05, n_leapfrog_steps=6,
        solver_max_iter=20, solver_check_every=1, solver_tol=1e-8,
        dtype=torch.float64, device=device,
    )
    torch.manual_seed(7)
    samples = sampler.sample(n_samples=200, dim=dim, n_steps=150)
    assert torch.isfinite(samples).all()
    emp_mean = samples.mean(dim=0)
    emp_std = samples.std(dim=0)
    assert torch.allclose(
        emp_mean, torch.zeros(dim, dtype=torch.float64, device=device), atol=0.4
    )
    assert torch.allclose(
        emp_std, torch.ones(dim, dtype=torch.float64, device=device), atol=0.4
    )


def test_rmhmc_stiff_metric_stable_default_solver():
    """The same stiff metric with the default solver must stay numerically
    stable (no NaN/Inf) thanks to the finite-proposal guard and safe mode,
    even if the chain is more biased or lower-acceptance.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 2
    model = _gaussian_target(dim, device)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=_stiff_metric(dim, alpha=4.0, beta=3.0),
        step_size=0.05, n_leapfrog_steps=6,
        dtype=torch.float64, device=device,
    )
    torch.manual_seed(8)
    samples = sampler.sample(n_samples=64, dim=dim, n_steps=80)
    assert samples.shape == (64, dim)
    assert torch.isfinite(samples).all()


def test_rmhmc_acceptance_rate_reasonable():
    """Tuned chain (small step, identity metric) must have high acceptance."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 2
    model = _gaussian_target(dim, device)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=_identity_metric(dim),
        step_size=0.1, n_leapfrog_steps=5,
        dtype=torch.float64, device=device,
    )
    torch.manual_seed(42)
    _, diag = sampler.sample(
        n_samples=32, dim=dim, n_steps=60, return_diagnostics=True
    )
    avg_accept = diag["acceptance_rate"][-20:].mean().item()
    assert avg_accept > 0.7, f"Acceptance rate {avg_accept} unexpectedly low"


# ----- Scheduler integration -----


def test_rmhmc_with_scheduler():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 2
    model = _gaussian_target(dim, device)
    n_steps = 30
    scheduler = LinearScheduler(start_value=0.2, end_value=0.05, n_steps=n_steps)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=_identity_metric(dim),
        step_size=scheduler, n_leapfrog_steps=5, device=device,
    )
    assert isinstance(sampler.schedulers["step_size"], LinearScheduler)
    initial_step = sampler.get_scheduled_value("step_size")
    assert np.isclose(initial_step, 0.2)

    sampler.sample(n_samples=4, dim=dim, n_steps=n_steps)
    final_step = sampler.get_scheduled_value("step_size")
    assert np.isclose(final_step, 0.05)
    assert final_step < initial_step


# ----- Device consistency -----


def test_rmhmc_device_consistency():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 3
    model = _gaussian_target(dim, device)
    sampler = RiemannianManifoldHMC(
        model, metric_fn=_identity_metric(dim),
        step_size=0.1, n_leapfrog_steps=5, device=device,
    )
    samples = sampler.sample(n_samples=4, dim=dim, n_steps=10)
    assert samples.device.type == sampler.device.type

    x_init = torch.randn(4, dim, device=device, dtype=sampler.dtype)
    samples_custom = sampler.sample(x=x_init, n_steps=10)
    assert samples_custom.device.type == sampler.device.type


# ----- Heavy statistical recovery (CUDA-gated for speed) -----


@rc
def test_rmhmc_gaussian_sampling_statistics():
    """Recover a correlated 2D Gaussian's mean and covariance."""
    device = "cuda"
    dim = 2
    mean = torch.tensor([1.0, -1.0], device=device)
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]], device=device)
    model = GaussianModel(mean=mean, cov=cov).to(device)

    sampler = RiemannianManifoldHMC(
        model, metric_fn=_identity_metric(dim),
        step_size=0.05, n_leapfrog_steps=12, device=device,
    )

    n_samples = 800
    n_steps = 200
    burn_in = 50
    traj = sampler.sample(
        n_samples=n_samples, dim=dim, n_steps=n_steps, return_trajectory=True
    )
    samples = traj[:, burn_in:, :].reshape(-1, dim)

    n = samples.shape[0]
    smean = samples.mean(dim=0)
    centered = samples - smean
    scov = centered.t() @ centered / (n - 1)

    assert torch.allclose(smean, mean, rtol=0.2, atol=0.2)
    assert torch.allclose(scov, cov, rtol=0.3, atol=0.3)


############################### Integrator Constructor Arg #####################


def _integrator_arg_model(dim=2):
    return GaussianModel(mean=torch.zeros(dim), cov=torch.eye(dim))


def test_hmc_integrator_string_and_instance():
    from torchebm.integrators import LeapfrogIntegrator

    sampler = HamiltonianMonteCarlo(
        _integrator_arg_model(), step_size=0.1, integrator="leapfrog"
    )
    assert type(sampler.integrator) is LeapfrogIntegrator

    inst = LeapfrogIntegrator(dtype=torch.float32)
    sampler = HamiltonianMonteCarlo(
        _integrator_arg_model(), step_size=0.1, integrator=inst
    )
    assert sampler.integrator is inst


def test_hmc_rejects_nonseparable_integrator():
    from torchebm.integrators import GeneralisedLeapfrogIntegrator

    with pytest.raises(TypeError, match="separable"):
        HamiltonianMonteCarlo(
            _integrator_arg_model(), step_size=0.1,
            integrator=GeneralisedLeapfrogIntegrator(dtype=torch.float32),
        )


def test_hmc_rejects_nonsymplectic_integrator():
    from torchebm.integrators import EulerMaruyamaIntegrator

    with pytest.raises(TypeError, match="BaseSymplecticIntegrator"):
        HamiltonianMonteCarlo(
            _integrator_arg_model(), step_size=0.1,
            integrator=EulerMaruyamaIntegrator(dtype=torch.float32),
        )


def _eye_metric(x):
    dim = x.shape[-1]
    eye = torch.eye(dim, dtype=x.dtype, device=x.device)
    return eye.expand(x.shape[0], dim, dim).contiguous()


def test_rmhmc_rejects_separable_integrator():
    from torchebm.integrators import LeapfrogIntegrator

    with pytest.raises(TypeError, match="non-separable"):
        RiemannianManifoldHMC(
            _integrator_arg_model(), metric_fn=_eye_metric,
            step_size=0.1,
            integrator=LeapfrogIntegrator(dtype=torch.float32),
        )


def test_rmhmc_solver_kwargs_deprecated_but_honored():
    with pytest.warns(DeprecationWarning, match="solver_max_iter"):
        sampler = RiemannianManifoldHMC(
            _integrator_arg_model(), metric_fn=_eye_metric,
            step_size=0.1, solver_max_iter=20,
        )
    assert sampler.integrator.solver_max_iter == 20
    # Untouched solver options keep their integrator defaults.
    assert sampler.integrator.solver_tol == 1e-6
    assert sampler.integrator.solver_check_every == 0


def test_rmhmc_solver_kwargs_conflict_with_integrator():
    from torchebm.integrators import GeneralisedLeapfrogIntegrator

    inst = GeneralisedLeapfrogIntegrator(dtype=torch.float32)
    with pytest.raises(ValueError, match="not both"):
        RiemannianManifoldHMC(
            _integrator_arg_model(), metric_fn=_eye_metric,
            step_size=0.1, solver_max_iter=20, integrator=inst,
        )


def test_rmhmc_integrator_instance_samples():
    from torchebm.integrators import GeneralisedLeapfrogIntegrator

    inst = GeneralisedLeapfrogIntegrator(
        dtype=torch.float32, solver_max_iter=4
    )
    sampler = RiemannianManifoldHMC(
        _integrator_arg_model(), metric_fn=_eye_metric,
        step_size=0.1, n_leapfrog_steps=3, integrator=inst,
    )
    assert sampler.integrator is inst
    samples = sampler.sample(n_samples=8, dim=2, n_steps=5)
    assert samples.shape == (8, 2)
    assert torch.isfinite(samples).all()


if __name__ == "__main__":
    pytest.main([__file__])
