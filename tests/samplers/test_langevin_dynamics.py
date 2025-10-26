import pytest
import torch
from torchebm.core.base_model import BaseModel, GaussianModel
from torchebm.samplers import LangevinDynamics
from tests.conftest import requires_cuda


@pytest.fixture
def energy_function(request):
    if not hasattr(request, "param"):
        return GaussianModel(mean=torch.zeros(10), cov=torch.eye(10))

    # Use parameters when provided through parametrize
    mean = request.param.get("mean", torch.zeros(10))
    cov = request.param.get("cov", torch.eye(10))

    return GaussianModel(mean=mean, cov=cov)


@pytest.fixture
def langevin_sampler(request, energy_function):
    if not hasattr(request, "param"):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    else:
        device = request.param.get(
            "device",
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        )

    energy_function = energy_function.to(device)

    return LangevinDynamics(model=energy_function, step_size=5e-3, device=device).to(
        device
    )


def test_langevin_dynamics_initialization(langevin_sampler):
    sampler = langevin_sampler
    assert isinstance(sampler, LangevinDynamics)
    assert sampler.get_scheduled_value("step_size") == 5e-3
    assert sampler.get_scheduled_value("noise_scale") == 1.0


def test_langevin_dynamics_initialization_invalid_params(energy_function):
    with pytest.raises(ValueError):
        LangevinDynamics(energy_function, step_size=-0.1, noise_scale=0.1)
    with pytest.raises(ValueError):
        LangevinDynamics(energy_function, step_size=0.1, noise_scale=-0.1)


@requires_cuda
@pytest.mark.parametrize(
    "energy_function, langevin_sampler",
    [
        (
            {"mean": torch.zeros(2), "cov": torch.eye(2)},
            {"device": torch.device("cuda")},
        ),
        (
            {"mean": torch.zeros(2), "cov": 2 * torch.eye(2)},
            {"device": torch.device("cuda")},
        ),
    ],
    indirect=True,  # This tells pytest to apply parameters to fixtures
)
def test_langevin_dynamics_sample(langevin_sampler):
    dim = 2
    n_steps = 100
    # langevin_sampler.mean = torch.zeros(dim)
    # langevin_sampler.cov = torch.eye(dim)
    final_state = langevin_sampler.sample(dim=dim, n_steps=n_steps)
    assert final_state.shape == (1, dim)  # (n_samples, dim)
    assert torch.all(torch.isfinite(final_state))


@requires_cuda
@pytest.mark.parametrize(
    "energy_function, langevin_sampler",
    [
        (
            {"mean": torch.zeros(2), "cov": torch.eye(2)},
            {"device": torch.device("cuda")},
        ),
        (
            {"mean": torch.zeros(2), "cov": 2 * torch.eye(2)},
            {"device": torch.device("cuda")},
        ),
    ],
    indirect=True,  # This tells pytest to apply parameters to fixtures
)
def test_langevin_dynamics_sample_trajectory(langevin_sampler):
    dim = 2
    n_steps = 100
    trajectory = langevin_sampler.sample(
        dim=dim, n_steps=n_steps, return_trajectory=True
    )
    assert trajectory.shape == (1, n_steps, dim)  # (n_samples, k_steps, dim)
    assert torch.all(torch.isfinite(trajectory))


@requires_cuda
@pytest.mark.parametrize(
    "energy_function, langevin_sampler",
    [
        (
            {"mean": torch.zeros(2), "cov": torch.eye(2)},
            {"device": torch.device("cuda")},
        ),
        (
            {"mean": torch.zeros(2), "cov": 2 * torch.eye(2)},
            {"device": torch.device("cuda")},
        ),
    ],
    indirect=True,  # This tells pytest to apply parameters to fixtures
)
def test_langevin_dynamics_sample_chain(langevin_sampler):
    dim = 2
    n_steps = 100
    n_samples = 10
    samples = langevin_sampler.sample(dim=dim, n_steps=n_steps, n_samples=n_samples)
    assert samples.shape == (n_samples, dim)
    assert torch.all(torch.isfinite(samples))


@requires_cuda
@pytest.mark.parametrize(
    "energy_function, langevin_sampler",
    [
        (
            {"mean": torch.zeros(2), "cov": torch.eye(2)},
            {"device": torch.device("cuda")},
        ),
        (
            {"mean": torch.zeros(2), "cov": 2 * torch.eye(2)},
            {"device": torch.device("cuda")},
        ),
    ],
    indirect=True,  # This tells pytest to apply parameters to fixtures
)
def test_langevin_dynamics_reproducibility(langevin_sampler):
    torch.manual_seed(42)
    sampler1 = langevin_sampler
    dim = 2
    n_steps = 100
    result1 = sampler1.sample(dim=dim, n_steps=n_steps)

    torch.manual_seed(42)
    sampler2 = langevin_sampler
    result2 = sampler2.sample(dim=dim, n_steps=n_steps)

    assert torch.allclose(result1, result2)


@requires_cuda
@pytest.mark.parametrize(
    "langevin_sampler",
    [
        ({"device": torch.device("cuda")}),
    ],
    indirect=True,  # This tells pytest to apply parameters to fixtures
)
def test_langevin_dynamics_sample_x_input(langevin_sampler):
    dim = 10
    n_samples = 150
    n_steps = 100
    device = langevin_sampler.device
    x_init = torch.randn(n_samples, dim, dtype=torch.float32, device=device)
    samples = langevin_sampler.sample(x=x_init, n_steps=n_steps)
    assert samples.shape == (n_samples, dim)
    assert torch.all(torch.isfinite(samples))


@requires_cuda
def test_cuda_device_available():
    """Test that CUDA device is available. This will be skipped if no NVIDIA driver is found."""
    # Create a tensor on CUDA device
    x = torch.zeros(1, device="cuda")
    assert x.device.type == "cuda"


@requires_cuda
def test_langevin_gaussian_sampling_statistics():
    """Test statistical properties of Langevin samples from a Gaussian."""
    device = "cuda"
    mean = torch.tensor([1.0, -1.0], device=device)
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]], device=device)
    energy_fn = GaussianModel(mean=mean, cov=cov).to(device)

    sampler = LangevinDynamics(
        model=energy_fn,
        step_size=0.05,
        noise_scale=1.0,  # Increased noise_scale for better exploration
        device=device,
        dtype=torch.float32,
    )

    n_samples = 4000  # Increased n_samples for better statistics
    n_steps = 500  # Increased n_steps for longer chains
    burn_in = 200  # Increased burn_in to discard more initial samples

    trajectory = sampler.sample(
        n_samples=n_samples, n_steps=n_steps, dim=2, return_trajectory=True
    )
    samples = trajectory[:, burn_in:, :].reshape(-1, 2)

    sample_mean = samples.mean(dim=0)
    centered = samples - sample_mean
    sample_cov = torch.matmul(centered.t(), centered) / (samples.shape[0] - 1)

    assert torch.allclose(sample_mean, mean, rtol=0.2, atol=0.2)
    assert torch.allclose(sample_cov, cov, rtol=0.4, atol=0.4)  # Increased tolerance
