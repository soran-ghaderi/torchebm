import pytest
import torch
import numpy as np

from torchebm.core import BaseEnergyFunction, GaussianEnergy, DoubleWellEnergy
from torchebm.samplers import LangevinDynamics
from torchebm.losses import (
    ContrastiveDivergence,
    PersistentContrastiveDivergence,
    ParallelTemperingCD,
)
from tests.conftest import requires_cuda


class MLPEnergy(BaseEnergyFunction):
    """A simple MLP to act as the energy function for testing."""

    def __init__(self, input_dim=2, hidden_dim=8):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


@pytest.fixture
def energy_function(request):
    """Fixture to create energy functions for testing."""
    if not hasattr(request, "param"):
        return GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2))

    params = request.param
    if params.get("type") == "gaussian":
        mean = params.get("mean", torch.zeros(2))
        cov = params.get("cov", torch.eye(2))
        return GaussianEnergy(mean=mean, cov=cov)
    elif params.get("type") == "double_well":
        barrier_height = params.get("barrier_height", 2.0)
        return DoubleWellEnergy(barrier_height=barrier_height)
    elif params.get("type") == "mlp":
        input_dim = params.get("input_dim", 2)
        hidden_dim = params.get("hidden_dim", 8)
        return MLPEnergy(input_dim=input_dim, hidden_dim=hidden_dim)
    else:
        return GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2))


@pytest.fixture
def sampler(request, energy_function):
    """Fixture to create a sampler for testing."""
    if not hasattr(request, "param"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return LangevinDynamics(
            energy_function=energy_function,
            step_size=0.1,
            noise_scale=0.01,
            device=device,
        )

    params = request.param
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    step_size = params.get("step_size", 0.1)
    noise_scale = params.get("noise_scale", 0.01)

    # Ensure energy function is on the correct device
    energy_function = energy_function.to(device)

    return LangevinDynamics(
        energy_function=energy_function,
        step_size=step_size,
        noise_scale=noise_scale,
        device=device,
    )


@pytest.fixture
def cd_loss(request, energy_function, sampler):
    """Fixture to create a ContrastiveDivergence loss for testing."""
    if not hasattr(request, "param"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return ContrastiveDivergence(
            energy_function=energy_function,
            sampler=sampler,
            n_steps=10,
            persistent=False,
            device=device,
        )

    params = request.param
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    n_steps = params.get("n_steps", 10)
    persistent = params.get("persistent", False)

    return ContrastiveDivergence(
        energy_function=energy_function,
        sampler=sampler,
        n_steps=n_steps,
        persistent=persistent,
        device=device,
    )


def test_contrastive_divergence_initialization(energy_function, sampler):
    """Test initialization of ContrastiveDivergence."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cd = ContrastiveDivergence(
        energy_function=energy_function,
        sampler=sampler,
        n_steps=10,
        persistent=False,
        device=device,
    )

    assert isinstance(cd, ContrastiveDivergence)
    assert cd.energy_function == energy_function
    assert cd.sampler == sampler
    assert cd.n_steps == 10
    assert cd.persistent is False
    assert cd.device == device
    assert cd.chain is None


def test_contrastive_divergence_forward(cd_loss):
    """Test the forward method of ContrastiveDivergence."""
    device = cd_loss.device
    x = torch.randn(10, 2, device=device)

    loss, samples = cd_loss(x)

    assert isinstance(loss, torch.Tensor)
    assert isinstance(samples, torch.Tensor)
    assert loss.shape == torch.Size([])  # Should be a scalar
    assert samples.shape == x.shape


def test_contrastive_divergence_compute_loss(cd_loss):
    """Test the compute_loss method of ContrastiveDivergence."""
    device = cd_loss.device
    x = torch.randn(10, 2, device=device)
    pred_x = torch.randn(10, 2, device=device)

    loss = cd_loss.compute_loss(x, pred_x)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # Should be a scalar


@pytest.mark.parametrize("persistent", [False, True])
def test_contrastive_divergence_persistence(energy_function, sampler, persistent):
    """Test CD with and without persistence."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cd = ContrastiveDivergence(
        energy_function=energy_function,
        sampler=sampler,
        n_steps=10,
        persistent=persistent,
        device=device,
    )

    x = torch.randn(10, 2, device=device)
    loss1, samples1 = cd(x)

    if persistent:
        assert cd.chain is not None
        assert torch.allclose(cd.chain, samples1)
    else:
        # Could be None or not depending on the implementation,
        # but if not None should not be the samples
        if cd.chain is not None:
            assert not torch.allclose(cd.chain, samples1)


@pytest.mark.parametrize(
    "energy_function, sampler, cd_loss",
    [
        (
            {"type": "gaussian", "mean": torch.zeros(2), "cov": torch.eye(2)},
            {"device": "cuda" if torch.cuda.is_available() else "cpu"},
            {"n_steps": 5, "persistent": False},
        ),
        (
            {"type": "gaussian", "mean": torch.zeros(2), "cov": 2 * torch.eye(2)},
            {"device": "cuda" if torch.cuda.is_available() else "cpu"},
            {"n_steps": 10, "persistent": True},
        ),
        (
            {"type": "double_well", "barrier_height": 2.0},
            {"device": "cuda" if torch.cuda.is_available() else "cpu"},
            {"n_steps": 20, "persistent": False},
        ),
        (
            {"type": "mlp", "input_dim": 2, "hidden_dim": 16},
            {"device": "cuda" if torch.cuda.is_available() else "cpu"},
            {"n_steps": 15, "persistent": True},
        ),
    ],
    indirect=True,
)
def test_contrastive_divergence_with_different_energy_functions(cd_loss):
    """Test CD with different energy functions."""
    device = cd_loss.device
    x = torch.randn(10, 2, device=device)

    loss, samples = cd_loss(x)

    assert isinstance(loss, torch.Tensor)
    assert samples.shape == x.shape


@pytest.mark.parametrize("n_steps", [1, 5, 20])
def test_contrastive_divergence_n_steps_effect(energy_function, sampler, n_steps):
    """Test the effect of different numbers of MCMC steps."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create CD with specific n_steps
    cd = ContrastiveDivergence(
        energy_function=energy_function,
        sampler=sampler,
        n_steps=n_steps,
        persistent=False,
        device=device,
    )

    x = torch.randn(10, 2, device=device)
    loss, samples = cd(x)

    assert isinstance(loss, torch.Tensor)
    assert samples.shape == x.shape


@pytest.mark.parametrize(
    "sampler",
    [
        ({"step_size": 0.01, "noise_scale": 0.001}),
        ({"step_size": 0.1, "noise_scale": 0.01}),
        ({"step_size": 0.5, "noise_scale": 0.05}),
    ],
    indirect=True,
)
def test_contrastive_divergence_with_different_sampler_settings(
    energy_function, sampler
):
    """Test CD with different sampler settings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cd = ContrastiveDivergence(
        energy_function=energy_function,
        sampler=sampler,
        n_steps=10,
        persistent=False,
        device=device,
    )

    x = torch.randn(10, 2, device=device)
    loss, samples = cd(x)

    assert isinstance(loss, torch.Tensor)
    assert samples.shape == x.shape


def test_contrastive_divergence_chain_reset_when_batch_size_changes():
    """Test that the persistent chain is reset when batch size changes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_fn = GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2)).to(device)
    sampler = LangevinDynamics(
        energy_function=energy_fn, step_size=0.1, noise_scale=0.01, device=device
    )

    cd = ContrastiveDivergence(
        energy_function=energy_fn,
        sampler=sampler,
        n_steps=10,
        persistent=True,
        device=device,
    )

    # First call with batch size 10
    x1 = torch.randn(10, 2, device=device)
    loss1, samples1 = cd(x1)
    assert cd.chain is not None
    assert cd.chain.shape[0] == 10

    # Second call with different batch size
    x2 = torch.randn(20, 2, device=device)
    loss2, samples2 = cd(x2)
    assert cd.chain.shape[0] == 20

    # Should match the new batch size
    assert cd.chain.shape[0] == x2.shape[0]


@requires_cuda
def test_contrastive_divergence_cuda():
    """Test CD on CUDA if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    energy_fn = GaussianEnergy(
        mean=torch.zeros(2, device=device), cov=torch.eye(2, device=device)
    )
    sampler = LangevinDynamics(
        energy_function=energy_fn, step_size=0.1, noise_scale=0.01, device=device
    )

    cd = ContrastiveDivergence(
        energy_function=energy_fn,
        sampler=sampler,
        n_steps=10,
        persistent=False,
        device=device,
    )

    x = torch.randn(10, 2, device=device)
    loss, samples = cd(x)

    assert loss.device.type == "cuda"
    assert samples.device.type == "cuda"


def test_contrastive_divergence_deterministic():
    """Test that CD is deterministic with fixed random seeds."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_fn = GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2)).to(device)

    # First run
    torch.manual_seed(42)
    sampler1 = LangevinDynamics(
        energy_function=energy_fn, step_size=0.1, noise_scale=0.01, device=device
    )
    cd1 = ContrastiveDivergence(
        energy_function=energy_fn,
        sampler=sampler1,
        n_steps=10,
        persistent=False,
        device=device,
    )
    x = torch.randn(10, 2, device=device)
    torch.manual_seed(123)  # Seed before forward
    loss1, samples1 = cd1(x)

    # Second run with same seeds
    torch.manual_seed(42)
    sampler2 = LangevinDynamics(
        energy_function=energy_fn, step_size=0.1, noise_scale=0.01, device=device
    )
    cd2 = ContrastiveDivergence(
        energy_function=energy_fn,
        sampler=sampler2,
        n_steps=10,
        persistent=False,
        device=device,
    )
    torch.manual_seed(123)  # Same seed before forward
    loss2, samples2 = cd2(x)

    # Results should be identical
    assert torch.allclose(loss1, loss2)
    assert torch.allclose(samples1, samples2)


def test_contrastive_divergence_gradient_flow():
    """Test that gradients flow correctly through the loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a trainable energy function (MLP)
    energy_fn = MLPEnergy(input_dim=2, hidden_dim=8).to(device)
    sampler = LangevinDynamics(
        energy_function=energy_fn, step_size=0.1, noise_scale=0.01, device=device
    )

    cd = ContrastiveDivergence(
        energy_function=energy_fn,
        sampler=sampler,
        n_steps=5,
        persistent=False,
        device=device,
    )

    # Create some data
    x = torch.randn(10, 2, device=device)

    # Check that we can compute gradients
    optimizer = torch.optim.Adam(energy_fn.parameters(), lr=0.01)
    optimizer.zero_grad()

    loss, _ = cd(x)
    loss.backward()

    # Verify that gradients exist and are not None and non-zero
    found_non_zero_grad = False
    for param in energy_fn.parameters():
        assert param.grad is not None
        if torch.any(param.grad != 0):
            found_non_zero_grad = True

    # It's possible, though unlikely with random init and data,
    # that *some* specific parameter might have a zero gradient by chance.
    # A better overall check is that *at least one* parameter has a non-zero gradient.
    assert found_non_zero_grad, "No non-zero gradients found for any parameter."
