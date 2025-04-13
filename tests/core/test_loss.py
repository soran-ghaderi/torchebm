import pytest
import torch
import numpy as np
from torch import nn

from torchebm.core import (
    BaseLoss,
    BaseContrastiveDivergence,
    BaseEnergyFunction,
    BaseSampler,
)
from torchebm.core import GaussianEnergy
from torchebm.samplers import LangevinDynamics
from tests.conftest import requires_cuda


class MockLoss(BaseLoss):
    """A simple implementation of BaseLoss for testing."""

    def forward(self, x, *args, **kwargs):
        return torch.mean(x)


class MockCD(BaseContrastiveDivergence):
    """A simple implementation of BaseContrastiveDivergence for testing."""

    def forward(self, x, *args, **kwargs):
        batch_size = x.shape[0]

        # Use the sampler to generate negative samples
        if self.persistent:
            if self.chain is None or self.chain.shape[0] != batch_size:
                self.chain = torch.randn_like(x).detach()
            start_points = self.chain.to(self.device, dtype=self.dtype)
        else:
            start_points = x.detach()

        # Generate negative samples
        pred_samples = self.sampler.sample(
            x=start_points, n_steps=self.n_steps
        ).detach()

        # Update persistent chain if needed
        if self.persistent:
            self.chain = pred_samples.detach()

        # Compute loss
        loss = self.compute_loss(x, pred_samples)

        return loss, pred_samples

    def compute_loss(self, x, pred_x, *args, **kwargs):
        # Simple loss: difference in energy between positive and negative samples
        x_energy = self.energy_function(x)
        pred_x_energy = self.energy_function(pred_x)
        return torch.mean(x_energy - pred_x_energy)


@pytest.fixture
def energy_function():
    """Fixture to create a simple energy function for testing."""
    return GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2))


@pytest.fixture
def sampler(energy_function):
    """Fixture to create a Langevin dynamics sampler for testing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return LangevinDynamics(
        energy_function=energy_function, step_size=0.1, noise_scale=0.01, device=device
    )


@pytest.fixture
def mock_loss():
    """Fixture to create a mock loss function."""
    return MockLoss()


@pytest.fixture
def mock_cd(energy_function, sampler):
    """Fixture to create a mock contrastive divergence loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MockCD(
        energy_function=energy_function,
        sampler=sampler,
        n_steps=10,
        persistent=False,
        device=device,
    )


def test_base_loss_initialization():
    """Test that BaseLoss can be initialized properly."""
    loss = MockLoss()
    assert isinstance(loss, BaseLoss)
    assert isinstance(loss, nn.Module)


def test_base_loss_forward():
    """Test the forward method of BaseLoss."""
    loss = MockLoss()
    x = torch.randn(10, 2)
    result = loss(x)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([])  # Should be a scalar


def test_base_loss_to_device():
    """Test moving the loss to a specific device."""
    loss = MockLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss = loss.to(device)
    assert hasattr(loss, "device")
    assert loss.device == device


def test_base_contrastive_divergence_initialization(energy_function, sampler):
    """Test that BaseContrastiveDivergence can be initialized properly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cd = MockCD(
        energy_function=energy_function,
        sampler=sampler,
        n_steps=10,
        persistent=False,
        device=device,
    )
    assert isinstance(cd, BaseContrastiveDivergence)
    assert cd.energy_function == energy_function
    assert cd.sampler == sampler
    assert cd.n_steps == 10
    assert cd.persistent is False
    assert cd.device == device
    assert cd.chain is None


def test_base_contrastive_divergence_initialization_persistent(
    energy_function, sampler
):
    """Test BaseContrastiveDivergence initialization with persistence."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cd = MockCD(
        energy_function=energy_function,
        sampler=sampler,
        n_steps=10,
        persistent=True,
        device=device,
    )
    assert cd.persistent is True
    assert cd.chain is None  # Should start as None even when persistent=True


def test_base_contrastive_divergence_init_chain():
    """Test the initialization of the persistent chain."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss = MockCD(
        energy_function=GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2)),
        sampler=LangevinDynamics(
            energy_function=GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2)),
            step_size=0.1,
            noise_scale=0.01,
            device=device,
        ),
        n_steps=10,
        persistent=True,
        device=device,
    )

    # Initialize the chain
    shape = (32, 2)  # batch_size, input_dim
    chain = loss.initialize_persistent_chain(shape)

    assert chain.shape == shape
    assert chain.device.type == device
    assert chain.dtype == torch.float32


def test_base_contrastive_divergence_to_device(mock_cd):
    """Test moving the loss to a specific device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mock_cd = mock_cd.to(device)
    assert mock_cd.device == device


def test_contrastive_divergence_call_forward(mock_cd):
    """Test that __call__ correctly calls forward."""
    x = torch.randn(10, 2, device=mock_cd.device)
    result_call = mock_cd(x)
    result_forward = mock_cd.forward(x)

    # Both should return a tuple (loss, samples)
    assert isinstance(result_call, tuple)
    assert isinstance(result_forward, tuple)
    assert len(result_call) == 2
    assert len(result_forward) == 2

    # The loss should be a scalar tensor
    assert isinstance(result_call[0], torch.Tensor)
    assert result_call[0].shape == torch.Size([])


@pytest.mark.parametrize("n_steps", [1, 5, 10])
def test_contrastive_divergence_n_steps(energy_function, sampler, n_steps):
    """Test CD with different numbers of sampling steps."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cd = MockCD(
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


def test_contrastive_divergence_persistent(energy_function, sampler):
    """Test CD with persistence enabled."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cd = MockCD(
        energy_function=energy_function,
        sampler=sampler,
        n_steps=10,
        persistent=True,
        device=device,
    )

    # First call should initialize the chain
    x = torch.randn(10, 2, device=device)
    loss1, samples1 = cd(x)

    # Chain should now be set
    assert cd.chain is not None
    assert cd.chain.shape == x.shape

    # Second call should use the existing chain
    loss2, samples2 = cd(x)

    # The chain should have been updated
    assert torch.any(torch.ne(cd.chain, samples1))
    assert torch.allclose(cd.chain, samples2)


@requires_cuda
def test_contrastive_divergence_cuda():
    """Test CD on CUDA device if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    energy_function = GaussianEnergy(
        mean=torch.zeros(2, device=device), cov=torch.eye(2, device=device)
    )
    sampler = LangevinDynamics(
        energy_function=energy_function, step_size=0.1, noise_scale=0.01, device=device
    )

    cd = MockCD(
        energy_function=energy_function,
        sampler=sampler,
        n_steps=10,
        persistent=False,
        device=device,
    )

    x = torch.randn(10, 2, device=device)
    loss, samples = cd(x)

    assert loss.device.type == "cuda"
    assert samples.device.type == "cuda"


def test_contrastive_divergence_dtype():
    """Test CD with different data types."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    energy_function = GaussianEnergy(
        mean=torch.zeros(2, dtype=dtype, device=device),
        cov=torch.eye(2, dtype=dtype, device=device),
    )

    sampler = LangevinDynamics(
        energy_function=energy_function, step_size=0.1, noise_scale=0.01, device=device
    )

    cd = MockCD(
        energy_function=energy_function,
        sampler=sampler,
        n_steps=10,
        persistent=False,
        dtype=dtype,
        device=device,
    )

    x = torch.randn(10, 2, dtype=dtype, device=device)
    loss, samples = cd(x)

    assert loss.dtype == dtype
    assert samples.dtype == dtype


def test_compute_loss_method(mock_cd):
    """Test the compute_loss method separately."""
    device = mock_cd.device
    x = torch.randn(10, 2, device=device)
    pred_x = torch.randn(10, 2, device=device)

    loss = mock_cd.compute_loss(x, pred_x)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # Should be a scalar


def test_different_batch_sizes(energy_function, sampler):
    """Test CD with different batch sizes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cd = MockCD(
        energy_function=energy_function,
        sampler=sampler,
        n_steps=10,
        persistent=True,
        device=device,
    )

    # First batch size
    x1 = torch.randn(10, 2, device=device)
    loss1, samples1 = cd(x1)
    assert cd.chain.shape == (10, 2)

    # Different batch size
    x2 = torch.randn(20, 2, device=device)
    loss2, samples2 = cd(x2)
    assert cd.chain.shape == (20, 2)

    # The chain should have been re-initialized for the new batch size
    assert cd.chain.shape[0] == x2.shape[0]
