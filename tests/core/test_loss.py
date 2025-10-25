import pytest
import torch
import numpy as np
from torch import nn

from torchebm.core import (
    BaseLoss,
    BaseContrastiveDivergence,
    BaseModel,
    BaseSampler,
)
from torchebm.core import GaussianModel
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
            if self.replay_buffer is None or self.replay_buffer.shape[0] != batch_size:
                self.replay_buffer = torch.randn_like(x).detach()
            start_points = self.replay_buffer.to(self.device, dtype=self.dtype)
        else:
            start_points = x.detach()

        # Generate negative samples
        pred_samples = self.sampler.sample(
            x=start_points, n_steps=self.k_steps
        ).detach()

        # Update persistent chain if needed
        if self.persistent:
            self.replay_buffer = pred_samples.detach()

        # Compute loss
        loss = self.compute_loss(x, pred_samples)

        return loss, pred_samples

    def compute_loss(self, x, pred_x, *args, **kwargs):
        # Simple loss: difference in energy between positive and negative samples
        x_energy = self.model(x)
        pred_x_energy = self.model(pred_x)
        # Ensure the result maintains the correct dtype
        loss = torch.mean(x_energy - pred_x_energy)
        # Explicitly cast to the requested dtype to maintain consistency
        return loss.to(dtype=self.dtype)


@pytest.fixture
def energy_function():
    """Fixture to create a simple energy function for testing."""
    return GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))


@pytest.fixture
def sampler(energy_function):
    """Fixture to create a Langevin dynamics sampler for testing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return LangevinDynamics(
        model=energy_function,
        step_size=0.1,
        noise_scale=0.01,
        device=device,
    )


@pytest.fixture
def mock_loss():
    """Fixture to create a mock loss function."""
    return MockLoss()


@pytest.fixture
def mock_cd(energy_function, sampler):
    """Fixture to create a mock contrastive divergence loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    return MockCD(
        model=energy_function,
        sampler=sampler,
        k_steps=10,
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss = loss.to(device)
    assert hasattr(loss, "device")
    assert loss.device == device


def test_base_contrastive_divergence_initialization(energy_function, sampler):
    """Test that BaseContrastiveDivergence can be initialized properly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    cd = MockCD(
        model=energy_function,
        sampler=sampler,
        k_steps=10,
        persistent=False,
        device=device,
    )
    assert isinstance(cd, BaseContrastiveDivergence)
    assert cd.model == energy_function
    assert cd.sampler == sampler
    assert cd.k_steps == 10
    assert cd.persistent is False
    assert cd.device == device
    assert cd.replay_buffer is None


def test_base_contrastive_divergence_initialization_persistent(
    energy_function, sampler
):
    """Test BaseContrastiveDivergence initialization with persistence."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cd = MockCD(
        model=energy_function,
        sampler=sampler,
        k_steps=10,
        persistent=True,
        device=device,
    )
    assert cd.persistent is True
    assert cd.replay_buffer is None  # Should start as None even when persistent=True


def test_base_contrastive_divergence_initialize_buffer():
    """Test the initialization of the persistent chain."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss = MockCD(
        model=GaussianModel(mean=torch.zeros(2), cov=torch.eye(2)),
        sampler=LangevinDynamics(
            model=GaussianModel(mean=torch.zeros(2), cov=torch.eye(2)),
            step_size=0.1,
            noise_scale=0.01,
            device=device,
        ),
        k_steps=10,
        persistent=True,
        buffer_size=32,
        device=device,
    )

    # Initialize the replay buffer
    shape = (32, 2)  # batch_size, input_dim
    replay_buffer = loss.initialize_buffer((2,))

    assert replay_buffer.shape == shape
    assert replay_buffer.device.type == device
    assert replay_buffer.dtype == torch.float32


def test_base_contrastive_divergence_to_device(mock_cd):
    """Test moving the loss to a specific device."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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


@pytest.mark.parametrize("k_steps", [1, 5, 10])
def test_contrastive_divergence_k_steps(energy_function, sampler, k_steps):
    """Test CD with different numbers of sampling steps."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cd = MockCD(
        model=energy_function,
        sampler=sampler,
        k_steps=k_steps,
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
        model=energy_function,
        sampler=sampler,
        k_steps=10,
        persistent=True,
        device=device,
    )

    # First call should initialize the replay buffer
    x = torch.randn(10, 2, device=device)
    loss1, samples1 = cd(x)

    # replay buffer should now be set
    assert cd.replay_buffer is not None
    assert cd.replay_buffer.shape == x.shape

    # Second call should use the existing replay_buffer
    loss2, samples2 = cd(x)

    # The replay_buffer should have been updated
    assert torch.any(torch.ne(cd.replay_buffer, samples1))
    assert torch.allclose(cd.replay_buffer, samples2)


@requires_cuda
def test_contrastive_divergence_cuda():
    """Test CD on CUDA device if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    energy_function = GaussianModel(
        mean=torch.zeros(2, device=device), cov=torch.eye(2, device=device)
    )
    sampler = LangevinDynamics(
        model=energy_function,
        step_size=0.1,
        noise_scale=0.01,
        device=device,
    )

    cd = MockCD(
        model=energy_function,
        sampler=sampler,
        k_steps=10,
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

    energy_function = GaussianModel(
        mean=torch.zeros(2, dtype=dtype, device=device),
        cov=torch.eye(2, dtype=dtype, device=device),
    )

    sampler = LangevinDynamics(
        model=energy_function,
        step_size=0.1,
        noise_scale=0.01,
        device=device,
        dtype=dtype,
    )

    cd = MockCD(
        model=energy_function,
        sampler=sampler,
        k_steps=10,
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
        model=energy_function,
        sampler=sampler,
        k_steps=10,
        persistent=True,
        device=device,
    )

    # First batch size
    x1 = torch.randn(10, 2, device=device)
    loss1, samples1 = cd(x1)
    assert cd.replay_buffer.shape == (10, 2)

    # Different batch size
    x2 = torch.randn(20, 2, device=device)
    loss2, samples2 = cd(x2)
    assert cd.replay_buffer.shape == (20, 2)

    # The replay_buffer should have been re-initialized for the new batch size
    assert cd.replay_buffer.shape[0] == x2.shape[0]
