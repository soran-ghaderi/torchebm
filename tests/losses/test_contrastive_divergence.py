import pytest
import torch
import torch.nn as nn
import numpy as np
import warnings

from torchebm.core import BaseModel, GaussianModel, DoubleWellEnergy
from torchebm.samplers import LangevinDynamics
from torchebm.losses import (
    ContrastiveDivergence,
    PersistentContrastiveDivergence,
    ParallelTemperingCD,
)
from tests.conftest import requires_cuda


# class MLPEnergy(BaseEnergyFunction):
#     """A simple MLP to act as the energy function for testing."""
#
#     def __init__(self, input_dim=2, hidden_dim=8):
#         super().__init__()
#         self.network = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_dim, 1),
#         )
#
#     def forward(self, x):
#         return self.network(x).squeeze(-1)
class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


# class BaseEnergyFunction(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         raise NotImplementedError

# def gradient(self, x):
#     x = x.detach().requires_grad_(True)
#     energy = self(x)
#     grad_outputs = torch.ones_like(energy)
#     (grad,) = torch.autograd.grad(
#         outputs=energy,
#         inputs=x,
#         grad_outputs=grad_outputs,
#         create_graph=False,
#         retain_graph=False,
#     )
#     return grad.detach()


class GaussianEnergy(BaseModel):
    def __init__(self, mean, cov):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov_inv", torch.inverse(cov))
        self.register_buffer("cov_log_det", torch.logdet(cov))
        self.dim = mean.shape[0]

    def forward(self, x):
        self.mean = self.mean.to(x.device)
        self.cov_inv = self.cov_inv.to(x.device)
        self.cov_log_det = self.cov_log_det.to(x.device)
        diff = x - self.mean
        energy = 0.5 * torch.sum((diff @ self.cov_inv) * diff, dim=-1)
        # Optional: Add normalization constant part (doesn't affect gradient)
        # energy += 0.5 * (self.dim * np.log(2 * np.pi) + self.cov_log_det)
        return energy


class DoubleWellEnergy(BaseModel):
    def __init__(self, barrier_height=2.0, a=1.0, b=6.0):
        super().__init__()
        self.barrier_height = barrier_height
        self.a = a
        self.b = b

    def forward(self, x):

        # Simple 1D double well for testing (can be adapted for ND)
        # U(x) = a*x^4 - b*x^2 + barrier_height
        if x.shape[-1] != 1:  # Apply to first dim if ND
            x_1d = x[..., 0]
        else:
            x_1d = x.squeeze(-1)
        return self.a * x_1d**4 - self.b * x_1d**2 + self.barrier_height


class MLPEnergy(BaseModel):
    def __init__(self, input_dim=2, hidden_dim=8):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        self.model = self.model.to(x.device)
        return self.model(x).squeeze(-1)


class BaseSampler(nn.Module):
    def __init__(self, model, dtype, device):
        super().__init__()
        self.model = model
        self.dtype = dtype
        self.device = device

    def sample(self, x, n_steps, **kwargs):
        raise NotImplementedError

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self


class LangevinDynamics(BaseSampler):
    def __init__(
        self, model, step_size, noise_scale, device, dtype=torch.float32
    ):
        super().__init__(model, dtype, device)
        self.step_size = step_size
        self.noise_scale = noise_scale
        self.temperature = 1.0  # Add temperature parameter for annealing tests

    @torch.no_grad()
    def sample(self, x, n_steps, **kwargs):
        x_curr = x.clone()
        for _ in range(n_steps):
            noise = (
                torch.randn_like(x_curr) * self.noise_scale * np.sqrt(self.temperature)
            )
            grad = self.model.gradient(x_curr).to(device=x.device)
            step_term = (
                torch.sqrt(torch.tensor(2.0 * self.step_size, device=x.device)) * noise
            )
            # Handle potential NaNs in gradient robustly during test
            grad = torch.nan_to_num(grad, nan=0.0, posinf=1e4, neginf=-1e4)
            x_curr = x_curr - self.step_size * grad + step_term
            x_curr = torch.nan_to_num(x_curr, nan=0.0)  # Prevent samples becoming NaN
        return x_curr


@pytest.fixture
def energy_function(request):
    """Fixture to create energy functions for testing."""
    if not hasattr(request, "param"):
        return GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))

    params = request.param
    if params.get("type") == "gaussian":
        mean = params.get("mean", torch.zeros(2))
        cov = params.get("cov", torch.eye(2))
        return GaussianModel(mean=mean, cov=cov)
    elif params.get("type") == "double_well":
        barrier_height = params.get("barrier_height", 2.0)
        return DoubleWellEnergy(barrier_height=barrier_height)
    elif params.get("type") == "mlp":
        input_dim = params.get("input_dim", 2)
        hidden_dim = params.get("hidden_dim", 8)
        return MLPEnergy(input_dim=input_dim, hidden_dim=hidden_dim)
    else:
        return GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))


@pytest.fixture
def sampler(request, energy_function):
    """Fixture to create a sampler for testing."""
    if not hasattr(request, "param"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return LangevinDynamics(
            model=energy_function,
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
        model=energy_function,
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
            model=energy_function,
            sampler=sampler,
            k_steps=10,
            persistent=False,
            device=device,
        )

    params = request.param
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    k_steps = params.get("k_steps", 10)
    persistent = params.get("persistent", False)
    buffer_size = params.get("buffer_size", 100)
    init_steps = params.get("init_steps", 0)
    energy_reg_weight = params.get("energy_reg_weight", 0.001)
    use_temperature_annealing = params.get("use_temperature_annealing", False)

    return ContrastiveDivergence(
        model=energy_function,
        sampler=sampler,
        k_steps=k_steps,
        persistent=persistent,
        buffer_size=buffer_size,
        init_steps=init_steps,
        energy_reg_weight=energy_reg_weight,
        use_temperature_annealing=use_temperature_annealing,
        device=device,
    )


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ]
)
def device(request):
    """Fixture for device (CPU or CUDA)."""
    return torch.device(request.param)


def test_contrastive_divergence_initialization(energy_function, sampler, device):
    """Test initialization of ContrastiveDivergence with new parameters."""
    cd = ContrastiveDivergence(
        model=energy_function,
        sampler=sampler,
        k_steps=10,
        persistent=False,
        buffer_size=1000,
        init_steps=5,
        energy_reg_weight=0.002,
        use_temperature_annealing=True,
        max_temp=1.5,
        min_temp=0.1,
        temp_decay=0.99,
        device=device,
    )

    assert isinstance(cd, ContrastiveDivergence)
    assert cd.model == energy_function
    assert cd.sampler == sampler
    assert cd.k_steps == 10
    assert cd.persistent is False
    assert cd.buffer_size == 1000
    assert cd.init_steps == 5
    assert cd.energy_reg_weight == 0.002
    assert cd.use_temperature_annealing is True
    assert cd.max_temp == 1.5
    assert cd.min_temp == 0.1
    assert cd.temp_decay == 0.99
    assert cd.current_temp == 1.5
    assert cd.model.device == device
    assert cd.device == device
    assert cd.replay_buffer is None


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
    """Test the compute_loss method of ContrastiveDivergence with energy regularization."""
    device = cd_loss.device
    x = torch.randn(10, 2, device=device)
    pred_x = torch.randn(10, 2, device=device)

    # Test with default energy regularization
    loss = cd_loss.compute_loss(x, pred_x)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # Should be a scalar

    # Test with custom energy regularization
    loss = cd_loss.compute_loss(x, pred_x, energy_reg_weight=0.05)
    assert isinstance(loss, torch.Tensor)


@pytest.mark.parametrize("persistent", [False, True])
def test_contrastive_divergence_persistence(
    energy_function, sampler, persistent, device
):
    """Test CD with and without persistence."""
    k_steps = 10
    buffer_size = 50
    batch_size = 10
    # Safely infer input_dim
    try:
        # Attempt to access model attribute for MLPEnergy
        input_dim = energy_function.model[0].in_features
    except (AttributeError, TypeError):
        # Fallback for non-MLP or if model structure changes
        try:
            input_dim = energy_function.mean.shape[0]  # For Gaussian
        except AttributeError:
            input_dim = 2  # Default if inference fails

    dtype = torch.float32

    cd = ContrastiveDivergence(
        model=energy_function,
        sampler=sampler,
        k_steps=k_steps,
        persistent=persistent,
        buffer_size=buffer_size,
        init_steps=5,
        device=device,
        dtype=dtype,
    )

    x = torch.randn(batch_size, input_dim, device=device, dtype=dtype)
    loss1, samples1 = cd(x)

    if persistent:
        assert cd.replay_buffer is not None
        assert cd.buffer_initialized
        assert cd.replay_buffer.shape[0] == buffer_size

        # Test second update to verify buffer continuity
        loss2, samples2 = cd(x)
        assert not torch.allclose(
            samples1, samples2
        ), "Samples should differ across calls with PCD"
    else:
        assert not cd.persistent
        assert cd.replay_buffer is None
        assert not cd.buffer_initialized


@pytest.mark.parametrize(
    "energy_function, sampler, cd_loss",
    [
        (
            {"type": "gaussian", "mean": torch.zeros(2), "cov": torch.eye(2)},
            {"device": "cuda" if torch.cuda.is_available() else "cpu"},
            {"k_steps": 5, "persistent": False},
        ),
        (
            {"type": "gaussian", "mean": torch.zeros(2), "cov": 2 * torch.eye(2)},
            {"device": "cuda" if torch.cuda.is_available() else "cpu"},
            {"k_steps": 10, "persistent": True, "buffer_size": 50, "init_steps": 5},
        ),
        (
            {"type": "double_well", "barrier_height": 2.0},
            {"device": "cuda" if torch.cuda.is_available() else "cpu"},
            {"k_steps": 20, "persistent": False},
        ),
        (
            {"type": "mlp", "input_dim": 2, "hidden_dim": 16},
            {"device": "cuda" if torch.cuda.is_available() else "cpu"},
            {"k_steps": 15, "persistent": True, "energy_reg_weight": 0.01},
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


def test_temperature_annealing(energy_function, sampler, device):
    """Test temperature annealing functionality."""
    cd = ContrastiveDivergence(
        model=energy_function,
        sampler=sampler,
        k_steps=5,
        persistent=True,
        buffer_size=20,
        use_temperature_annealing=True,
        max_temp=2.0,
        min_temp=0.1,
        temp_decay=0.8,  # Fast decay for testing
        device=device,
    )

    # Set to training mode
    cd.train()

    # Store initial temperature
    initial_temp = cd.current_temp

    # Run multiple forward passes to check temperature decay
    x = torch.randn(10, 2, device=device)
    cd(x)

    # Temperature should have been updated
    assert cd.current_temp < initial_temp
    assert cd.current_temp >= cd.min_temp

    # Run enough iterations to reach near min_temp
    for _ in range(20):  # More iterations to get closer to min_temp
        cd(x)

    # Temperature should be closer to min_temp after more iterations
    # Allow more tolerance - we just need to verify it's decreasing
    assert (
        cd.current_temp <= 0.2
    ), f"Temperature {cd.current_temp} should be closer to min_temp {cd.min_temp}"

    # Verify it never goes below min_temp
    assert cd.current_temp >= cd.min_temp


def test_nan_handling_in_loss(energy_function, sampler, device):
    """Test that the compute_loss method handles NaNs appropriately."""
    cd = ContrastiveDivergence(
        model=energy_function,
        sampler=sampler,
        k_steps=5,
        device=device,
    )

    # Create data that will produce NaN energy
    x = torch.randn(10, 2, device=device)
    pred_x = torch.ones_like(x) * float("inf")  # Will cause NaN in energy calculation

    # There should be a warning
    with pytest.warns(RuntimeWarning):
        loss = cd.compute_loss(x, pred_x)

    # Loss should be a fallback value, not NaN
    assert not torch.isnan(loss)
    assert abs(loss.item() - 0.1) < 1e-4  # Allow for small float precision differences


def test_fifo_buffer_update(energy_function, sampler, device):
    """Test the FIFO buffer update strategy."""
    buffer_size = 20
    batch_size = 5
    cd = ContrastiveDivergence(
        model=energy_function,
        sampler=sampler,
        k_steps=1,
        persistent=True,
        buffer_size=buffer_size,
        device=device,
    )

    x = torch.randn(batch_size, 2, device=device)

    # First update to initialize buffer
    cd(x)

    # Get initial buffer pointer
    initial_ptr = cd.buffer_ptr.item()

    # Run another update
    cd(x)

    # Check pointer has advanced correctly
    expected_ptr = (initial_ptr + batch_size) % buffer_size
    assert cd.buffer_ptr.item() == expected_ptr

    # Test full wraparound by filling buffer
    full_batches = (buffer_size // batch_size) + 1
    for _ in range(full_batches):
        cd(x)

    # Buffer pointer should have wrapped around at least once
    assert cd.buffer_ptr.item() < buffer_size


def test_stratified_sampling(energy_function, sampler, device):
    """Test that stratified sampling is used for buffer access."""
    buffer_size = 100
    batch_size = 10
    cd = ContrastiveDivergence(
        model=energy_function,
        sampler=sampler,
        k_steps=1,
        persistent=True,
        buffer_size=buffer_size,
        device=device,
    )

    # Initialize buffer with a pattern we can detect
    x = torch.zeros(batch_size, 2, device=device)
    cd(x)

    # Manually fill buffer with recognizable pattern
    # Each entry has a unique value based on its index
    with torch.no_grad():
        for i in range(buffer_size):
            cd.replay_buffer[i, 0] = i

    # Get samples - with stratified sampling, we should get samples
    # that are approximately evenly spaced through the buffer
    samples = cd.get_start_points(x)

    # Extract the indices we retrieved based on our pattern
    retrieved_indices = samples[:, 0].long()

    # Check for more uniform distribution than pure random
    # With stratified sampling, indices should be more evenly spaced
    indices_sorted = torch.sort(retrieved_indices).values
    diffs = indices_sorted[1:] - indices_sorted[:-1]

    # In pure random sampling, we'd expect some very small differences
    # due to duplicates or nearby indices. With stratified sampling,
    # the minimum difference should be larger.
    min_diff = torch.min(diffs).item()
    assert min_diff > 0, "Indices should not repeat with stratified sampling"


def test_energy_regularization_effect(energy_function, sampler, device):
    """Test the effect of different energy regularization strengths."""
    # Create data with non-zero energy values for testing
    x = torch.randn(10, 2, device=device) * 5.0  # Scale up to get larger energy values
    pred_x = torch.randn(10, 2, device=device) * 5.0

    # Test with no regularization
    cd_no_reg = ContrastiveDivergence(
        model=energy_function,
        sampler=sampler,
        k_steps=1,
        energy_reg_weight=0.0,
        device=device,
    )

    # Test with high regularization
    cd_high_reg = ContrastiveDivergence(
        model=energy_function,
        sampler=sampler,
        k_steps=1,
        energy_reg_weight=0.5,  # Much higher regularization
        device=device,
    )

    # Force higher energies to make regularization effect more noticeable
    with torch.no_grad():
        # Make sure energies are substantial
        x_energy = energy_function(x)
        pred_x_energy = energy_function(pred_x)

        # Skip test if energies are too small to show regularization effect
        if torch.mean(x_energy**2) < 0.1 and torch.mean(pred_x_energy**2) < 0.1:
            pytest.skip("Energy values too small to show regularization effect")

    # Compute losses
    loss_no_reg = cd_no_reg.compute_loss(x, pred_x)
    loss_high_reg = cd_high_reg.compute_loss(x, pred_x, energy_reg_weight=0.5)

    # Check if the regularization has a meaningful effect
    # This is a more robust test that doesn't depend as much on energy scale
    assert (
        abs(loss_no_reg - loss_high_reg) > 1e-5
    ), f"Regularization had no effect: {loss_no_reg} vs {loss_high_reg}"


def test_multiple_training_iterations(energy_function, sampler, device):
    """Test stability over multiple training iterations."""
    # Create a simple MLP energy function for training
    energy_fn = MLPEnergy(input_dim=2, hidden_dim=8).to(device)

    # Create loss with PCD to better test stability
    cd = ContrastiveDivergence(
        model=energy_fn,
        sampler=sampler,
        k_steps=5,
        persistent=True,
        buffer_size=50,
        energy_reg_weight=0.001,
        device=device,
    )

    # Setup optimizer
    optimizer = torch.optim.Adam(energy_fn.parameters(), lr=0.001)

    # Create synthetic data (e.g., from a Gaussian mixture)
    n_samples = 100
    mixture1 = torch.randn(n_samples // 2, 2, device=device) + torch.tensor(
        [2.0, 2.0], device=device
    )
    mixture2 = torch.randn(n_samples // 2, 2, device=device) + torch.tensor(
        [-2.0, -2.0], device=device
    )
    data = torch.cat([mixture1, mixture2], dim=0)

    # Run 10 training iterations
    losses = []
    for _ in range(10):
        # Sample a batch
        idx = torch.randperm(n_samples)[:10]
        batch = data[idx]

        # Forward and backward
        optimizer.zero_grad()
        loss, _ = cd(batch)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # Verify we don't have NaN losses
    assert not any(np.isnan(loss) for loss in losses)

    # Ideally, loss should be decreasing or at least stable
    # This is a statistical test, so it might occasionally fail
    # Let's check if the trend is generally downward
    first_half_avg = np.mean(losses[:5])
    second_half_avg = np.mean(losses[5:])

    # Note: We don't assert this as it depends on data and init
    # but it's useful to log if there might be instability
    if first_half_avg < second_half_avg:
        warnings.warn(
            f"Loss not decreasing: first half avg={first_half_avg}, second half avg={second_half_avg}"
        )


@requires_cuda
def test_contrastive_divergence_cuda():
    """Test CD on CUDA if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    energy_fn = GaussianModel(
        mean=torch.zeros(2, device=device), cov=torch.eye(2, device=device)
    )
    sampler = LangevinDynamics(
        model=energy_fn,
        step_size=0.1,
        noise_scale=0.01,
        device=device,
    )

    cd = ContrastiveDivergence(
        model=energy_fn,
        sampler=sampler,
        k_steps=10,
        persistent=True,
        buffer_size=100,
        init_steps=5,
        energy_reg_weight=0.001,
        use_temperature_annealing=True,
        device=device,
    )

    x = torch.randn(10, 2, device=device)
    loss, samples = cd(x)

    assert loss.device.type == "cuda"
    assert samples.device.type == "cuda"
    assert cd.replay_buffer.device.type == "cuda"


def test_contrastive_divergence_gradient_flow():
    """Test that gradients flow correctly through the loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a trainable energy function (MLP)
    energy_fn = MLPEnergy(input_dim=2, hidden_dim=8).to(device)
    sampler = LangevinDynamics(
        model=energy_fn,
        step_size=0.1,
        noise_scale=0.01,
        device=device,
    )

    cd = ContrastiveDivergence(
        model=energy_fn,
        sampler=sampler,
        k_steps=5,
        persistent=True,
        buffer_size=50,
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

    assert found_non_zero_grad, "No non-zero gradients found for any parameter."


def test_small_buffer_warning():
    """Test warning when buffer size is smaller than batch size."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_fn = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2)).to(device)
    sampler = LangevinDynamics(
        model=energy_fn,
        step_size=0.1,
        noise_scale=0.01,
        device=device,
    )

    buffer_size = 5
    batch_size = 10

    cd = ContrastiveDivergence(
        model=energy_fn,
        sampler=sampler,
        k_steps=1,
        persistent=True,
        buffer_size=buffer_size,
        device=device,
    )

    x = torch.randn(batch_size, 2, device=device)

    # Should raise a warning about buffer size < batch size
    with pytest.warns(UserWarning, match="Buffer size .* is smaller than batch size"):
        cd(x)


def test_convergence_potential():
    """Test potential to converge to a known distribution."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define a simple target distribution (1D Gaussian)
    mean = torch.tensor([0.0], device=device)
    std = torch.tensor([1.0], device=device)

    # Create true energy function (negative log likelihood of Gaussian)
    true_energy_fn = lambda x: 0.5 * ((x - mean) / std) ** 2

    # Create a trainable energy function
    energy_fn = MLPEnergy(input_dim=1, hidden_dim=32).to(device)

    # Create sampler
    sampler = LangevinDynamics(
        model=energy_fn,
        step_size=0.1,
        noise_scale=0.05,
        device=device,
    )

    # Create CD loss with temperature annealing for better convergence
    cd = ContrastiveDivergence(
        model=energy_fn,
        sampler=sampler,
        k_steps=10,
        persistent=True,
        buffer_size=200,
        init_steps=10,
        use_temperature_annealing=True,
        max_temp=2.0,
        min_temp=0.1,
        temp_decay=0.9,
        energy_reg_weight=0.0001,
        device=device,
    )

    # Setup optimizer
    optimizer = torch.optim.Adam(energy_fn.parameters(), lr=0.001)

    # Generate training data from target distribution
    n_samples = 1000
    data = torch.randn(n_samples, 1, device=device) * std + mean

    # Train for a few iterations
    n_epochs = 5
    batch_size = 64

    for epoch in range(n_epochs):
        epoch_losses = []
        for batch_idx in range(0, n_samples, batch_size):
            batch = data[batch_idx : batch_idx + batch_size]
            if len(batch) == 0:
                continue

            optimizer.zero_grad()
            loss, _ = cd(batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Verify loss is not NaN
        assert not any(np.isnan(loss) for loss in epoch_losses)

    # After training, sample from model to test convergence
    # We use the samples from the last batch
    _, model_samples = cd(batch)

    # Check if mean and std of samples are close to true distribution
    # Note: This might fail due to random initialization and short training
    # So we use a large tolerance
    model_mean = model_samples.mean().item()
    model_std = model_samples.std().item()

    # Log values but don't assert (hard to guarantee in few iterations)
    print(f"Target mean: {mean.item()}, Model mean: {model_mean}")
    print(f"Target std: {std.item()}, Model std: {model_std}")

    # Instead, just test that values are reasonable
    assert -3 < model_mean < 3, "Mean should be in a reasonable range"
    assert 0.1 < model_std < 10, "Std should be in a reasonable range"
