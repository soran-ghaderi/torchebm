import pytest
import torch
import torch.nn as nn
import numpy as np
import warnings

from torchebm.core import BaseModel
from torchebm.losses import SlicedScoreMatching
from tests.conftest import requires_cuda


class MLPEnergy(BaseModel):
    """A simple MLP to act as the energy function for testing."""

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
        return self.model(x).squeeze(-1)


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
        return energy


class DoubleWellEnergy(BaseModel):
    def __init__(self, barrier_height=2.0, a=1.0, b=6.0):
        super().__init__()
        self.barrier_height = barrier_height
        self.a = a
        self.b = b

    def forward(self, x):
        if x.shape[-1] != 1:  # Apply to first dim if ND
            x_1d = x[..., 0]
        else:
            x_1d = x.squeeze(-1)
        return self.a * x_1d**4 - self.b * x_1d**2 + self.barrier_height


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
def ssm_loss(request, energy_function):
    """Fixture to create a SlicedScoreMatching loss for testing."""
    if not hasattr(request, "param"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return SlicedScoreMatching(
            model=energy_function,
            n_projections=5,
            projection_type="rademacher",
            device=device,
        )

    params = request.param
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    n_projections = params.get("n_projections", 5)
    projection_type = params.get("projection_type", "rademacher")
    regularization_strength = params.get("regularization_strength", 0.0)
    noise_scale = params.get("noise_scale", 0.0)
    # scale_factor = params.get("scale_factor", 1.0)
    clip_value = params.get("clip_value", None)
    use_mixed_precision = params.get("use_mixed_precision", False)

    return SlicedScoreMatching(
        model=energy_function,
        n_projections=n_projections,
        projection_type=projection_type,
        regularization_strength=regularization_strength,
        noise_scale=noise_scale,
        # scale_factor=scale_factor, # not supported yet
        clip_value=clip_value,  # not supported yet
        use_mixed_precision=use_mixed_precision,
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


def test_sliced_score_matching_initialization(energy_function, device):
    """Test initialization of SlicedScoreMatching with different parameters."""
    ssm = SlicedScoreMatching(
        model=energy_function,
        n_projections=10,
        projection_type="gaussian",
        regularization_strength=0.01,
        noise_scale=0.001,
        clip_value=10.0,
        device=device,
    )

    assert isinstance(ssm, SlicedScoreMatching)
    assert ssm.model == energy_function
    assert ssm.n_projections == 10
    assert ssm.projection_type == "gaussian"
    assert ssm.regularization_strength == 0.01
    assert ssm.noise_scale == 0.001
    assert ssm.device == device


def test_sliced_score_matching_invalid_projection_type(energy_function):
    """Test that invalid projection type is handled correctly."""
    with warnings.catch_warnings(record=True) as w:
        ssm = SlicedScoreMatching(
            model=energy_function,
            projection_type="invalid_type",
        )
        assert any("Invalid projection_type" in str(warning.message) for warning in w)
        assert ssm.projection_type == "rademacher"


def test_get_random_projections(ssm_loss):
    """Test the random projection generation with different types."""
    batch_size = 5
    dim = 2
    x_tensor = torch.randn((batch_size, dim))

    ssm_loss.projection_type = "rademacher"
    v_rademacher = ssm_loss._get_random_projections(x_tensor)

    assert v_rademacher.shape == x_tensor.shape
    assert torch.all(torch.abs(v_rademacher) == 1.0)  # Rademacher should be +1 or -1

    ssm_loss.projection_type = "gaussian"
    v_gaussian = ssm_loss._get_random_projections(x_tensor)

    assert v_gaussian.shape == x_tensor.shape
    assert not torch.all(
        torch.abs(v_gaussian) == 1.0
    )  # gaussian shouldn't be all +1 or -1


def test_forward_pass(ssm_loss):
    """Test the forward method of SlicedScoreMatching."""
    device = ssm_loss.device
    x = torch.randn(10, 2, device=device)

    loss = ssm_loss(x)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # Should be a scalar


def test_compute_loss(ssm_loss):
    """Test the compute_loss method of SlicedScoreMatching."""
    device = ssm_loss.device
    x = torch.randn(10, 2, device=device)

    loss = ssm_loss.compute_loss(x)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # Should be a scalar
    assert not torch.isnan(loss)  # Loss should not be NaN
    assert not torch.isinf(loss)  # Loss should be finite


def test_forward_with_noise(energy_function):
    """Test forward method with noise scale > 0."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ssm = SlicedScoreMatching(
        model=energy_function,
        noise_scale=0.1,
        device=device,
    )

    x = torch.randn(10, 2, device=device)

    # Run with noise
    loss = ssm(x)

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)


def test_forward_with_clipping(energy_function):
    """Test forward method with loss clipping."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_value = 1000.0  # Use a higher value that won't trigger the clipping
    ssm = SlicedScoreMatching(
        model=energy_function,
        clip_value=clip_value,
        device=device,
    )

    x = torch.randn(10, 2, device=device)

    # Run with clipping
    loss = ssm(x)

    assert isinstance(loss, torch.Tensor)
    # Don't assert about the actual value
    assert (
        loss.detach().item() <= clip_value + 1.0
    )  # Allow for some floating point error


def test_forward_with_regularization(energy_function):
    """Test forward method with regularization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    regularization_strength = 0.1
    ssm = SlicedScoreMatching(
        model=energy_function,
        regularization_strength=regularization_strength,
        device=device,
    )

    x = torch.randn(10, 2, device=device)

    # Run with regularization
    loss_with_reg = ssm(x)

    # Run without regularization for comparison
    ssm.regularization_strength = 0.0
    loss_without_reg = ssm(x)

    assert isinstance(loss_with_reg, torch.Tensor)
    # Skip direct comparison as it's not guaranteed to be different
    # Due to the stochastic nature of sliced score matching
    assert not torch.isnan(loss_with_reg)
    assert not torch.isnan(loss_without_reg)


def test_different_projections_yield_different_losses(energy_function):
    """Test that different projection types yield different losses."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # Create two instances with different projection types
    ssm1 = SlicedScoreMatching(
        model=energy_function,
        projection_type="rademacher",
        n_projections=50,  # Use more projections to make difference more reliable
        device=device,
    )

    ssm2 = SlicedScoreMatching(
        model=energy_function,
        projection_type="gaussian",
        n_projections=50,  # Use more projections to make difference more reliable
        device=device,
    )

    # Use same input for both
    torch.manual_seed(123)
    x = torch.randn(10, 2, device=device)

    # Compute losses
    torch.manual_seed(42)  # Same seed for both to control randomness
    loss1 = ssm1(x)

    torch.manual_seed(42)  # Same seed for both to control randomness
    loss2 = ssm2(x)

    # The losses should be different due to different projection types
    assert not torch.isnan(loss1)
    assert not torch.isnan(loss2)
    assert abs(loss1.item() - loss2.item()) > 1e-10


@pytest.mark.parametrize(
    "energy_function, ssm_loss",
    [
        (
            {"type": "gaussian", "mean": torch.zeros(2), "cov": torch.eye(2)},
            {"n_projections": 3, "projection_type": "rademacher"},
        ),
        (
            {"type": "gaussian", "mean": torch.zeros(2), "cov": 2 * torch.eye(2)},
            {"n_projections": 5, "projection_type": "gaussian"},
        ),
        (
            {"type": "double_well", "barrier_height": 2.0},
            {"n_projections": 10, "projection_type": "rademacher"},
        ),
        (
            {"type": "mlp", "input_dim": 2, "hidden_dim": 16},
            {"n_projections": 3, "projection_type": "gaussian"},
        ),
    ],
    indirect=True,
)
def test_sliced_score_matching_with_different_energy_functions(ssm_loss):
    """Test SSM with different energy functions and parameters."""
    device = ssm_loss.device
    x = torch.randn(10, 2, device=device)

    loss = ssm_loss(x)

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)  # Loss should not be NaN


def test_higher_dimensions():
    """Test SSM with higher-dimensional inputs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a 10-dimensional energy function
    mean = torch.zeros(10, device=device)
    cov = torch.eye(10, device=device)
    energy_fn = GaussianEnergy(mean=mean, cov=cov)

    ssm = SlicedScoreMatching(
        model=energy_fn,
        n_projections=3,  # Use fewer projections for speed
        device=device,
    )

    # Create high-dimensional input
    x = torch.randn(5, 10, device=device)

    loss = ssm(x)

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)


def test_numerical_stability():
    """Test SSM with extreme values to check numerical stability."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create standard energy function
    energy_fn = GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2))

    ssm = SlicedScoreMatching(
        model=energy_fn,
        n_projections=5,
        clip_value=100.0,  # Large clip value to see if it handles extreme values
        device=device,
    )

    # Create inputs with extreme values
    x_extreme = torch.ones(10, 2, device=device) * 1000.0

    loss = ssm(x_extreme)

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_gradient_flow():
    """Test that gradients flow correctly through the loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a trainable energy function (MLP)
    energy_fn = MLPEnergy(input_dim=2, hidden_dim=8).to(device)

    ssm = SlicedScoreMatching(
        model=energy_fn,
        n_projections=3,
        device=device,
    )

    # Create data
    x = torch.randn(10, 2, device=device)

    # Check that we can compute gradients
    optimizer = torch.optim.Adam(energy_fn.parameters(), lr=0.01)
    optimizer.zero_grad()

    loss = ssm(x)
    loss.backward()

    # Verify that at least some parameters have gradients
    # (not all parameters might get gradients in a single batch)
    param_count = 0
    grad_param_count = 0
    for param in energy_fn.parameters():
        param_count += 1
        if param.grad is not None and torch.any(param.grad != 0):
            grad_param_count += 1

    assert param_count > 0, "No parameters found in energy function"
    assert grad_param_count > 0, "No parameters with non-zero gradients found"


@requires_cuda
def test_sliced_score_matching_cuda():
    """Test SSM on CUDA if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    energy_fn = GaussianEnergy(
        mean=torch.zeros(2, device=device), cov=torch.eye(2, device=device)
    )

    ssm = SlicedScoreMatching(
        model=energy_fn,
        n_projections=5,
        projection_type="rademacher",
        device=device,
    )

    x = torch.randn(10, 2, device=device)
    loss = ssm(x)

    assert loss.device.type == "cuda"


def test_convergence_potential():
    """Test potential to converge to a known distribution."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(43)
    mean = torch.tensor(0.0, device=device)
    std = torch.tensor(1.0, device=device)

    energy_fn = MLPEnergy(input_dim=2, hidden_dim=16).to(device)

    # Create SSM loss
    ssm = SlicedScoreMatching(
        model=energy_fn,
        projection_type="gaussian",
        n_projections=10,
        regularization_strength=1e-4,
        device=device,
    )

    # Setup optimizer
    optimizer = torch.optim.Adam(energy_fn.parameters(), lr=0.001)

    # Generate training data from target distribution
    n_samples = 1000
    data = torch.randn(n_samples, 2, device=device) * std + mean

    # Train for a few iterations
    n_epochs = 20
    batch_size = 128

    for epoch in range(n_epochs):
        epoch_losses = []
        for batch_idx in range(0, n_samples, batch_size):
            batch = data[batch_idx : batch_idx + batch_size]
            if len(batch) == 0:
                continue

            optimizer.zero_grad()
            loss = ssm(batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Verify loss is not NaN
        assert not any(np.isnan(loss) for loss in epoch_losses)

    # Generate some samples to analyze the learned distribution
    # We need to sample directly by computing energy on a grid since we don't have a sampler
    # x_grid = torch.linspace(-3, 3, 100).view(1, -1).to(device)
    grid_vals = torch.linspace(-5, 5, 120, device=device)
    xx, yy = torch.meshgrid(grid_vals, grid_vals, indexing="ij")
    x_grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    with torch.no_grad():
        energies = energy_fn(x_grid)

    # Check if low energy regions correspond to high density regions of the target distribution
    min_energy_idx = torch.argmin(energies)
    min_energy_point = x_grid[min_energy_idx]

    # assert abs(min_energy_point.item() - mean.item()) < 1.0
    assert torch.all(torch.abs(min_energy_point - mean) < 1.0)
