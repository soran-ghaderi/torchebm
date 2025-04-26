import pytest
import torch
import torch.nn as nn
import numpy as np

from torchebm.core import BaseEnergyFunction, GaussianEnergy, DoubleWellEnergy
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


class GaussianEnergy(BaseEnergyFunction):
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


class DoubleWellEnergy(BaseEnergyFunction):
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


class MLPEnergy(BaseEnergyFunction):
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
    def __init__(self, energy_function, dtype, device):
        super().__init__()
        self.energy_function = energy_function
        self.dtype = dtype
        self.device = device

    def sample(self, x, n_steps, **kwargs):
        raise NotImplementedError

    def to(self, device):
        self.device = device
        self.energy_function.to(device)
        return self


class LangevinDynamics(BaseSampler):
    def __init__(
        self, energy_function, step_size, noise_scale, device, dtype=torch.float32
    ):
        super().__init__(energy_function, dtype, device)
        self.step_size = step_size
        self.noise_scale = noise_scale

    @torch.no_grad()
    def sample(self, x, n_steps, **kwargs):
        x_curr = x.clone()
        for _ in range(n_steps):
            noise = torch.randn_like(x_curr) * self.noise_scale
            grad = self.energy_function.gradient(x_curr).to(device=x.device)
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
            k_steps=10,
            persistent=False,
            device=device,
        )

    params = request.param
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    k_steps = params.get("k_steps", 10)
    persistent = params.get("persistent", False)

    return ContrastiveDivergence(
        energy_function=energy_function,
        sampler=sampler,
        k_steps=k_steps,
        persistent=persistent,
        device=device,
    )


def test_contrastive_divergence_initialization(energy_function, sampler, device):
    """Test initialization of ContrastiveDivergence."""
    cd = ContrastiveDivergence(
        energy_function=energy_function,
        sampler=sampler,
        k_steps=10,
        persistent=False,
        device=device,
    )

    assert isinstance(cd, ContrastiveDivergence)
    assert cd.energy_function == energy_function
    assert cd.sampler == sampler
    assert cd.k_steps == 10
    assert cd.persistent is False
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
    """Test the compute_loss method of ContrastiveDivergence."""
    device = cd_loss.device
    x = torch.randn(10, 2, device=device)
    pred_x = torch.randn(10, 2, device=device)

    loss = cd_loss.compute_loss(x, pred_x)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # Should be a scalar


@pytest.mark.parametrize("persistent", [False, True])
def test_contrastive_divergence_persistence(
    energy_function, sampler, persistent, device
):
    """Test CD with and without persistence."""
    k_steps = 10
    buffer_size = 50  # Define a buffer size for the PCD case
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

    # Ensure energy function and sampler are on the correct device (handled by fixtures)
    cd = ContrastiveDivergence(
        energy_function=energy_function,
        sampler=sampler,
        k_steps=k_steps,
        persistent=persistent,
        buffer_size=buffer_size,
        init_steps=0,
        device=device,  # 'device' now correctly receives "cpu" or "cuda"
        dtype=dtype,
    )

    # This call should now work correctly
    x = torch.randn(batch_size, input_dim, device=device, dtype=dtype)

    # The rest of your test logic...
    loss1, samples1 = cd(x)

    if persistent:
        assert cd.replay_buffer is not None
        assert cd.buffer_initialized  # Check initialization flag
        # Check update - buffer content should roughly match samples after first step if buffer size allows
        # This check needs refinement based on update strategy (random vs FIFO)
        # For random replacement and batch_size <= buffer_size:
        num_samples_in_buffer = min(batch_size, buffer_size)
        # A loose check: are the samples found *anywhere* in the buffer?
        # This is complex with random replacement. A simpler check might be:
        assert cd.replay_buffer.shape[0] == buffer_size

    else:
        assert not cd.persistent
        assert cd.replay_buffer is None  # Should remain None for non-persistent
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
            {"k_steps": 10, "persistent": True},
        ),
        (
            {"type": "double_well", "barrier_height": 2.0},
            {"device": "cuda" if torch.cuda.is_available() else "cpu"},
            {"k_steps": 20, "persistent": False},
        ),
        (
            {"type": "mlp", "input_dim": 2, "hidden_dim": 16},
            {"device": "cuda" if torch.cuda.is_available() else "cpu"},
            {"k_steps": 15, "persistent": True},
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


@pytest.mark.parametrize("k_steps", [1, 5, 20])
def test_contrastive_divergence_k_steps_effect(energy_function, sampler, k_steps):
    """Test the effect of different numbers of MCMC steps."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create CD with specific k_steps
    cd = ContrastiveDivergence(
        energy_function=energy_function,
        sampler=sampler,
        k_steps=k_steps,
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
        k_steps=10,
        persistent=False,
        device=device,
    )

    x = torch.randn(10, 2, device=device)
    batch_size = x.shape[0]
    loss, samples = cd(x)

    assert isinstance(loss, torch.Tensor)
    assert samples[:batch_size].shape == x.shape


def test_pcd_buffer_persistence_across_batch_sizes(device):
    """Test that the PCD buffer persists (fixed size) and is updated across varying batch sizes."""
    input_dim = 2
    buffer_size = 100
    k_steps = 5
    dtype = torch.float32

    # Use consistent energy function and sampler setup
    energy_fn = (
        GaussianEnergy(
            mean=torch.zeros(input_dim, device=device),
            cov=torch.eye(input_dim, device=device),
        )
        .to(device)
        .to(dtype)
    )
    sampler = LangevinDynamics(
        energy_function=energy_fn,
        step_size=0.1,
        noise_scale=0.01,
        device=device,
        dtype=dtype,
    )
    cd_loss = ContrastiveDivergence(
        energy_function=energy_fn,
        sampler=sampler,
        k_steps=k_steps,
        persistent=True,  # Ensure PCD is active
        buffer_size=buffer_size,
        init_steps=10,  # Give buffer some reasonable initialization
        device=device,
        dtype=dtype,
    )

    # --- First call with batch size 10 ---
    batch_size_1 = 10
    x1 = torch.randn(batch_size_1, input_dim, device=device, dtype=dtype)
    loss1, samples1 = cd_loss(x1)

    # Check buffer state after first call
    assert cd_loss.persistent  # Verify mode
    assert cd_loss.buffer_initialized
    assert cd_loss.replay_buffer is not None
    assert cd_loss.replay_buffer.shape == (
        buffer_size,
        input_dim,
    )  # Check shape matches config
    assert cd_loss.replay_buffer.device == torch.device(device)
    assert cd_loss.replay_buffer.dtype == dtype
    buffer_after_b1 = cd_loss.replay_buffer.clone()  # Store state
    buffer_id_after_b1 = id(cd_loss.replay_buffer)  # Store object ID

    # --- Second call with different batch size (e.g., 20) ---
    batch_size_2 = 20
    x2 = torch.randn(batch_size_2, input_dim, device=device, dtype=dtype)
    loss2, samples2 = cd_loss(x2)

    # --- Assertions for Persistence and Update ---
    # Buffer should still be the same object and correctly configured
    assert cd_loss.buffer_initialized
    assert (
        id(cd_loss.replay_buffer) == buffer_id_after_b1
    )  # Check it's the *same* buffer object
    assert cd_loss.replay_buffer.shape == (
        buffer_size,
        input_dim,
    )  # Shape must remain unchanged

    # Buffer content should have been updated (unless sampler/k_steps leads to identical samples, highly unlikely)
    assert not torch.allclose(
        cd_loss.replay_buffer, buffer_after_b1, atol=1e-5
    ), "Buffer content did not change after second PCD step with different batch size"

    # Check shapes of outputs from the second call are correct
    assert loss2.shape == torch.Size([])
    assert samples2.shape == x2.shape  # Samples match input batch size


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
        energy_function=energy_fn,
        step_size=0.1,
        noise_scale=0.01,
        device=device,
    )

    cd = ContrastiveDivergence(
        energy_function=energy_fn,
        sampler=sampler,
        k_steps=10,
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
        energy_function=energy_fn,
        step_size=0.1,
        noise_scale=0.01,
        device=device,
    )
    cd1 = ContrastiveDivergence(
        energy_function=energy_fn,
        sampler=sampler1,
        k_steps=10,
        persistent=False,
        device=device,
    )
    x = torch.randn(10, 2, device=device)
    torch.manual_seed(123)  # Seed before forward
    loss1, samples1 = cd1(x)

    # Second run with same seeds
    torch.manual_seed(42)
    sampler2 = LangevinDynamics(
        energy_function=energy_fn,
        step_size=0.1,
        noise_scale=0.01,
        device=device,
    )
    cd2 = ContrastiveDivergence(
        energy_function=energy_fn,
        sampler=sampler2,
        k_steps=10,
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
        energy_function=energy_fn,
        step_size=0.1,
        noise_scale=0.01,
        device=device,
    )

    cd = ContrastiveDivergence(
        energy_function=energy_fn,
        sampler=sampler,
        k_steps=5,
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


# new tests for PersistentContrastiveDivergence


@pytest.fixture(
    params=[
        {"type": "gaussian", "mean": torch.tensor([0.0, 0.0]), "cov": torch.eye(2)},
        {
            "type": "gaussian",
            "mean": torch.tensor([1.0, -1.0]),
            "cov": torch.diag(torch.tensor([0.5, 1.5])),
        },
        {"type": "double_well", "barrier_height": 2.0},  # Note: Only uses first dim
        {"type": "mlp", "input_dim": 2, "hidden_dim": 8},
    ]
)
def energy_function_config(request):
    """Provides configuration for different energy functions."""
    return request.param


@pytest.fixture
def energy_function(energy_function_config):
    """Fixture to create energy functions based on config."""
    params = energy_function_config
    if params.get("type") == "gaussian":
        return GaussianEnergy(mean=params["mean"], cov=params["cov"])
    elif params.get("type") == "double_well":
        return DoubleWellEnergy(barrier_height=params["barrier_height"])
    elif params.get("type") == "mlp":
        return MLPEnergy(input_dim=params["input_dim"], hidden_dim=params["hidden_dim"])
    else:  # Default fallback
        return GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2))


@pytest.fixture(
    params=[
        {"step_size": 0.1, "noise_scale": 0.05},
        {"step_size": 0.01, "noise_scale": 0.01},
    ]
)
def sampler_config(request):
    """Provides configuration for the sampler."""
    return request.param


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
    return request.param


@pytest.fixture
def sampler(sampler_config, energy_function, device):
    """Fixture to create a LangevinDynamics sampler."""
    energy_function = energy_function.to(
        device
    )  # Ensure energy function is on the correct device
    return LangevinDynamics(
        energy_function=energy_function,
        step_size=sampler_config["step_size"],
        noise_scale=sampler_config["noise_scale"],
        device=device,
        dtype=torch.float32,  # Use float32 for stability in tests
    )


@pytest.fixture(
    params=[
        # --- Non-Persistent (CD) ---
        {"k_steps": 1, "persistent": False},
        {"k_steps": 10, "persistent": False},
        # --- Persistent (PCD) ---
        {"k_steps": 1, "persistent": True, "buffer_size": 50, "init_steps": 0},
        {"k_steps": 5, "persistent": True, "buffer_size": 100, "init_steps": 10},
        {
            "k_steps": 1,
            "persistent": True,
            "buffer_size": 20,
            "init_steps": 5,
        },  # Test buffer < typical batch
    ]
)
def cd_loss_config(request):
    """Provides configuration for ContrastiveDivergence loss."""
    return request.param


@pytest.fixture
def cd_loss(cd_loss_config, energy_function, sampler, device):
    """Fixture to create a ContrastiveDivergence loss instance."""
    # Sampler fixture already ensures energy_function is on the right device
    return ContrastiveDivergence(
        energy_function=energy_function,  # Already on `device` via `sampler` fixture
        sampler=sampler,  # Already on `device`
        k_steps=cd_loss_config["k_steps"],
        persistent=cd_loss_config["persistent"],
        buffer_size=cd_loss_config.get("buffer_size", 100),  # Default if not specified
        init_steps=cd_loss_config.get("init_steps", 0),  # Default if not specified
        device=device,
        dtype=torch.float32,  # Use float32 for stability in tests
    )


# Test Cases
# -----------


def test_cd_initialization(cd_loss, cd_loss_config):
    """Test initialization attributes."""
    assert cd_loss.k_steps == cd_loss_config["k_steps"]
    assert cd_loss.persistent == cd_loss_config["persistent"]
    assert cd_loss.device == cd_loss.sampler.device
    # assert cd_loss.dtype == torch.float32 # Let's keep this if we intend float32 tests
    assert cd_loss.dtype == cd_loss.sampler.dtype  # More general: should match sampler

    if cd_loss_config["persistent"]:
        # Test PCD specific init attributes
        assert cd_loss.buffer_size == cd_loss_config.get(
            "buffer_size", 10000
        )  # Check against actual default or passed value
        assert cd_loss.init_steps == cd_loss_config.get(
            "init_steps", 100
        )  # Check against actual default or passed value

        # Buffer tensor should be None initially, before first forward call
        assert cd_loss.replay_buffer is None
        assert not cd_loss.buffer_initialized
    else:  # persistent=False
        # Configuration attributes like buffer_size might still exist but are irrelevant.
        # Crucially, the buffer tensor itself should remain None and uninitialized.
        assert hasattr(cd_loss, "buffer_size")  # Attribute exists due to __init__ args
        assert hasattr(cd_loss, "init_steps")  # Attribute exists due to __init__ args

        # Check that buffer functionality is inactive
        assert cd_loss.replay_buffer is None
        assert not cd_loss.buffer_initialized

        # Optional: Verify they remain inactive even after a forward call
        batch_size = 5
        try:
            # Infer input_dim robustly
            input_dim = (
                cd_loss.energy_function.model[0].in_features
                if hasattr(cd_loss.energy_function, "model")
                else 2
            )
        except (AttributeError, IndexError):
            input_dim = 2  # Fallback for non-MLP energies
        x = torch.randn(
            batch_size, input_dim, device=cd_loss.device, dtype=cd_loss.dtype
        )
        cd_loss(x)  # Perform a forward pass
        assert cd_loss.replay_buffer is None  # Should still be None
        assert not cd_loss.buffer_initialized  # Should still be False


def test_cd_forward_pass_shapes_types(cd_loss, device):
    """Test output shapes and types from the forward pass."""
    batch_size = 32
    input_dim = (
        cd_loss.energy_function.model[0].in_features
        if isinstance(cd_loss.energy_function, MLPEnergy)
        else 2
    )  # Infer dim
    x = torch.randn(batch_size, input_dim, device=device, dtype=cd_loss.dtype)

    loss, neg_samples = cd_loss(x)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # Scalar loss
    assert loss.dtype == cd_loss.dtype
    assert loss.device.type == torch.device(device).type

    assert isinstance(neg_samples, torch.Tensor)
    assert neg_samples.shape == x.shape
    assert neg_samples.dtype == cd_loss.dtype
    assert neg_samples.device.type == torch.device(device).type


def test_compute_loss_calculation(cd_loss, device):
    """Test the static compute_loss method directly."""
    batch_size = 16
    input_dim = (
        cd_loss.energy_function.model[0].in_features
        if isinstance(cd_loss.energy_function, MLPEnergy)
        else 2
    )
    x_pos = torch.randn(batch_size, input_dim, device=device, dtype=cd_loss.dtype)
    x_neg = (
        torch.randn(batch_size, input_dim, device=device, dtype=cd_loss.dtype) * 2.0
    )  # Make neg samples different

    # Calculate expected loss manually
    with torch.no_grad():
        e_pos = cd_loss.energy_function(x_pos)
        e_neg = cd_loss.energy_function(x_neg)
        expected_loss = torch.mean(e_pos) - torch.mean(e_neg)

    # Calculate using the method
    computed_loss = cd_loss.compute_loss(x_pos, x_neg)

    assert torch.allclose(computed_loss, expected_loss, atol=1e-5)


def test_non_persistent_cd_start_points(energy_function, sampler, device):
    """Test that for non-persistent CD, k_steps=0 returns the input."""
    input_dim = (
        energy_function.model[0].in_features
        if isinstance(energy_function, MLPEnergy)
        else 2
    )

    # Sampler needs to handle k_steps=0 gracefully (identity) for this test
    # Modify mock sampler or skip if real sampler doesn't support k=0
    class MockSamplerK0(LangevinDynamics):
        @torch.no_grad()
        def sample(self, x, n_steps, **kwargs):
            if n_steps == 0:
                return x.clone()
            # Call parent for n_steps > 0
            return super().sample(x, n_steps, **kwargs)

    mock_sampler = MockSamplerK0(
        energy_function=energy_function, step_size=0.1, noise_scale=0.01, device=device
    )

    cd = ContrastiveDivergence(
        energy_function=energy_function,
        sampler=mock_sampler,
        k_steps=0,  # Test with zero steps
        persistent=False,
        device=device,
        dtype=torch.float32,
    )

    x = torch.randn(10, input_dim, device=device, dtype=cd.dtype)
    loss, neg_samples = cd(x)

    # With k_steps=0, neg_samples should be identical to input x
    assert torch.allclose(neg_samples, x, atol=1e-6)
    # Loss should be zero in this specific case
    assert torch.allclose(
        loss, torch.tensor(0.0, device=device, dtype=cd.dtype), atol=1e-6
    )


def test_pcd_buffer_initialization_on_first_call(cd_loss, device):
    """Test that the buffer is initialized only for PCD and on the first forward call."""
    if not cd_loss.persistent:
        pytest.skip("Test only relevant for persistent CD (PCD)")

    batch_size = 16
    input_dim = (
        cd_loss.energy_function.model[0].in_features
        if isinstance(cd_loss.energy_function, MLPEnergy)
        else 2
    )
    x = torch.randn(batch_size, input_dim, device=device, dtype=cd_loss.dtype)

    # Before first call
    assert cd_loss.replay_buffer is None
    assert not cd_loss.buffer_initialized

    # First call
    cd_loss(x)

    # After first call
    assert cd_loss.buffer_initialized
    assert isinstance(cd_loss.replay_buffer, torch.Tensor)
    expected_shape = (cd_loss.buffer_size,) + x.shape[1:]
    assert cd_loss.replay_buffer.shape == expected_shape
    assert cd_loss.replay_buffer.device.type == torch.device(device).type
    assert cd_loss.replay_buffer.dtype == cd_loss.dtype

    # Check content based on init_steps
    buffer_content_after_first_call = cd_loss.replay_buffer.clone()
    if cd_loss.init_steps > 0:
        # Should not be pure Gaussian noise if init_steps > 0
        # Check mean/std deviation are likely different from N(0,1)
        # This is a heuristic check
        mean = buffer_content_after_first_call.mean()
        std = buffer_content_after_first_call.std()
        assert not (
            -0.1 < mean < 0.1 and 0.9 < std < 1.1
        ), "Buffer seems like pure noise despite init_steps > 0"

    # Second call - buffer should persist and not be re-initialized (check object ID or content)
    buffer_id_before = id(cd_loss.replay_buffer)
    cd_loss(x)  # Second call
    assert id(cd_loss.replay_buffer) == buffer_id_before
    # Content should have changed due to update_buffer
    assert not torch.allclose(
        cd_loss.replay_buffer, buffer_content_after_first_call
    ), "Buffer content did not change after second PCD step"


def test_pcd_buffer_update(cd_loss, device):
    """Test that the buffer is updated with new samples in PCD."""
    if not cd_loss.persistent:
        pytest.skip("Test only relevant for persistent CD (PCD)")

    batch_size = cd_loss.buffer_size // 2  # Ensure batch size <= buffer size
    if batch_size == 0:
        batch_size = 1  # Handle tiny buffer size case
    input_dim = (
        cd_loss.energy_function.model[0].in_features
        if isinstance(cd_loss.energy_function, MLPEnergy)
        else 2
    )
    x = torch.randn(batch_size, input_dim, device=device, dtype=cd_loss.dtype)

    # First call to initialize and get initial buffer state
    _, first_neg_samples = cd_loss(x)
    buffer_state_1 = cd_loss.replay_buffer.clone()

    # Second call
    _, second_neg_samples = cd_loss(x)
    buffer_state_2 = cd_loss.replay_buffer.clone()

    # Buffer state should change
    assert not torch.allclose(buffer_state_1, buffer_state_2)

    # Check if *some* samples from the second batch ended up in the buffer.
    # This is hard to check definitively with random replacement.
    # We check if the buffers differ, and that the new buffer isn't identical to the second samples
    # (unless buffer_size == batch_size).
    if cd_loss.buffer_size > batch_size:
        # If buffer > batch, the buffer won't be *identical* to the last samples
        assert not torch.allclose(
            buffer_state_2,
            second_neg_samples.repeat(cd_loss.buffer_size // batch_size + 1, 1)[
                : cd_loss.buffer_size
            ],
        )
    elif cd_loss.buffer_size == batch_size:
        # If buffer == batch, the buffer should be identical to the last samples
        assert torch.allclose(buffer_state_2, second_neg_samples)


def test_pcd_start_points_origin(energy_function, sampler, device):
    """Test that for PCD, k_steps=0 returns samples from the buffer (not input x)."""
    input_dim = (
        energy_function.model[0].in_features
        if isinstance(energy_function, MLPEnergy)
        else 2
    )
    buffer_size = 50
    batch_size = 10

    # Use the Mock Sampler that handles k_steps=0
    class MockSamplerK0(LangevinDynamics):
        @torch.no_grad()
        def sample(self, x, n_steps, **kwargs):
            if n_steps == 0:
                return x.clone()
            return super().sample(x, n_steps, **kwargs)

    mock_sampler = MockSamplerK0(
        energy_function=energy_function, step_size=0.1, noise_scale=0.01, device=device
    )

    cd = ContrastiveDivergence(
        energy_function=energy_function,
        sampler=mock_sampler,
        k_steps=0,  # Test with zero steps
        persistent=True,
        buffer_size=buffer_size,
        init_steps=5,  # Initialize buffer with some steps
        device=device,
        dtype=torch.float32,
    )

    x = torch.randn(batch_size, input_dim, device=device, dtype=cd.dtype)

    # First call to initialize buffer
    cd(x)
    initial_buffer = cd.replay_buffer.clone()

    # Second call with k_steps=0
    loss, neg_samples = cd(x)

    # neg_samples should come from the buffer, NOT be identical to x
    assert not torch.allclose(neg_samples, x)

    # neg_samples should be samples that were present in the buffer before this call
    # Check if each neg_sample exists somewhere in the initial_buffer (can be slow for large buffers)
    found_in_buffer_count = 0
    for i in range(batch_size):
        # Check if neg_samples[i] is close to any row in initial_buffer
        is_present = torch.any(
            torch.all(
                torch.isclose(initial_buffer, neg_samples[i : i + 1], atol=1e-5), dim=1
            )
        )
        if is_present:
            found_in_buffer_count += 1
    assert (
        found_in_buffer_count == batch_size
    ), "Negative samples did not originate from the buffer"


def test_pcd_buffer_persistence_across_batch_sizes(cd_loss, device):
    """Test that the PCD buffer persists and is used across varying batch sizes."""
    if not cd_loss.persistent:
        pytest.skip("Test only relevant for persistent CD (PCD)")

    input_dim = (
        cd_loss.energy_function.model[0].in_features
        if isinstance(cd_loss.energy_function, MLPEnergy)
        else 2
    )
    buffer_size = cd_loss.buffer_size
    dtype = cd_loss.dtype

    # Ensure buffer is reasonably larger than batches for a good test
    if buffer_size < 40:
        pytest.skip("Buffer size too small for robust testing of this feature")

    # First call with batch size 10
    x1 = torch.randn(10, input_dim, device=device, dtype=dtype)
    cd_loss(x1)  # Initialize and update buffer
    assert cd_loss.buffer_initialized
    # Store buffer state AFTER the update from the first call
    buffer_after_b10 = cd_loss.replay_buffer.clone()
    buffer_id_after_b10 = id(cd_loss.replay_buffer)

    # Second call with different batch size (e.g., 20)
    x2 = torch.randn(20, input_dim, device=device, dtype=dtype)
    loss2, samples2 = cd_loss(x2)  # Second call, different batch size

    # --- Assertions ---
    # 1. Buffer should still be the same object and marked as initialized
    assert cd_loss.buffer_initialized
    assert id(cd_loss.replay_buffer) == buffer_id_after_b10

    # 2. Buffer's shape should remain unchanged (fixed size)
    assert cd_loss.replay_buffer.shape[0] == buffer_size
    assert (
        cd_loss.replay_buffer.shape[1:] == x1.shape[1:]
    )  # Data dimensions should match

    # 3. Buffer content should have been updated by the second call
    assert not torch.allclose(
        cd_loss.replay_buffer, buffer_after_b10
    ), "Buffer content did not change after second call with different batch size"

    # 4. Check that the output shapes of the second call are correct
    assert (
        samples2.shape[0] == x2.shape[0]
    )  # Output batch size matches input batch size
    assert samples2.shape[1:] == x2.shape[1:]
    assert loss2.shape == torch.Size([])


def test_pcd_buffer_size_warning(energy_function, sampler, device):
    """Test warning when batch_size > buffer_size during PCD start point sampling."""
    buffer_size = 10
    batch_size = 20  # batch > buffer

    cd = ContrastiveDivergence(
        energy_function=energy_function,
        sampler=sampler,
        k_steps=1,
        persistent=True,
        buffer_size=buffer_size,
        init_steps=1,  # Need some init
        device=device,
        dtype=torch.float32,
    )

    x = torch.randn(batch_size, 2, device=device, dtype=cd.dtype)

    # Expect a warning during get_start_points
    with pytest.warns(
        UserWarning,
        match=r"Buffer size \(\d+\) is smaller than batch size \(\d+\)\. Sampling with replacement.",
    ):
        cd(x)  # First call triggers initialization and potentially the warning

    # Second call definitely triggers the warning in get_start_points
    with pytest.warns(
        UserWarning,
        match=r"Buffer size \(\d+\) is smaller than batch size \(\d+\)\. Sampling with replacement.",
    ):
        loss, neg_samples = cd(x)

    assert neg_samples.shape[0] == batch_size  # Should still return correct batch size


def test_invalid_buffer_size_error(energy_function, sampler, device):
    """Test that initializing PCD with buffer_size <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="Replay buffer size must be positive"):
        cd = ContrastiveDivergence(
            energy_function=energy_function,
            sampler=sampler,
            k_steps=1,
            persistent=True,
            buffer_size=0,  # Invalid size
            init_steps=0,
            device=device,
            dtype=torch.float32,
        )
        # Initialization happens on first call
        x = torch.randn(10, 2, device=device, dtype=torch.float32)
        cd(x)

    with pytest.raises(ValueError, match="Replay buffer size must be positive"):
        cd = ContrastiveDivergence(
            energy_function=energy_function,
            sampler=sampler,
            k_steps=1,
            persistent=True,
            buffer_size=-10,  # Invalid size
            init_steps=0,
            device=device,
            dtype=torch.float32,
        )
        x = torch.randn(10, 2, device=device, dtype=torch.float32)
        cd(x)


@pytest.mark.parametrize("persistent", [False, True])
def test_cd_determinism_with_seeds(energy_function, sampler_config, device, persistent):
    """Test determinism for both CD and PCD with fixed seeds."""
    input_dim = (
        energy_function.model[0].in_features
        if isinstance(energy_function, MLPEnergy)
        else 2
    )
    k_steps = 3
    buffer_size = 30
    init_steps = 2
    batch_size = 5

    def run_cd(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        # Need to re-create energy function and sampler for full determinism check if they have internal state
        ef = energy_function.to(device)  # Re-create or ensure stateless fixture
        samp = LangevinDynamics(
            energy_function=ef,
            step_size=sampler_config["step_size"],
            noise_scale=sampler_config["noise_scale"],
            device=device,
            dtype=torch.float32,
        )

        cd_instance = ContrastiveDivergence(
            energy_function=ef,
            sampler=samp,
            k_steps=k_steps,
            persistent=persistent,
            buffer_size=buffer_size,
            init_steps=init_steps,
            device=device,
            dtype=torch.float32,
        )

        # Same input data for both runs
        torch.manual_seed(seed + 1)  # Seed for data generation
        x_data = torch.randn(batch_size, input_dim, device=device, dtype=torch.float32)

        # Seed right before the sampling call inside forward might also be needed
        # depending on where RNG is used. Let's assume seeding before cd() is enough.
        torch.manual_seed(seed + 2)  # Seed before execution
        loss_val, samples_val = cd_instance(x_data)

        buffer_state = (
            cd_instance.replay_buffer.clone()
            if persistent and cd_instance.buffer_initialized
            else None
        )
        return loss_val, samples_val, buffer_state

    seed = 42
    loss1, samples1, buffer1 = run_cd(seed)
    loss2, samples2, buffer2 = run_cd(seed)  # Rerun with same seed

    assert torch.allclose(loss1, loss2, atol=1e-6)
    assert torch.allclose(samples1, samples2, atol=1e-6)
    if persistent:
        assert buffer1 is not None and buffer2 is not None
        assert torch.allclose(buffer1, buffer2, atol=1e-6)


@pytest.mark.parametrize("persistent", [False, True])
def test_cd_gradient_flow(device, persistent):
    """Test gradient flow through the loss for trainable energy functions."""
    # Use a trainable energy function
    energy_fn = MLPEnergy(input_dim=2, hidden_dim=8).to(device).to(torch.float32)
    # Ensure parameters require grad
    for p in energy_fn.parameters():
        p.requires_grad_(True)

    sampler = LangevinDynamics(
        energy_function=energy_fn,
        step_size=0.1,
        noise_scale=0.01,
        device=device,
        dtype=torch.float32,
    )

    cd = ContrastiveDivergence(
        energy_function=energy_fn,
        sampler=sampler,
        k_steps=3,
        persistent=persistent,
        buffer_size=50,  # Need buffer for PCD test
        init_steps=2,
        device=device,
        dtype=torch.float32,
    )

    x = torch.randn(10, 2, device=device, dtype=torch.float32)

    # Check gradients before backward
    for param in energy_fn.parameters():
        if param.grad is not None:
            param.grad = None  # Zero out just in case

    # Compute loss and backward
    loss, _ = cd(x)
    loss.backward()

    # Verify gradients exist and are non-zero for *some* parameters
    found_non_zero_grad = False
    for name, param in energy_fn.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        if torch.any(param.grad != 0):
            found_non_zero_grad = True
        # print(f"{name} grad norm: {param.grad.norm().item()}") # Optional debug print

    assert found_non_zero_grad, "No non-zero gradients found for any parameter."


def test_empty_input_batch(cd_loss, device):
    """Test behavior with an empty input batch."""
    input_dim = (
        cd_loss.energy_function.model[0].in_features
        if isinstance(cd_loss.energy_function, MLPEnergy)
        else 2
    )
    x_empty = torch.empty((0, input_dim), device=device, dtype=cd_loss.dtype)

    # Option 1: Expect an error (e.g., due to mean reduction)
    # with pytest.raises(RuntimeError): # Or IndexError, depending on failure point
    #    cd_loss(x_empty)

    # Option 2: Expect specific output (e.g., loss=0 or NaN, empty samples) - Requires careful implementation
    loss, neg_samples = cd_loss(x_empty)
    assert neg_samples.shape == x_empty.shape  # Output samples should also be empty
    # Loss might be 0 or NaN depending on how torch.mean handles empty tensors.
    # torch.mean of empty tensor results in NaN.
    assert torch.isnan(loss) or loss == 0.0  # Allow 0 if mean is robustly handled
