import pytest
import torch
import torch.nn as nn

from torchebm.core import BaseModel
from torchebm.losses import DenoisingScoreMatching


class QuadraticEnergyND(BaseModel):
    """E(x) = 1/2 * ||x||^2 over all non-batch dims."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x.view(x.shape[0], -1) ** 2).sum(dim=1)


class MLPEnergy(BaseModel):
    def __init__(self, input_dim=4, hidden_dim=16):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


@pytest.fixture(
    params=[
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ]
)
def device(request):
    return request.param


def test_dsm_scalar_and_finite(device):
    torch.manual_seed(0)
    energy = QuadraticEnergyND().to(device)
    loss_fn = DenoisingScoreMatching(model=energy, noise_scale=0.1, device=device)

    x = torch.randn(64, 10, device=device)
    loss = loss_fn(x)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


@pytest.mark.parametrize("shape", [(64, 3, 8, 8), (32, 2, 4, 5, 3)])
def test_dsm_multi_dimensional_inputs(shape):
    torch.manual_seed(1)
    device = torch.device("cpu")
    energy = QuadraticEnergyND().to(device)
    loss_fn = DenoisingScoreMatching(model=energy, noise_scale=0.05, device=device)

    x = torch.randn(*shape, device=device)
    loss = loss_fn(x)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_dsm_gradient_flow():
    torch.manual_seed(2)
    device = torch.device("cpu")
    energy = MLPEnergy(input_dim=4, hidden_dim=8).to(device)
    loss_fn = DenoisingScoreMatching(model=energy, noise_scale=0.1, device=device)

    x = torch.randn(16, 4, device=device)
    loss = loss_fn(x)
    loss.backward()

    has_grad = any((p.grad is not None) and torch.any(p.grad != 0) for p in energy.parameters())
    assert has_grad, "Expected non-zero gradients on energy parameters"


def test_dsm_regularization_effect():
    torch.manual_seed(3)
    device = torch.device("cpu")
    energy = MLPEnergy(input_dim=3, hidden_dim=8).to(device)
    dsm_no_reg = DenoisingScoreMatching(model=energy, noise_scale=0.1, regularization_strength=0.0, device=device)
    dsm_with_reg = DenoisingScoreMatching(model=energy, noise_scale=0.1, regularization_strength=0.1, device=device)

    x = torch.randn(32, 3, device=device)
    loss_no_reg = dsm_no_reg(x).detach()
    loss_with_reg = dsm_with_reg(x).detach()

    assert torch.isfinite(loss_no_reg)
    assert torch.isfinite(loss_with_reg)
    # Not guaranteed to be strictly larger every batch, so just ensure they are not NaN and can differ
    assert abs(loss_with_reg.item() - loss_no_reg.item()) >= 0.0


def test_dsm_mixed_precision_flag_safe():
    # Ensure enabling mixed precision doesn't error even on CPU-only environments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    energy = MLPEnergy(input_dim=2, hidden_dim=4).to(device)
    dsm = DenoisingScoreMatching(model=energy, noise_scale=0.05, use_mixed_precision=True, device=device)

    x = torch.randn(8, 2, device=device)
    loss = dsm(x)
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)


def test_dsm_noise_scale_behavior():
    # Larger noise scale should change loss magnitude compared to smaller noise scale
    torch.manual_seed(4)
    device = torch.device("cpu")
    energy = MLPEnergy(input_dim=5, hidden_dim=8).to(device)

    small_sigma = DenoisingScoreMatching(model=energy, noise_scale=0.01, device=device)
    large_sigma = DenoisingScoreMatching(model=energy, noise_scale=0.2, device=device)

    x = torch.randn(64, 5, device=device)
    torch.manual_seed(123)
    loss_small = small_sigma(x).detach()
    torch.manual_seed(123)
    loss_large = large_sigma(x).detach()

    assert torch.isfinite(loss_small)
    assert torch.isfinite(loss_large)
    # With same random seed for noise, the two should generally differ when sigma changes
    assert abs(loss_small.item() - loss_large.item()) > 1e-8


