import pytest
import torch
import torch.nn as nn

from torchebm.core import BaseModel
from torchebm.losses.score_matching import ScoreMatching


class QuadraticEnergyND(BaseModel):
    """E(x) = 1/2 * ||x||^2 over all non-batch dims.

    For standard normal data, exact score matching objective has expectation -d/2
    where d is the flattened feature dimension.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x.view(x.shape[0], -1) ** 2).sum(dim=1)


class LinearEnergy(BaseModel):
    """E(x) = w^T x (flattened). Hessian is zero, gradient is constant w."""

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.shape[0], -1)
        return (x_flat @ self.w).view(-1)


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_exact_sm_scalar_and_finite(device):
    torch.manual_seed(0)
    energy = QuadraticEnergyND().to(device)
    loss_fn = ScoreMatching(model=energy, hessian_method="exact", device=device)

    x = torch.randn(64, 10, device=device)
    loss = loss_fn(x)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "batch, dim, tol",
    [
        (4096, 10, 0.2),  # large batch, moderate dim
        (64, 192, 3.0),  # smaller batch, large dim (multi-d tolerance)
    ],
)
def test_exact_sm_matches_gaussian_constant_cpu(batch, dim, tol):
    torch.manual_seed(1)
    device = torch.device("cpu")
    energy = QuadraticEnergyND().to(device)
    loss_fn = ScoreMatching(model=energy, hessian_method="exact", device=device)

    x = torch.randn(batch, dim, device=device)
    expected = -dim / 2.0
    loss = loss_fn(x)
    assert abs(loss.item() - expected) < tol


@pytest.mark.parametrize(
    "shape",
    [
        (64, 3, 8, 8),  # image-like data
        (32, 2, 4, 5, 3),  # higher-order tensor
    ],
)
def test_exact_sm_multi_dimensional_inputs(shape):
    torch.manual_seed(2)
    device = torch.device("cpu")
    energy = QuadraticEnergyND().to(device)
    loss_fn = ScoreMatching(model=energy, hessian_method="exact", device=device)

    x = torch.randn(*shape, device=device)
    d = int(torch.tensor(shape[1:]).prod().item())
    expected = -d / 2.0

    loss = loss_fn(x)
    # Use a relaxed tolerance due to smaller batch sizes
    assert abs(loss.item() - expected) < max(3.0, 0.05 * d)


def test_exact_sm_batch_consistency():
    torch.manual_seed(3)
    device = torch.device("cpu")
    energy = QuadraticEnergyND().to(device)
    loss_fn = ScoreMatching(model=energy, hessian_method="exact", device=device)

    x = torch.randn(200, 16, device=device)
    full = loss_fn(x).item()

    a = loss_fn(x[:80]).item()
    b = loss_fn(x[80:]).item()
    weighted = (a * 80 + b * 120) / 200
    assert abs(full - weighted) < 1e-5


def test_exact_sm_gradient_flow_linear_energy():
    torch.manual_seed(4)
    device = torch.device("cpu")
    dim = 20
    energy = LinearEnergy(dim=dim).to(device)
    loss_fn = ScoreMatching(model=energy, hessian_method="exact", device=device)

    x = torch.randn(64, dim, device=device)
    loss = loss_fn(x)
    loss.backward()

    assert energy.w.grad is not None
    assert torch.any(energy.w.grad != 0)


