import pytest
import torch
from torchebm.core import BaseEnergyFunction


class SimpleQuadraticEnergy(BaseEnergyFunction):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x**2, dim=-1)

    # def gradient(self, x: torch.Tensor) -> torch.Tensor:
    #     return 2 * x


def test_energy_function():
    # Initialize the energy function
    energy_fn = SimpleQuadraticEnergy()

    # Create a sample input tensor
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # Test forward method
    energy = energy_fn.forward(x)
    expected_energy = torch.tensor([5.0, 25.0])
    assert torch.allclose(
        energy, expected_energy
    ), f"Expected energy {expected_energy}, but got {energy}"

    # Test gradient method
    grad = energy_fn.gradient(x)
    expected_grad = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    assert torch.allclose(
        grad, expected_grad
    ), f"Expected gradient {expected_grad}, but got {grad}"

    # Test cuda_forward method (should default to forward if not implemented)
    # cuda_energy = energy_fn.cuda_forward(x)
    # assert torch.allclose(cuda_energy, energy), f"cuda_forward should match forward, but got {cuda_energy} vs {energy}"

    # Test cuda_gradient method (should default to gradient if not implemented)
    # cuda_grad = energy_fn.cuda_gradient(x)
    # assert torch.allclose(cuda_grad, grad), f"cuda_gradient should match gradient, but got {cuda_grad} vs {grad}"


if __name__ == "__main__":
    pytest.main([__file__])
