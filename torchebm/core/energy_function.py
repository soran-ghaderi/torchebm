import math
from abc import ABC, abstractmethod
import torch
from torch import nn


class EnergyFunction(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        x.requires_grad_(True)
        energy = self.forward(x)
        return torch.autograd.grad(energy.sum(), x, create_graph=True)[0]

    def __call__(self, x: torch.Tensor, *args, **kwargs):
        return self.forward(x)

    def to(self, device):
        self.device = device
        return self


class DoubleWellEnergy(EnergyFunction):
    def __init__(self, barrier_height: float = 2.0):
        super().__init__()
        self.barrier_height = barrier_height

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.barrier_height * (x.pow(2) - 1).pow(2).sum(dim=-1)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return 4 * self.barrier_height * x * (x.pow(2) - 1)


class GaussianEnergy(EnergyFunction):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = mean.to(self.device)
        self.cov = cov.to(device)
        self.cov_inv = torch.inverse(cov).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assuming the cov is already positive definite and symmetric
        x = x.to(self.device)
        delta = x - self.mean
        # cov_inv = torch.inverse(
        #     self.cov
        # )  # it's dynamic here, but see if it's fine to compute it only in the init
        return 0.5 * torch.einsum("...i,...ij,...j->...", delta, self.cov_inv, delta)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return torch.einsum("...ij,...j->...i", self.cov_inv, x - self.mean)

    def to(self, device):
        self.device = device
        self.mean = self.mean.to(device)
        self.cov_inv = self.cov_inv.to(device)
        return self


class HarmonicEnergy(EnergyFunction):
    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.k * x.pow(2).sum(dim=-1)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return self.k * x


class RosenbrockEnergy(EnergyFunction):
    def __init__(self, a: float = 1.0, b: float = 100.0):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.a - x[..., 0]) ** 2 + self.b * (x[..., 1] - x[..., 0] ** 2) ** 2

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        grad = torch.zeros_like(x)
        grad[..., 0] = -2 * (self.a - x[..., 0]) - 4 * self.b * x[..., 0] * (
            x[..., 1] - x[..., 0] ** 2
        )
        grad[..., 1] = 2 * self.b * (x[..., 1] - x[..., 0] ** 2)
        return grad


class AckleyEnergy(EnergyFunction):
    def __init__(self, a: float = 20.0, b: float = 0.2, c: float = 2 * math.pi):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        sum1 = torch.sum(x**2, dim=-1)
        sum2 = torch.sum(torch.cos(self.c * x), dim=-1)
        term1 = -self.a * torch.exp(-self.b * torch.sqrt(sum1 / n))
        term2 = -torch.exp(sum2 / n)
        return term1 + term2 + self.a + math.e

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        sum1 = torch.sum(x**2, dim=-1, keepdim=True)
        sum2 = torch.sum(torch.cos(self.c * x), dim=-1, keepdim=True)
        term1 = (
            self.a
            * self.b
            * torch.exp(-self.b * torch.sqrt(sum1 / n))
            * (x / torch.sqrt(sum1 / n))
        )
        term2 = torch.exp(sum2 / n) * torch.sin(self.c * x) * self.c / n
        return term1 + term2


class RastriginEnergy(EnergyFunction):
    def __init__(self, a: float = 10.0):
        super().__init__()
        self.a = a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        return self.a * n + torch.sum(
            x**2 - self.a * torch.cos(2 * math.pi * x), dim=-1
        )

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * x + 2 * math.pi * self.a * torch.sin(2 * math.pi * x)
