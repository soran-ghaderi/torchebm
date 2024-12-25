import math
from abc import ABC, abstractmethod
import torch
from torch import nn

class EnergyFunction(nn.Module, ABC):
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
        return self.barrier_height * (x.pow(2) - 1).pow(2)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return 4 * self.barrier_height * x * (x.pow(2) - 1)

    def to(self, device):
        self.device = device
        return self

class GaussianEnergy(EnergyFunction):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = mean.to(self.device)
        self.precision = torch.inverse(cov).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        delta = x - self.mean
        return 0.5 * torch.einsum('...i,...ij,...j->...', delta, self.precision, delta)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return torch.einsum('...ij,...j->...i', self.precision, x - self.mean)

    def to(self, device):
        self.device = device
        self.mean = self.mean.to(device)
        self.precision = self.precision.to(device)
        return self

class HarmonicEnergy(EnergyFunction):
    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.k * x.pow(2)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return self.k * x

    def to(self, device):
        self.device = device
        return self


class RosenbrockEnergy(EnergyFunction):
    def __init__(self, a: float = 1.0, b: float = 100.0):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.a - x[..., 0])**2 + self.b * (x[..., 1] - x[..., 0]**2)**2

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        grad = torch.zeros_like(x)
        grad[..., 0] = -2 * (self.a - x[..., 0]) - 4 * self.b * x[..., 0] * (x[..., 1] - x[..., 0]**2)
        grad[..., 1] = 2 * self.b * (x[..., 1] - x[..., 0]**2)
        return grad

    def to(self, device):
        self.device = device
        return self

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
        term1 = self.a * self.b * torch.exp(-self.b * torch.sqrt(sum1 / n)) * (x / torch.sqrt(sum1 / n))
        term2 = torch.exp(sum2 / n) * torch.sin(self.c * x) * self.c / n
        return term1 + term2

    def to(self, device):
        self.device = device
        return self