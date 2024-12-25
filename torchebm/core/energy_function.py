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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
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

