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