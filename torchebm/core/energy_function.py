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
