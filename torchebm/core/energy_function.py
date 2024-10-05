from abc import ABC, abstractmethod
import torch

class EnergyFunction(ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def cuda_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)  # Default to PyTorch implementation

    def cuda_gradient(self, x: torch.Tensor) -> torch.Tensor:
        return self.gradient(x)