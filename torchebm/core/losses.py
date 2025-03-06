import torch
from torch import nn
import math
from abc import abstractmethod


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass

    def to(self, device):
        self.device = device
        return self
