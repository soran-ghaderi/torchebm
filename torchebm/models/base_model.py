from abc import ABC, abstractmethod
import torch
from torchebm.core.energy_function import EnergyFunction
from torchebm.core.basesampler import BaseSampler


class BaseModel(ABC):
    def __init__(self, energy_function: EnergyFunction, sampler: BaseSampler):
        self.energy_function = energy_function
        self.sampler = sampler

    @abstractmethod
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        pass

    @abstractmethod
    def train_step(self, real_data: torch.Tensor) -> dict:
        pass
