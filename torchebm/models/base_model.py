from abc import ABC, abstractmethod
import torch
from torchebm.core.base_energy_function import BaseEnergyFunction
from torchebm.core.base_sampler import BaseSampler


class BaseModel(ABC):
    """
    Base class for models.

    Args:
        energy_function (BaseEnergyFunction): Energy function to sample from.
        sampler (BaseSampler): Sampler to use for sampling.

    Methods:
        energy(x): Compute the energy of the input.
        sample(num_samples): Sample from the model.
        train_step(real_data): Perform a single training step
    """

    def __init__(self, energy_function: BaseEnergyFunction, sampler: BaseSampler):
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
