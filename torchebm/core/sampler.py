from abc import ABC, abstractmethod

import torch
from torchebm.core.energy_function import EnergyFunction


class Sampler(ABC):
    def __init__(self, energy_function: EnergyFunction):
        self.energy_function = energy_function

    @abstractmethod
    def sample(self, initial_state: torch.Tensor, n_steps: int) -> torch.Tensor:
        pass


