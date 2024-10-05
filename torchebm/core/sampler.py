from abc import ABC, abstractmethod
import torch

class Sampler(ABC):
    @abstractmethod
    def sample(self, energy_function: EnergyFunction, initial_state: torch.Tensor, num_steps: int) -> torch.Tensor:
        pass

    def cuda_sample(self, energy_function: EnergyFunction, initial_state: torch.Tensor, num_steps: int) -> torch.Tensor:
        return self.sample(energy_function, initial_state, num_steps)  # Default to PyTorch implementation
