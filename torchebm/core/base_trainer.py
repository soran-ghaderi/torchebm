import torch
from .base_energy_function import BaseEnergyFunction
from .base_sampler import BaseSampler


class ContrastiveDivergenceTrainer:
    def __init__(
        self,
        energy_function: BaseEnergyFunction,
        sampler: BaseSampler,
        learning_rate: float = 0.01,
    ):
        self.energy_function = energy_function
        self.sampler = sampler
        self.optimizer = torch.optim.Adam(
            self.energy_function.parameters(), lr=learning_rate
        )

    def train_step(self, real_data: torch.Tensor) -> dict:
        self.optimizer.zero_grad()

        # Positive phase
        positive_energy = self.energy_function(real_data)

        # Negative phase
        initial_samples = torch.randn_like(real_data)
        negative_samples = self.sampler.sample(
            self.energy_function, initial_samples, num_steps=10
        )
        negative_energy = self.energy_function(negative_samples)

        # Compute loss
        loss = positive_energy.mean() - negative_energy.mean()

        # Backpropagation
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "positive_energy": positive_energy.mean().item(),
            "negative_energy": negative_energy.mean().item(),
        }
