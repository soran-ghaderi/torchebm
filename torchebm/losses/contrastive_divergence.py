from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch
from torch import nn
import math
from abc import abstractmethod

from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle

from torchebm.core import BaseSampler
from torchebm.core.energy_function import EnergyFunction
from torchebm.core.losses import Loss


class ContrastiveDivergenceBase(Loss):
    def __init__(self, k=1):
        super().__init__()
        self.k = k  # Number of sampling steps

    @abstractmethod
    def sample(self, energy_model, x_pos):
        """Abstract method: Generate negative samples from the energy model.
        Args:
            energy_model: Energy-based model (e.g., RBM)
            x_pos: Positive samples (data)
        Returns:
            x_neg: Negative samples (model samples)
        """
        raise NotImplementedError

    def forward(self, energy_model, x_pos):
        """Compute the CD loss: E(x_pos) - E(x_neg)"""
        x_neg = self.sample(energy_model, x_pos)
        loss = energy_model(x_pos).mean() - energy_model(x_neg).mean()
        return loss


class ContrastiveDivergence(ContrastiveDivergenceBase):
    def __init__(self, k=1):
        super().__init__(k)

    def sample(self, energy_model, x_pos):
        x_neg = x_pos.clone().detach()
        for _ in range(self.k):
            x_neg = energy_model.gibbs_step(
                x_neg
            )  # todo: implement `gibbs_step` in energy_model
        return x_neg


class PersistentContrastiveDivergence(ContrastiveDivergenceBase):
    def __init__(self, buffer_size=100):
        super().__init__(k=1)
        self.buffer = None  # Persistent chain state
        self.buffer_size = buffer_size

    # def sample(self, energy_model, x_pos):
    #     if self.buffer is None or len(self.buffer) != self.buffer_size:
    #         # Initialize buffer with random noise
    #         self.buffer = torch.randn(self.buffer_size, *x_pos.shape[1:],
    #                                   device=x_pos.device)
    #
    #     # Update buffer with Gibbs steps
    #     for _ in range(self.k):
    #         self.buffer = energy_model.gibbs_step(self.buffer)
    #
    #     # Return a subset of the buffer as negative samples
    #     idx = torch.randint(0, self.buffer_size, (x_pos.shape[0],))
    #     return self.buffer[idx]


class ParallelTemperingCD(ContrastiveDivergenceBase):
    def __init__(self, temps=[1.0, 0.5], k=5):
        super().__init__(k)
        self.temps = temps  # List of temperatures

    # def sample(self, energy_model, x_pos):
    #     chains = [x_pos.detach().clone() for _ in self.temps]
    #     for _ in range(self.k):
    #         # Run Gibbs steps at each temperature
    #         for i, temp in enumerate(self.temps):
    #             chains[i] = energy_model.gibbs_step(chains[i], temp=temp)
    #
    #         # Swap states between chains (temperature exchange)
    #         swap_idx = torch.randint(0, len(self.temps) - 1, (1,))
    #         chains[swap_idx], chains[swap_idx + 1] = chains[swap_idx + 1], chains[swap_idx]
    #
    #     return chains[0]  # Return samples from the highest-temperature chain
