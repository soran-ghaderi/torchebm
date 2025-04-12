from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch
from torch import nn
import math
from abc import abstractmethod

from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle

from torchebm.core import BaseSampler
from torchebm.core.base_energy_function import BaseEnergyFunction
from torchebm.core.base_loss import BaseContrastiveDivergence


class ContrastiveDivergence(BaseContrastiveDivergence):

    def forward(
        self, x: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = x.shape[0]

        if self.persistent and self.chain is not None:
            init_samples = self.chain
        else:
            init_samples = torch.randn_like(x)

        # generate negative samples
        pred_samples = self.sampler.sample(
            x=x,
            n_steps=self.n_steps,
            n_samples=batch_size,
        )

        loss = self.compute_loss(x, pred_samples, *args, **kwargs)

        return loss, pred_samples

    def compute_loss(
        self, x: torch.Tensor, pred_x: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:

        x_energy = self.energy_function(x)
        pred_x_energy = self.energy_function(pred_x)

        # Contrastive Divergence loss: E[data] - E[model]
        loss = torch.mean(pred_x_energy - x_energy)

        return loss


class PersistentContrastiveDivergence(BaseContrastiveDivergence):
    def __init__(self, buffer_size=100):
        super().__init__(n_steps=1)
        self.buffer = None  # Persistent chain state
        self.buffer_size = buffer_size

    # def sample(self, energy_model, x_pos):
    #     if self.buffer is None or len(self.buffer) != self.buffer_size:
    #         # Initialize buffer with random noise
    #         self.buffer = torch.randn(self.buffer_size, *x_pos.shape[1:],
    #                                   device=x_pos.device)
    #
    #     # Update buffer with Gibbs steps
    #     for _ in range(self.n_steps):
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
    #     for _ in range(self.n_steps):
    #         # Run Gibbs steps at each temperature
    #         for i, temp in enumerate(self.temps):
    #             chains[i] = energy_model.gibbs_step(chains[i], temp=temp)
    #
    #         # Swap states between chains (temperature exchange)
    #         swap_idx = torch.randint(0, len(self.temps) - 1, (1,))
    #         chains[swap_idx], chains[swap_idx + 1] = chains[swap_idx + 1], chains[swap_idx]
    #
    #     return chains[0]  # Return samples from the highest-temperature chain
