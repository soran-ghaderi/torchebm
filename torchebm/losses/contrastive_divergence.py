from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch
from torch import nn
import math
from abc import abstractmethod

from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle

from torchebm.core import Sampler
from torchebm.core.energy_function import EnergyFunction
from torchebm.core.losses import Loss


class ContrastiveDivergence(Loss):
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
