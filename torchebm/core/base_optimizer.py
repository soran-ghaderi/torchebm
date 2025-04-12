from abc import abstractmethod

import torch
import math
from torch import nn


class Optimizer(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def zero_grad(self):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

    @abstractmethod
    def to(self, device):
        pass
