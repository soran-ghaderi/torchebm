
from typing import Optional, Union, Tuple, List
from functools import partial

import torch

from torchebm.core.base_model import BaseModel, GaussianEnergy
from torchebm.core.base_sampler import BaseSampler
from torchebm.core import BaseScheduler, ConstantScheduler, ExponentialDecayScheduler


class NestrovAcceleratedGradient(BaseSampler):

    def __init__(
        self,
        model: BaseModel,
        step_size: Union[float, BaseScheduler] = 1e-3,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(model=model, dtype=dtype, device=device)
        print("hi")
        

