from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch

from torchebm.core import DeviceMixin, BaseModel


class Integrator(DeviceMixin, ABC):
    """
    Abstract integrator that advances a sampler state according to dynamics.

    The integrator operates on a generic state dict to remain reusable across
    samplers (e.g., Langevin uses only position `x`, HMC uses position `x` and
    momentum `p`).

    Methods follow PyTorch conventions and respect `device`/`dtype` from
    `DeviceMixin`.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        *args,
        **kwargs,
    ):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)

    @abstractmethod
    def step(
        self,
        state: Dict[str, torch.Tensor],
        model: BaseModel,
        step_size: torch.Tensor,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Advance the dynamical state by one integrator application.

        Args:
            state: Mapping containing required tensors (e.g., {'x': ..., 'p': ...}).
            model: Energy-based model providing `forward` and `gradient`.
            step_size: Step size for the integration.
            *args: Additional positional arguments specific to the integrator.
            **kwargs: Additional keyword arguments specific to the integrator.

        Returns:
            Updated state dict with the same keys as the input `state`.
        """
        raise NotImplementedError
