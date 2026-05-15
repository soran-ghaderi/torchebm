from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from torchebm.core import Schedulable, TorchEBMModule


class BaseSampler(Schedulable, TorchEBMModule, ABC):
    r"""
    Abstract base class for samplers.

    Args:
        model (nn.Module): The model to sample from. For MCMC samplers, this is
            typically a `BaseModel` energy function; for learned samplers it may be
            any `nn.Module`.
        dtype (torch.dtype): The data type for computations.
        device (Optional[Union[str, torch.device]]): The device for computations.

    Sampling output contract:
        `sample(return_diagnostics=False)` → `Tensor`.
        `sample(return_diagnostics=True)` → `(Tensor, Dict[str, Tensor])` where
        the dict's keys are sampler-specific metric names and values have shape
        `[n_kept, ...]`. Standard keys (when produced):

        - ``"mean"`` (`[n_kept, *data_shape]`): batch-mean of `x` at each kept step.
        - ``"var"`` (`[n_kept, *data_shape]`): batch-variance of `x`.
        - ``"energy"`` (`[n_kept]`): batch-mean energy at each kept step.
        - ``"acceptance_rate"`` (`[n_kept]`): MH-acceptance fraction (HMC).

        When ``thin > 1``, ``n_kept = n_steps // thin``. Otherwise ``n_kept = n_steps``.

        Each sampler's `sample()` docstring lists the keys it produces.
    """

    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)
        self.model = model

    @abstractmethod
    def sample(
        self,
        x: Optional[torch.Tensor] = None,
        dim: int = 10,
        n_steps: int = 100,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
        reset_schedulers: bool = True,
        *args,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        r"""
        Run the sampling process.

        Args:
            x: Initial state. If `None`, samples from `N(0, I)`.
            dim: Dimension of the state space (used when `x is None`).
            n_steps: Number of MCMC steps.
            n_samples: Number of parallel chains/samples.
            thin: Keep every `thin`-th sample. Final stored length is
                `n_steps // thin`. Must be `>= 1`.
            return_trajectory: If True, return the full kept trajectory of shape
                `[n_samples, n_steps // thin, *data_shape]` instead of the final
                state.
            return_diagnostics: If True, also return a `Dict[str, torch.Tensor]`
                of per-step metrics. See class docstring for the key contract.
            reset_schedulers: If True (default), reset registered schedulers
                before sampling so each call starts from step 0. Pass False for
                lifetime schedules driven by an outer training loop.

        Returns:
            Either a tensor of samples (or trajectory) of shape
            `[n_samples, *data_shape]` or `[n_samples, n_steps // thin, *data_shape]`,
            optionally paired with a diagnostics dict if `return_diagnostics=True`.

        Raises:
            ValueError: If `thin < 1`.
        """
        raise NotImplementedError
