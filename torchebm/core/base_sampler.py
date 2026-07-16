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
    ):
        super().__init__(device=device, dtype=dtype)
        self.model = model

    def _init_state(
        self,
        x: Optional[torch.Tensor],
        dim: Optional[Union[int, Tuple[int, ...]]],
        n_samples: int,
    ) -> torch.Tensor:
        r"""Coerce `x` to the sampler's device/dtype, or draw from `N(0, I)`.

        Args:
            x: Initial state, or `None` to synthesize one.
            dim: State dimension (int) or shape (tuple), used when `x is None`.
            n_samples: Number of parallel chains, used when `x is None`.

        Returns:
            State tensor of shape `[n_samples, *shape]`.

        Raises:
            ValueError: If both `x` and `dim` are `None`.
        """
        if x is not None:
            return x.to(device=self.device, dtype=self.dtype)
        if dim is None:
            raise ValueError("dim must be provided when x is None")
        shape = (dim,) if isinstance(dim, int) else tuple(dim)
        return torch.randn(n_samples, *shape, dtype=self.dtype, device=self.device)

    def _model_gradient(
        self, x: torch.Tensor, model_kwargs: Dict[str, object]
    ) -> torch.Tensor:
        r"""Route a gradient call through the conditioning convention.

        This is the single back-compat chokepoint: when `model_kwargs` is empty
        the model is called exactly as before (``model.gradient(x)``), so
        analytic `gradient(self, x)` overrides with no `model_kwargs` parameter
        keep working. Pass an already-normalized dict (see
        `_prepare_model_kwargs`); it is reused per step with no re-normalization.
        """
        if model_kwargs:
            return self.model.gradient(x, model_kwargs=model_kwargs)
        return self.model.gradient(x)

    def _model_energy(
        self, x: torch.Tensor, model_kwargs: Dict[str, object]
    ) -> torch.Tensor:
        r"""Route an energy call through the conditioning convention.

        Mirrors `_model_gradient`: unconditional call when `model_kwargs` is
        empty, so unconditional scalar models (``forward(self, x)``) are
        untouched.
        """
        if model_kwargs:
            return self.model(x, **model_kwargs)
        return self.model(x)

    @abstractmethod
    def sample(
        self,
        x: Optional[torch.Tensor] = None,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        n_steps: int = 100,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
        reset_schedulers: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        r"""
        Run the sampling process.

        Args:
            x: Initial state. If `None`, samples from `N(0, I)`.
            dim: Dimension (int) or shape (tuple) of the state space, used when
                `x is None`. Samplers that can infer it from the model (the HMC
                family, via `model.mean`) accept `None`; otherwise a
                `ValueError` is raised.
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
            ValueError: If `thin < 1`, or if `x` is `None` and `dim` is `None`
                for samplers that cannot infer the state shape.
        """
        raise NotImplementedError
