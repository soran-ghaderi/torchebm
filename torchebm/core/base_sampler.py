from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from torchebm.core import Schedulable, TorchEBMModule
from torchebm.models.functional import energy_gradient


def _score_from_energy(
    call: Callable[..., torch.Tensor], model: nn.Module
) -> Callable[..., torch.Tensor]:
    r"""Closure differentiating an energy call into the score, the
    ``prediction_type="score"`` path of `BaseModel.__call__`."""
    force_fp32 = bool(getattr(model, "force_fp32_gradient", False))

    def score(x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            energy = call(x.float() if force_fp32 else x, t)
            return -energy_gradient(energy, x).detach()

    return score


class BaseSampler(Schedulable, TorchEBMModule, ABC):
    r"""
    Abstract base class for samplers.

    The model is an argument of the call, as the function is for
    `torchdiffeq.odeint`:

    ```python
    sampler = LangevinDynamics(step_size=0.1, noise_scale=1.0)
    x = sampler(model, x0, n_steps=100)
    ```

    `forward` runs the sampler's updates with the model called as
    ``model(x)`` (`timestep_conditioned=False`) or ``model(x, t)``, matching
    `model.prediction_type` to what the sampler consumes (`prediction_type`):
    equal types pass through, a tuple such as ``("energy", "score")`` is
    indexed, and an energy is differentiated into the score. A model without
    the attribute, or declaring ``None``, is taken to output what the sampler
    consumes. Step sizes, noise scales and temperatures are `BaseScheduler`
    parameters set at construction, reset at every call and advanced once per
    step together with those of a `Schedulable` model such as
    `InteractionModel`; `FlowSampler` also accepts a custom time grid per call.

    Args:
        model (nn.Module): Deprecated. Model sampled by `sample()`; pass the
            model to the call instead.
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

    #: Model output the sampler consumes: ``"score"``, ``"velocity"`` or ``"energy"``.
    prediction_type: str = "score"
    #: Whether the model is called as ``model(x, t)`` rather than ``model(x)``.
    timestep_conditioned: bool = False

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.model = model

    @abstractmethod
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        *,
        n_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        r"""Run `n_steps` updates of `model` from `sample`.

        Runs on the device and dtype of `x`, under `torch.no_grad()`, with
        the schedulers reset at the start.

        Args:
            model: Model called at every evaluation; see the class docstring
                for how its output is matched to the sampler.
            x: Initial state of shape ``(batch_size, *sample_shape)``.
            n_steps: Number of sampler updates.
            generator: RNG for the sampler's noise; the global RNG when
                ``None``.
            model_kwargs: Conditioning arguments forwarded to the model at
                every evaluation.
            return_trajectory: If True, return the state after every update,
                shape ``(batch_size, n_steps, *sample_shape)``,
                instead of the final state.

        Returns:
            The final state, or the trajectory.

        Raises:
            ValueError: If `model.prediction_type` cannot feed the sampler.
        """
        raise NotImplementedError

    def _model_call(
        self, model: nn.Module, model_kwargs: Optional[Dict[str, Any]]
    ) -> Callable[..., Any]:
        r"""Closure ``(x, t=None) -> model(...)`` following `timestep_conditioned`."""
        kwargs = model_kwargs or {}
        if self.timestep_conditioned:

            def call(x: torch.Tensor, t: Optional[torch.Tensor] = None):
                return model(x, t.expand(x.shape[0]), **kwargs)

        else:

            def call(x: torch.Tensor, t: Optional[torch.Tensor] = None):
                return model(x, **kwargs)

        return call

    def _model_fn(
        self, model: nn.Module, model_kwargs: Optional[Dict[str, Any]]
    ) -> Callable[..., torch.Tensor]:
        r"""Closure ``(x, t=None) -> model_output`` in the form the sampler
        consumes.

        Raises:
            ValueError: If `model.prediction_type` cannot feed the sampler.
        """
        call = self._model_call(model, model_kwargs)
        consumed = self.prediction_type
        produced = getattr(model, "prediction_type", None)
        if produced is None or produced == consumed:
            return call
        if isinstance(produced, tuple):
            if consumed not in produced:
                raise ValueError(
                    f"model outputs {produced!r}, none of which is the "
                    f"{consumed!r} that {type(self).__name__} consumes"
                )
            index = produced.index(consumed)

            def pick(x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
                return call(x, t)[index]

            return pick
        if produced == "energy" and consumed == "score":
            return _score_from_energy(call, model)
        raise ValueError(
            f"model outputs {produced!r} but {type(self).__name__} consumes "
            f"{consumed!r}; set prediction_type on the model or the sampler"
        )

    def reset_schedulers(self, model: Optional[nn.Module] = None) -> None:
        r"""Reset the schedulers of this sampler and of a `Schedulable` model."""
        super().reset_schedulers()
        if isinstance(model, Schedulable) and model is not self.model:
            model.reset_schedulers()

    def step_schedulers(self, model: Optional[nn.Module] = None) -> None:
        r"""Advance the schedulers of this sampler and of a `Schedulable` model."""
        super().step_schedulers()
        if isinstance(model, Schedulable) and model is not self.model:
            model.step_schedulers()

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError(
                f"{type(self).__name__}.sample() runs the model bound at "
                "construction (deprecated); call sampler(model, x, ...) instead"
            )

    def _init_state(
        self,
        x: Optional[torch.Tensor],
        dim: Optional[Union[int, Tuple[int, ...]]],
        n_samples: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        r"""Coerce `x` to the sampler's device/dtype, or draw from `N(0, I)`.

        Args:
            x: Initial state, or `None` to synthesize one.
            dim: State dimension (int) or shape (tuple), used when `x is None`.
            n_samples: Number of parallel chains, used when `x is None`.
            generator: RNG for the `N(0, I)` draw; the global RNG when `None`.

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
        return torch.randn(
            n_samples, *shape, dtype=self.dtype, device=self.device, generator=generator
        )

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
        *,
        generator: Optional[torch.Generator] = None,
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
            generator: RNG for every draw this call makes (initial state, MCMC
                noise, accept/reject). ``None`` (default) uses the global RNG,
                reproducing the pre-existing behavior exactly. A generator is
                bound to the device it was created for, so it must match the
                sampler's device. In distributed runs, seed one generator per
                rank (e.g. ``base_seed + rank``) to decorrelate chains.

        Returns:
            Either a tensor of samples (or trajectory) of shape
            `[n_samples, *data_shape]` or `[n_samples, n_steps // thin, *data_shape]`,
            optionally paired with a diagnostics dict if `return_diagnostics=True`.

        Raises:
            ValueError: If `thin < 1`, or if `x` is `None` and `dim` is `None`
                for samplers that cannot infer the state shape.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={type(self.model).__name__})"
