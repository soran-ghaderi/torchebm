r"""Flow-based sampler for trained generative models.

Supports ODE (probability flow) and SDE (diffusion) sampling, configured at
construction and executed through the standard `BaseSampler.sample` contract.
"""

import enum
import math
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import torch
from torch import nn

from torchebm.core import (
    BaseInterpolant,
    BaseRungeKuttaIntegrator,
    BaseSampler,
    BaseScheduler,
    BaseSDERungeKuttaIntegrator,
    expand_t_like_x,
)
from torchebm._deprecation import declare_deprecation
from torchebm.integrators.integrator_utils import resolve_integrator
from torchebm.interpolants import (
    CosineInterpolant,
    LinearInterpolant,
    VariancePreservingInterpolant,
)
from torchebm.interpolants.interpolant_utils import resolve_interpolant

# First warned in v0.7.5; window restarted when the deprecation ledger was
# adopted.
_BARE_SAMPLE_KWARGS_DEPRECATION = declare_deprecation(
    module=__name__,
    name="FlowSampler.sample bare conditioning kwargs",
    since="0.8.5",
    deprecated_on="2026-08-31",
    replacement="model_kwargs={...}",
    message=(
        "Passing conditioning to FlowSampler.sample() as bare keyword "
        "arguments is deprecated; pass model_kwargs={...} instead."
    ),
    removal="the **legacy_model_kwargs parameter and its merge in FlowSampler.sample",
)

_MODEL_BOUND_DEPRECATION = declare_deprecation(
    module=__name__,
    name="model-bound FlowSampler",
    since="0.8.10",
    deprecated_on="2026-09-29",
    grace=4,
    grace_months=4,
    replacement="FlowSampler(prediction_type=...)(model, x, num_inference_steps=...)",
    message=(
        "FlowSampler(model=...), prediction=, train_eps=, negate_velocity= and "
        "sample() are deprecated: construct the sampler without a model, name "
        "the model output with prediction_type=, and call sampler(model, x, ...)."
    ),
    removal=(
        "the model, prediction, train_eps and negate_velocity arguments, the "
        "train_eps property, sample, _require_model, _get_drift, _get_score, "
        "_grid_and_drift, _sde_dynamics, _apply_last_step, _record_stats, "
        "_batch_stats and the PredictionType export in FlowSampler"
    ),
)

# Sampling-call kwargs removed with the sample_ode/sample_sde retirement.
# Guarded so stale call sites fail loudly instead of silently forwarding
# these to the model.
_REMOVED_SAMPLE_KWARGS = frozenset(
    {
        "mode",
        "shape",
        "ode_method",
        "sde_method",
        "method",
        "atol",
        "rtol",
        "reverse",
        "diffusion_form",
        "diffusion_norm",
        "last_step",
        "last_step_size",
        "num_steps",
        "z",
    }
)

# Sentinel distinguishing "not passed" from the legitimate last_step=None.
_UNSET = object()

_LAST_STEPS = ("Mean", "Euler", "Tweedie", None)


class PredictionType(str, enum.Enum):
    r"""Model output consumed by `FlowSampler`; compares equal to its name
    as used by the models (``"velocity"``, ``"score"``, ``"epsilon"``)."""

    NOISE = "epsilon"
    SCORE = "score"
    VELOCITY = "velocity"


class FlowSampler(BaseSampler):
    r"""
    Sampler for flow-based and diffusion generative models.

    The sampling process is configured at construction: ``mode="ode"``
    integrates the probability-flow ODE, ``mode="sde"`` the reverse-time
    diffusion SDE. The model is called as ``model(x, t, **model_kwargs)`` and
    its output, named by `prediction_type`, is converted through the
    interpolant to the velocity (and, in SDE mode, the score). With a
    fixed-step integrator `forward` runs the step contract (see
    `BaseSampler`); with an adaptive integrator it solves the ODE and returns
    the final state only.

    `sample()` (deprecated, model-bound) follows the `BaseSampler` contract;
    with a fixed-step integrator it supports ``thin``, ``return_trajectory``,
    and ``return_diagnostics``, while adaptive integrators (``dopri5``, ...)
    return only the final state.

    Args:
        model: Deprecated. Trained network sampled by `sample()`; construct
            without it and call ``sampler(model, x, ...)``.
        mode: `"ode"` (probability flow, default) or `"sde"` (diffusion).
        interpolant: Interpolant name ('linear', 'cosine', 'vp') or instance.
        prediction: Deprecated name of `prediction_type`, with ``"noise"``
            for ``"epsilon"``.
        train_eps: Deprecated and unused.
        sample_eps: Epsilon for sampling time interval. Accepts a float or a
            `BaseScheduler` (advanced via `step_schedulers()`).
        negate_velocity: Deprecated; `sample()` only. Negate the velocity
            during sampling. Set True for EqM models which learn (ε - x)
            direction; velocity is v = -f(x). EqM equilibrium fields are
            time-invariant, so the model is then conditioned on zeroed time
            (matching training) while the integration clock still drives the
            grid and interpolant math.
        reverse: If True, integrate from data to noise (ODE mode only).
        diffusion_form: SDE-only. Form of the diffusion coefficient:
            'constant', 'SBDM' (default), 'sigma', 'linear', 'decreasing',
            or 'increasing-decreasing'.
        diffusion_norm: SDE-only. Scaling factor for diffusion (default 1.0).
        last_step: SDE-only. Final denoising step: 'Mean' (default),
            'Euler', 'Tweedie', or None (no correction).
        last_step_size: SDE-only. Size of the final step (default 0.04).
        prediction_type: Model output `step` consumes: ``"velocity"``
            (default), ``"score"``, or ``"epsilon"``.
        dtype: Data type for computations.
        device: Device for computations.
        integrator: `None` (default) uses `Dopri5Integrator` for ODE mode
            and `EulerMaruyamaIntegrator` for SDE mode; a registry name
            (e.g. `"rk4"`) constructs that integrator with defaults; an
            instance is used as-is and must match the sampler's
            device/dtype. SDE mode requires a `BaseSDERungeKuttaIntegrator`;
            SDE integrators are valid for ODE mode (zero diffusion). The step
            contract needs a fixed-step integrator.

    Example:
        ```python
        import torch
        from torchebm.samplers import FlowSampler

        sampler = FlowSampler(interpolant="linear", integrator="midpoint")
        samples = sampler(velocity_net, torch.randn(100, 2), num_inference_steps=50)
        ```
    """

    timestep_conditioned = True

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        mode: Literal["ode", "sde"] = "ode",
        interpolant: Union[str, BaseInterpolant] = "linear",
        prediction: Optional[Literal["velocity", "score", "noise"]] = None,
        train_eps: Union[float, BaseScheduler, None] = None,
        sample_eps: Union[float, BaseScheduler] = 0.0,
        negate_velocity: bool = False,
        reverse: bool = False,
        diffusion_form: Optional[str] = None,
        diffusion_norm: Optional[float] = None,
        last_step: Union[str, None, object] = _UNSET,
        last_step_size: Optional[float] = None,
        prediction_type: Literal["velocity", "score", "epsilon"] = "velocity",
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        integrator: Union[str, BaseRungeKuttaIntegrator, None] = None,
    ):
        super().__init__(model=model, dtype=dtype, device=device)
        if (
            model is not None
            or prediction is not None
            or train_eps is not None
            or negate_velocity
        ):
            _MODEL_BOUND_DEPRECATION.warn()
        self._register_param("train_eps", 0.0 if train_eps is None else train_eps)
        self._register_param("sample_eps", sample_eps)
        self.negate_velocity = negate_velocity

        if mode not in ("ode", "sde"):
            raise ValueError(f"Unknown mode: {mode!r}. Choose from ['ode', 'sde']")
        self.mode = mode

        # Interpolants are stateless math objects (no device-bound tensors).
        self.interpolant = resolve_interpolant(
            interpolant, default="linear", owner="FlowSampler"
        )

        if prediction is None:
            try:
                self.prediction_type = PredictionType(prediction_type)
            except ValueError:
                raise ValueError(
                    f"Unknown prediction_type: {prediction_type!r}. "
                    "Choose from ['velocity', 'score', 'epsilon']"
                ) from None
        else:
            prediction_map = {
                "velocity": PredictionType.VELOCITY,
                "score": PredictionType.SCORE,
                "noise": PredictionType.NOISE,
            }
            try:
                self.prediction_type = prediction_map[prediction]
            except KeyError:
                raise ValueError(
                    f"Unknown prediction: {prediction!r}. "
                    f"Choose from {list(prediction_map)}"
                ) from None

        if mode == "ode":
            offenders = [
                name
                for name, value in (
                    ("diffusion_form", diffusion_form),
                    ("diffusion_norm", diffusion_norm),
                    ("last_step_size", last_step_size),
                )
                if value is not None
            ]
            if last_step is not _UNSET:
                offenders.append("last_step")
            if offenders:
                raise ValueError(
                    f"{', '.join(sorted(offenders))} only apply to mode='sde'"
                )
            self.diffusion_form = None
            self.diffusion_norm = None
            self.last_step = None
            self.last_step_size = None
        else:
            if reverse:
                raise ValueError("reverse=True is not supported for mode='sde'")
            self.diffusion_form = (
                diffusion_form if diffusion_form is not None else "SBDM"
            )
            self.diffusion_norm = diffusion_norm if diffusion_norm is not None else 1.0
            self.last_step = "Mean" if last_step is _UNSET else last_step
            if self.last_step not in _LAST_STEPS:
                raise ValueError(
                    f"Unknown last_step: {self.last_step!r}. "
                    f"Choose from {list(_LAST_STEPS)}"
                )
            self.last_step_size = last_step_size if last_step_size is not None else 0.04
            if self.last_step is None:
                self.last_step_size = 0.0
        self.reverse = reverse

        family = (
            BaseRungeKuttaIntegrator if mode == "ode" else BaseSDERungeKuttaIntegrator
        )
        self.integrator = resolve_integrator(
            integrator,
            default="dopri5" if mode == "ode" else "euler_maruyama",
            family=family,
            owner="FlowSampler",
            device=self.device,
            dtype=self.dtype,
        )
        if mode == "sde" and self.integrator.error_weights is not None:
            raise ValueError(
                "Adaptive integrators are ODE-only; mode='sde' requires a "
                f"fixed-step integrator, got {type(self.integrator).__name__}"
            )
        self._default_n_steps = 50 if mode == "ode" else 250
        self.order = sum(self._stage_evaluations())

    @property
    def train_eps(self) -> float:
        return self.get_scheduled_value("train_eps")

    @train_eps.setter
    def train_eps(self, value: Union[float, BaseScheduler]) -> None:
        self._register_param("train_eps", value)

    @property
    def sample_eps(self) -> float:
        return self.get_scheduled_value("sample_eps")

    @sample_eps.setter
    def sample_eps(self, value: Union[float, BaseScheduler]) -> None:
        self._register_param("sample_eps", value)

    @torch.no_grad()
    def forward(
        self,
        model: nn.Module,
        sample: torch.Tensor,
        *,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        r"""Run `num_inference_steps` integration steps of `model` from `sample`.

        A fixed-step integrator runs the step contract (see
        `BaseSampler.forward`). An adaptive integrator solves the ODE over the
        sampling interval, `num_inference_steps` setting its initial step
        size, and returns the final state only; it advances no schedulers.

        Raises:
            ValueError: If `num_inference_steps` is not positive.
            NotImplementedError: If `return_trajectory` is requested with an
                adaptive integrator.
        """
        if self.integrator.error_weights is None:
            return super().forward(
                model,
                sample,
                num_inference_steps=num_inference_steps,
                generator=generator,
                model_kwargs=model_kwargs,
                return_trajectory=return_trajectory,
            )
        if return_trajectory:
            raise NotImplementedError(
                "return_trajectory needs a fixed-step integrator; adaptive "
                f"{type(self.integrator).__name__} returns only the final state"
            )
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        model_output = self._model_fn(model, model_kwargs)
        self.reset_schedulers()
        t0, t1 = self._check_interval()
        grid = torch.linspace(
            t0, t1, num_inference_steps + 1, device=sample.device, dtype=self.dtype
        )

        def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self._velocity(model_output(x, t), x, t)

        drift = velocity
        if self.reverse:
            grid = grid - t0

            def drift(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
                return -velocity(x, t0 + s)

        return self.integrator.integrate(
            state={"x": sample},
            step_size=grid[1] - grid[0],
            n_steps=num_inference_steps,
            drift=drift,
            t=grid,
        )["x"]

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        r"""Build the stage times of `num_inference_steps` integration steps
        over the sampling interval and reset the schedulers.

        `timesteps` holds the interpolant time of every model evaluation,
        `order` per step (an implicit stage repeats its time once per Picard
        iteration). In SDE mode with a `last_step` correction it ends with
        the final time, one more evaluation than the integration steps.

        Raises:
            ValueError: If `num_inference_steps` is not positive.
            NotImplementedError: If the integrator has no fixed evaluation
                count per update.
        """
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        self._check_step_integrator()
        self.reset_schedulers()
        self.num_inference_steps = num_inference_steps
        t0, t1 = self._check_interval()
        t_phys = torch.linspace(
            t0,
            t1,
            num_inference_steps + 1,
            device=self.device if device is None else device,
            dtype=self.dtype,
        )
        grid = t_phys - t0 if self.reverse else t_phys
        self._step_sizes = grid[1:] - grid[:-1]
        _, _, c, _ = self.integrator._tableau_on(grid.device, grid.dtype)
        stage_times = grid[:-1, None] + c * self._step_sizes[:, None]
        if self.order != c.shape[0]:
            stage_times = stage_times.repeat_interleave(
                torch.tensor(self._stage_evaluations(), device=grid.device), dim=1
            )
        timesteps = stage_times.reshape(-1)
        if self.reverse:
            timesteps = t0 + timesteps
        self._last_step_index = -1
        if self.mode == "sde" and self.last_step is not None:
            timesteps = torch.cat([timesteps, t_phys[-1:]])
            self._last_step_index = timesteps.shape[0] - 1
        self.timesteps = timesteps
        self._step_index = 0
        self._stage = self._iteration = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        r"""One integrator stage from the model output at ``(sample, timestep)``.

        Args:
            model_output: Output named by `prediction_type`, evaluated at
                `sample` and `timestep`.
            timestep: Entry of `timesteps`, a 0-d tensor.
            sample: Current state, or the point returned by the previous call
                while a multi-stage or implicit update is in progress.
            generator: RNG for the SDE noise; the global RNG when ``None``.

        Returns:
            The next state, or the next point to evaluate of an unfinished
            update.
        """
        i = self._step_index
        self._step_index = i + 1
        t = timestep.expand(sample.shape[0])
        if i == self._last_step_index:
            return self._denoise(model_output, sample, t)
        sde = self.mode == "sde"
        drift = self._velocity(model_output, sample, t)
        if self.reverse:
            drift = -drift
        if sde:
            diffusion = self.interpolant.compute_diffusion(
                sample, t, form=self.diffusion_form, norm=self.diffusion_norm
            )
            if self._stage == 0 and self._iteration == 0:
                self._diffusion = diffusion
            drift = drift + diffusion * self._score(model_output, sample, t)
        step_size = self._step_sizes[i // self.order]
        x = self._runge_kutta_stage(drift, sample, step_size)
        if self._stage or self._iteration:
            return x
        if sde:
            noise = torch.randn_like(x, generator=generator)
            x = x + (2.0 * self._diffusion) ** 0.5 * (noise * step_size**0.5)
        self.step_schedulers()
        return x

    def _velocity(
        self, model_output: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        r"""Probability-flow velocity from the model output at ``(x, t)``."""
        if self.prediction_type is PredictionType.VELOCITY:
            return model_output
        drift_mean, drift_var = self.interpolant.compute_drift(x, t)
        return -drift_mean + drift_var * self._score(model_output, x, t)

    def _score(
        self, model_output: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        r"""Score from the model output at ``(x, t)``."""
        if self.prediction_type is PredictionType.SCORE:
            return model_output
        if self.prediction_type is PredictionType.VELOCITY:
            return self.interpolant.velocity_to_score(model_output, x, t)
        sigma_t, _ = self.interpolant.compute_sigma_t(expand_t_like_x(t, x))
        return model_output / (-sigma_t + 1e-8)

    def _denoise(
        self, model_output: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        r"""The SDE `last_step` correction from the model output at the final time."""
        if self.last_step == "Mean":
            diffusion = self.interpolant.compute_diffusion(
                x, t, form=self.diffusion_form, norm=self.diffusion_norm
            )
            drift = self._velocity(model_output, x, t) + diffusion * self._score(
                model_output, x, t
            )
            return x + drift * self.last_step_size
        if self.last_step == "Euler":
            return x + self._velocity(model_output, x, t) * self.last_step_size
        t_expanded = expand_t_like_x(t, x)
        alpha, _ = self.interpolant.compute_alpha_t(t_expanded)
        sigma, _ = self.interpolant.compute_sigma_t(t_expanded)
        return x / alpha + sigma.square() / alpha * self._score(model_output, x, t)

    def _get_drift(self) -> Callable:
        r"""Get drift function for probability flow ODE."""

        def velocity_drift(x, t, **model_kwargs):
            if self.negate_velocity:
                # EqM equilibrium fields are time-invariant: trained with
                # zeroed time, so the model is conditioned on zeros here too;
                # t still drives the integration grid.
                return -self.model(x, torch.zeros_like(t), **model_kwargs)
            return self.model(x, t, **model_kwargs)

        def score_drift(x, t, **model_kwargs):
            drift_mean, drift_var = self.interpolant.compute_drift(x, t)
            model_output = self.model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output

        def noise_drift(x, t, **model_kwargs):
            drift_mean, drift_var = self.interpolant.compute_drift(x, t)
            t_expanded = expand_t_like_x(t, x)
            sigma_t, _ = self.interpolant.compute_sigma_t(t_expanded)
            model_output = self.model(x, t, **model_kwargs)
            score = model_output / (-sigma_t + 1e-8)
            return -drift_mean + drift_var * score

        drifts = {
            PredictionType.VELOCITY: velocity_drift,
            PredictionType.SCORE: score_drift,
            PredictionType.NOISE: noise_drift,
        }
        return drifts[self.prediction_type]

    def _get_score(self) -> Callable:
        r"""Get score function from model output."""

        def velocity_score(x, t, **model_kwargs):
            if self.negate_velocity:
                # Match velocity_drift: the sampling velocity is v = -f with
                # the field evaluated at zeroed time.
                velocity = -self.model(x, torch.zeros_like(t), **model_kwargs)
            else:
                velocity = self.model(x, t, **model_kwargs)
            return self.interpolant.velocity_to_score(velocity, x, t)

        def score_score(x, t, **model_kwargs):
            return self.model(x, t, **model_kwargs)

        def noise_score(x, t, **model_kwargs):
            t_expanded = expand_t_like_x(t, x)
            sigma_t, _ = self.interpolant.compute_sigma_t(t_expanded)
            return self.model(x, t, **model_kwargs) / (-sigma_t + 1e-8)

        scores = {
            PredictionType.VELOCITY: velocity_score,
            PredictionType.SCORE: score_score,
            PredictionType.NOISE: noise_score,
        }
        return scores[self.prediction_type]

    def _check_interval(self) -> Tuple[float, float]:
        r"""Forward time interval `(t0, t1)` for the configured process."""
        t0 = 0.0
        t1 = 1.0
        eps = self.sample_eps
        sde = self.mode == "sde"
        last_step_size = self.last_step_size if sde else 0.0

        is_vp = isinstance(self.interpolant, VariancePreservingInterpolant)
        is_linear_or_cosine = isinstance(
            self.interpolant, (LinearInterpolant, CosineInterpolant)
        )

        if is_vp:
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        elif is_linear_or_cosine and (
            self.prediction_type != PredictionType.VELOCITY or sde
        ):
            t0 = (
                eps
                if (self.diffusion_form == "SBDM" and sde)
                or self.prediction_type != PredictionType.VELOCITY
                else 0
            )
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        return t0, t1

    def _grid_and_drift(
        self, drift_fn: Callable, n_steps: int
    ) -> Tuple[Callable, torch.Tensor, torch.Tensor]:
        r"""Integration drift, grid, and physical model times per kept point.

        Forward mode integrates ``dx/dt = f(x, t)`` on ``[t0, t1]``. Reverse
        mode applies the change of variables ``s = t - t0``, integrating
        ``dy/ds = -f(y, t0 + s)`` on ``[0, t1 - t0]``; this single form
        serves fixed-step and adaptive integrators alike. The returned
        ``t_phys`` is the interpolant time at each grid point (identical in
        both modes).
        """
        t0, t1 = self._check_interval()
        t_phys = torch.linspace(
            t0, t1, n_steps + 1, device=self.device, dtype=self.dtype
        )
        if not self.reverse:
            return drift_fn, t_phys, t_phys

        def reversed_drift(x, s):
            return -drift_fn(x, t0 + s)

        grid = t_phys - t0
        return reversed_drift, grid, t_phys

    def _sde_dynamics(self) -> Tuple[Callable, Callable]:
        r"""Reverse-SDE drift and diffusion callables.

        The diffusion coefficient enters twice by design: as the score
        correction inside the drift and as the Wiener noise magnitude.
        """
        drift_fn = self._get_drift()
        score_fn = self._get_score()

        def diffusion_fn(x, t):
            return self.interpolant.compute_diffusion(
                x, t, form=self.diffusion_form, norm=self.diffusion_norm
            )

        def sde_drift(x, t, **model_kwargs):
            diffusion = diffusion_fn(x, t)
            return drift_fn(x, t, **model_kwargs) + diffusion * score_fn(
                x, t, **model_kwargs
            )

        return sde_drift, diffusion_fn

    @torch.no_grad()
    def sample(
        self,
        x: Optional[torch.Tensor] = None,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        n_steps: Optional[int] = None,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
        reset_schedulers: bool = True,
        *,
        y: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        generator: Optional[torch.Generator] = None,
        **legacy_model_kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        r"""Sample by integrating the configured ODE or SDE.

        Args:
            x: Initial state (typically noise). If `None`, samples from
                `N(0, I)`.
            dim: State dimension (int) or shape (tuple), used when `x is None`.
            n_steps: Number of integration steps. `None` (default) resolves
                to 50 in ODE mode and 250 in SDE mode. For adaptive
                integrators only the implied time interval matters.
            n_samples: Number of parallel samples.
            thin: Keep every `thin`-th step. Final stored length is
                `n_steps // thin`. Must be `>= 1`. Fixed-step integrators only.
            return_trajectory: If True, return the full kept trajectory of
                shape `[n_samples, n_steps // thin, *data_shape]`. Fixed-step
                integrators only. In SDE mode the final kept entry reflects
                the `last_step` correction, so the trajectory ends at the
                returned sample.
            return_diagnostics: If True, also return a dict with keys
                ``"mean"`` (`[n_kept, *data_shape]`), ``"var"``
                (`[n_kept, *data_shape]`), and ``"t"`` (`[n_kept]`, the
                interpolant time of each kept step). With an adaptive
                integrator the dict has a single entry for the final state.
            reset_schedulers: If True (default), reset registered schedulers.
                Fixed-step sampling advances schedulers once per step;
                adaptive integrators do not step them (the actual step count
                is controller-dependent).
            y: Optional conditioning tensor forwarded to the model; shorthand
                for ``model_kwargs={'y': y}``.
            model_kwargs: Conditioning arguments (e.g. class labels) forwarded to
                the model at every step. Normalized to the sampler device once at
                entry; ``None`` (default) is the exact unconditional path.
            generator: RNG for the initial state and, in SDE mode, the per-step
                diffusion noise; the global RNG when ``None``. ODE mode is
                deterministic once the initial state is fixed.
            **legacy_model_kwargs: Deprecated. Passing conditioning as bare
                keyword arguments still works for one release but emits a
                ``DeprecationWarning``; pass ``model_kwargs={...}`` instead. When
                both are given, keys in the explicit dict win.

        Returns:
            Sample tensor (or trajectory if `return_trajectory=True`),
            optionally paired with the diagnostics dict.

        Raises:
            ValueError: If `thin < 1`, `n_steps <= 0`, or `x` and `dim` are
                both `None`.
            NotImplementedError: If `return_trajectory=True` or `thin != 1`
                with an adaptive integrator.
            TypeError: If a removed legacy sampling kwarg is passed.
        """
        self._require_model()
        removed = _REMOVED_SAMPLE_KWARGS.intersection(legacy_model_kwargs)
        if removed:
            raise TypeError(
                f"{', '.join(sorted(removed))}: removed from "
                "FlowSampler.sample(). mode/reverse/diffusion/"
                "last-step options and the integrator moved to the "
                "constructor; use x/n_steps instead of z/num_steps and "
                "dim=(...) instead of shape."
            )
        if legacy_model_kwargs:
            _BARE_SAMPLE_KWARGS_DEPRECATION.warn()
        model_kwargs = self._prepare_model_kwargs(
            self._merge_condition(
                {**legacy_model_kwargs, **(model_kwargs or {})}, y
            )
        )
        if thin < 1:
            raise ValueError("thin must be >= 1")
        if n_steps is None:
            n_steps = self._default_n_steps
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        adaptive = self.integrator.error_weights is not None
        if adaptive and (return_trajectory or thin != 1):
            raise NotImplementedError(
                "return_trajectory/thin require a fixed-step integrator; "
                f"adaptive {type(self.integrator).__name__} returns only the "
                "final state. Construct FlowSampler(integrator='euler') or "
                "another fixed-step method."
            )
        if reset_schedulers:
            self.reset_schedulers()

        x = self._init_state(x, dim, n_samples, generator)
        n_samples = x.shape[0]
        data_shape = x.shape[1:]

        sde = self.mode == "sde"
        if sde:
            sde_drift, diffusion_fn = self._sde_dynamics()

            def base_drift(x_, t_):
                return sde_drift(x_, t_, **model_kwargs)

        else:
            drift_fn = self._get_drift()

            def base_drift(x_, t_):
                return drift_fn(x_, t_, **model_kwargs)

        drift, grid, t_phys = self._grid_and_drift(base_drift, n_steps)

        if adaptive:
            with self.autocast_context():
                x = self.integrator.integrate(
                    state={"x": x},
                    step_size=grid[1] - grid[0],
                    n_steps=n_steps,
                    drift=drift,
                    t=grid,
                )["x"]
            if not return_diagnostics:
                return x
            return x, self._batch_stats(x, t_phys[-1:])

        n_kept = n_steps // thin
        if return_trajectory:
            trajectory = torch.empty(
                (n_samples, n_kept, *data_shape), dtype=self.dtype, device=self.device
            )
        diagnostics: Optional[Dict[str, torch.Tensor]] = None
        if return_diagnostics:
            diagnostics = {
                "mean": torch.empty(
                    n_kept, *data_shape, dtype=self.dtype, device=self.device
                ),
                "var": torch.empty(
                    n_kept, *data_shape, dtype=self.dtype, device=self.device
                ),
                "t": torch.empty(n_kept, dtype=self.dtype, device=self.device),
            }

        keep_idx = 0
        with self.autocast_context():
            for i in range(n_steps):
                dt = grid[i + 1] - grid[i]
                t_batch = grid[i].expand(n_samples)
                if sde:
                    diff_val = diffusion_fn(x, t_batch)
                    x = self.integrator.step(
                        state={"x": x},
                        step_size=dt,
                        drift=drift,
                        diffusion=diff_val,
                        t=t_batch,
                        generator=generator,
                    )["x"]
                else:
                    x = self.integrator.step(
                        state={"x": x}, step_size=dt, drift=drift, t=t_batch
                    )["x"]
                self.step_schedulers()

                if (i + 1) % thin == 0:
                    if return_trajectory:
                        trajectory[:, keep_idx] = x
                    if return_diagnostics:
                        self._record_stats(diagnostics, keep_idx, x, t_phys[i + 1])
                    keep_idx += 1

        if sde and self.last_step is not None:
            t1 = t_phys[-1]
            t_final = t1.expand(n_samples)
            x = self._apply_last_step(
                x,
                t_final,
                sde_drift,
                self.last_step,
                self.last_step_size,
                **model_kwargs,
            )
            # Keep the recorded end state equal to the returned sample.
            if n_kept > 0 and n_steps % thin == 0:
                if return_trajectory:
                    trajectory[:, -1] = x
                if return_diagnostics:
                    self._record_stats(
                        diagnostics, n_kept - 1, x, t1 + self.last_step_size
                    )

        output = trajectory if return_trajectory else x
        return (output, diagnostics) if return_diagnostics else output

    def _record_stats(
        self,
        diagnostics: Dict[str, torch.Tensor],
        keep_idx: int,
        x: torch.Tensor,
        t: Union[float, torch.Tensor],
    ) -> None:
        r"""Record batch mean/var and the interpolant time at a kept step."""
        if x.shape[0] > 1:
            diagnostics["mean"][keep_idx] = x.mean(dim=0)
            diagnostics["var"][keep_idx] = x.var(dim=0, unbiased=False).clamp_(
                min=1e-10, max=1e10
            )
        else:
            diagnostics["mean"][keep_idx] = x.squeeze(0)
            diagnostics["var"][keep_idx].zero_()
        diagnostics["t"][keep_idx] = t

    def _batch_stats(self, x: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""Single-entry diagnostics for the final state (adaptive path)."""
        diagnostics = {
            "mean": torch.empty(1, *x.shape[1:], dtype=self.dtype, device=self.device),
            "var": torch.empty(1, *x.shape[1:], dtype=self.dtype, device=self.device),
            "t": torch.empty(1, dtype=self.dtype, device=self.device),
        }
        self._record_stats(diagnostics, 0, x, t[0])
        return diagnostics

    def _apply_last_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        sde_drift: Callable,
        last_step: str,
        last_step_size: float,
        **model_kwargs,
    ) -> torch.Tensor:
        r"""Apply final denoising step."""
        if last_step == "Mean":
            return x + sde_drift(x, t, **model_kwargs) * last_step_size
        elif last_step == "Euler":
            drift_fn = self._get_drift()
            return x + drift_fn(x, t, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            t_expanded = expand_t_like_x(t, x)
            alpha, _ = self.interpolant.compute_alpha_t(t_expanded)
            sigma, _ = self.interpolant.compute_sigma_t(t_expanded)
            score = self._get_score()(x, t, **model_kwargs)
            return x / alpha + sigma.square() / alpha * score
        else:
            return x

    def prior_logp(self, z: torch.Tensor) -> torch.Tensor:
        r"""Compute log probability under standard Gaussian prior."""
        N = z[0].numel()
        return (
            -N / 2.0 * math.log(2 * math.pi)
            - torch.sum(z.square(), dim=tuple(range(1, z.ndim))) / 2.0
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={type(self.model).__name__}, "
            f"mode={self.mode!r}, "
            f"interpolant={type(self.interpolant).__name__}, "
            f"prediction={self.prediction_type.name.lower()!r}, "
            f"integrator={type(self.integrator).__name__})"
        )


__all__ = ["FlowSampler", "PredictionType"]
