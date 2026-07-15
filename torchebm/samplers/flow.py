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
    warn_once,
)
from torchebm.integrators.integrator_utils import resolve_integrator
from torchebm.interpolants import (
    CosineInterpolant,
    LinearInterpolant,
    VariancePreservingInterpolant,
)
from torchebm.interpolants.interpolant_utils import resolve_interpolant

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


class PredictionType(enum.Enum):
    r"""Model prediction type for generative models."""

    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()


class FlowSampler(BaseSampler):
    r"""
    Sampler for flow-based and diffusion generative models.

    The sampling process is configured at construction: ``mode="ode"``
    integrates the probability-flow ODE, ``mode="sde"`` the reverse-time
    diffusion SDE. `sample()` follows the standard `BaseSampler` contract;
    with a fixed-step integrator it supports ``thin``, ``return_trajectory``,
    and ``return_diagnostics``, while adaptive integrators (``dopri5``, ...)
    return only the final state.

    Args:
        model: Trained neural network predicting velocity/score/noise,
            called as ``model(x, t, **model_kwargs)``.
        mode: `"ode"` (probability flow, default) or `"sde"` (diffusion).
        interpolant: Interpolant name ('linear', 'cosine', 'vp') or instance.
        prediction: Model prediction type ('velocity', 'score', or 'noise').
        train_eps: Epsilon used during training for time interval stability.
            Accepts a float or a `BaseScheduler`.
        sample_eps: Epsilon for sampling time interval. Accepts a float or a
            `BaseScheduler` (advanced via `step_schedulers()`).
        negate_velocity: Negate the velocity during sampling. Set True for
            EqM models which learn (ε - x) direction; velocity is v = -f(x).
        reverse: If True, integrate from data to noise (ODE mode only).
        diffusion_form: SDE-only. Form of the diffusion coefficient:
            'constant', 'SBDM' (default), 'sigma', 'linear', 'decreasing',
            or 'increasing-decreasing'.
        diffusion_norm: SDE-only. Scaling factor for diffusion (default 1.0).
        last_step: SDE-only. Final denoising step: 'Mean' (default),
            'Euler', 'Tweedie', or None (no correction).
        last_step_size: SDE-only. Size of the final step (default 0.04).
        dtype: Data type for computations.
        device: Device for computations.
        integrator: `None` (default) uses `Dopri5Integrator` for ODE mode
            and `EulerMaruyamaIntegrator` for SDE mode; a registry name
            (e.g. `"rk4"`) constructs that integrator with defaults; an
            instance is used as-is and must match the sampler's
            device/dtype. SDE mode requires a `BaseSDERungeKuttaIntegrator`;
            SDE integrators are valid for ODE mode (zero diffusion).

    Example:
        ```python
        from torchebm.samplers import FlowSampler
        import torch.nn as nn
        import torch

        model = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 2))
        sampler = FlowSampler(
            model=model,
            interpolant="linear",
            prediction="velocity",
            integrator="euler",
        )
        samples = sampler.sample(n_samples=100, dim=2, n_steps=50)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        mode: Literal["ode", "sde"] = "ode",
        interpolant: Union[str, BaseInterpolant] = "linear",
        prediction: Literal["velocity", "score", "noise"] = "velocity",
        train_eps: Union[float, BaseScheduler] = 0.0,
        sample_eps: Union[float, BaseScheduler] = 0.0,
        negate_velocity: bool = False,
        reverse: bool = False,
        diffusion_form: Optional[str] = None,
        diffusion_norm: Optional[float] = None,
        last_step: Union[str, None, object] = _UNSET,
        last_step_size: Optional[float] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        integrator: Union[str, BaseRungeKuttaIntegrator, None] = None,
    ):
        super().__init__(model=model, dtype=dtype, device=device)
        self._register_param("train_eps", train_eps)
        self._register_param("sample_eps", sample_eps)
        self.negate_velocity = negate_velocity

        if mode not in ("ode", "sde"):
            raise ValueError(f"Unknown mode: {mode!r}. Choose from ['ode', 'sde']")
        self.mode = mode

        # Interpolants are stateless math objects (no device-bound tensors).
        self.interpolant = resolve_interpolant(
            interpolant, default="linear", owner="FlowSampler"
        )

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

    def _get_drift(self) -> Callable:
        r"""Get drift function for probability flow ODE."""

        def velocity_drift(x, t, **model_kwargs):
            v = self.model(x, t, **model_kwargs)
            return -v if self.negate_velocity else v

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
        model_kwargs: Optional[Dict[str, Any]] = None,
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
            model_kwargs: Conditioning arguments (e.g. class labels) forwarded to
                the model at every step. Normalized to the sampler device once at
                entry; ``None`` (default) is the exact unconditional path.
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
            warn_once(
                "flowsampler-bare-model-kwargs",
                "Passing conditioning to FlowSampler.sample() as bare keyword "
                "arguments is deprecated; pass model_kwargs={...} instead.",
            )
        # Explicit dict wins over the deprecated bare kwargs; normalize once.
        model_kwargs = self._prepare_model_kwargs(
            {**legacy_model_kwargs, **(model_kwargs or {})}
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

        x = self._init_state(x, dim, n_samples)
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


__all__ = ["FlowSampler", "PredictionType"]
