r"""Flow-based sampler for trained generative models.

Supports both ODE (probability flow) and SDE (diffusion) sampling modes
with various numerical integration methods.
"""

import enum
import warnings
from typing import Callable, Literal, Optional, Tuple, Union

import torch
from torch import nn
import numpy as np

from torchebm.core import (
    BaseInterpolant,
    BaseRungeKuttaIntegrator,
    BaseSampler,
    BaseScheduler,
    BaseSDERungeKuttaIntegrator,
    expand_t_like_x,
)
from torchebm.integrators.integrator_utils import (
    get_integrator,
    resolve_integrator,
)
from torchebm.interpolants import (
    # BaseInterpolant,
    LinearInterpolant,
    CosineInterpolant,
    VariancePreservingInterpolant,
    # expand_t_like_x,
)
from torchebm.losses.loss_utils import get_interpolant


class PredictionType(enum.Enum):
    r"""Model prediction type for generative models."""

    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()


class FlowSampler(BaseSampler):
    r"""
    Sampler for flow-based and diffusion generative models.

    Supports ODE (probability flow) and SDE (diffusion) sampling with various
    numerical integration methods including Euler, Heun, and adaptive solvers.

    Args:
        model: Trained neural network predicting velocity/score/noise.
        interpolant: Interpolant type ('linear', 'cosine', 'vp') or instance.
        prediction: Model prediction type ('velocity', 'score', or 'noise').
        train_eps: Epsilon used during training for time interval stability.
            Accepts a float or a `BaseScheduler`.
        sample_eps: Epsilon for sampling time interval. Accepts a float or a
            `BaseScheduler` (advanced via `step_schedulers()`).
        negate_velocity: Negate the velocity during sampling. Set True for
            EqM models which learn (ε - x) direction; velocity is v = -f(x).
        dtype: Data type for computations.
        device: Device for computations.
        integrator: Integrator used by `sample_ode`/`sample_sde`. `None`
            (default) keeps the per-mode defaults (`Dopri5Integrator` for
            ODE, `EulerMaruyamaIntegrator` for SDE); a registry name (e.g.
            `"rk4"`) constructs that integrator with defaults; a
            `BaseRungeKuttaIntegrator` instance is used as-is and must
            match the sampler's device/dtype. SDE sampling additionally
            requires a `BaseSDERungeKuttaIntegrator`; SDE integrators are
            valid for ODE sampling (zero diffusion).

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
        z = torch.randn(100, 2)
        samples = sampler.sample_ode(z, num_steps=50)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        interpolant: Union[str, BaseInterpolant] = "linear",
        prediction: Literal["velocity", "score", "noise"] = "velocity",
        train_eps: Union[float, BaseScheduler] = 0.0,
        sample_eps: Union[float, BaseScheduler] = 0.0,
        negate_velocity: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        integrator: Union[str, BaseRungeKuttaIntegrator, None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            dtype=dtype,
            device=device,
        )
        self._register_param("train_eps", train_eps)
        self._register_param("sample_eps", sample_eps)
        self.negate_velocity = negate_velocity

        if isinstance(interpolant, str):
            self.interpolant = get_interpolant(interpolant)
        else:
            self.interpolant = interpolant

        prediction_map = {
            "velocity": PredictionType.VELOCITY,
            "score": PredictionType.SCORE,
            "noise": PredictionType.NOISE,
        }
        self.prediction_type = prediction_map[prediction]
        # Interpolants are stateless math objects (no device-bound tensors).

        if integrator is None:
            # Per-mode defaults; resolved by _integrator_for at call time.
            self.integrator = None
            self._ode_default = get_integrator(
                "dopri5", device=self.device, dtype=self.dtype
            )
            self._sde_default = get_integrator(
                "euler_maruyama", device=self.device, dtype=self.dtype
            )
        else:
            self.integrator = resolve_integrator(
                integrator,
                default="dopri5",
                family=BaseRungeKuttaIntegrator,
                owner="FlowSampler",
                device=self.device,
                dtype=self.dtype,
            )
            self._ode_default = None
            self._sde_default = None

    def _integrator_for(
        self,
        mode: Literal["ode", "sde"],
        integ: Optional[BaseRungeKuttaIntegrator] = None,
    ) -> BaseRungeKuttaIntegrator:
        r"""Return the integrator to use for ``mode``, validated.

        Args:
            mode: `"ode"` or `"sde"`.
            integ: Pre-resolved integrator (deprecated per-call path).
                `None` uses the constructor integrator or the mode default.

        Raises:
            TypeError: If SDE sampling is requested with an integrator that
                cannot handle diffusion.
        """
        if integ is None:
            integ = self.integrator
        if integ is None:
            integ = self._ode_default if mode == "ode" else self._sde_default
        if mode == "sde" and not isinstance(integ, BaseSDERungeKuttaIntegrator):
            raise TypeError(
                "SDE sampling requires a BaseSDERungeKuttaIntegrator; got "
                f"{type(integ).__name__}"
            )
        return integ

    def _deprecated_call_integrator(
        self,
        mode: Literal["ode", "sde"],
        method: Optional[str],
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ) -> Optional[BaseRungeKuttaIntegrator]:
        r"""Resolve the deprecated per-call ``method``/``atol``/``rtol`` args.

        Returns `None` when no deprecated argument was given.
        """
        if method is None and atol is None and rtol is None:
            return None
        warnings.warn(
            "method/atol/rtol on FlowSampler sampling calls are deprecated; "
            "pass integrator= to the FlowSampler constructor (atol/rtol are "
            "set on the integrator instance).",
            # 4 frames: helper -> sample_ode/sample_sde -> torch.no_grad
            # wrapper -> caller.
            DeprecationWarning,
            stacklevel=4,
        )
        default = "dopri5" if mode == "ode" else "euler_maruyama"
        integ = get_integrator(
            method if method is not None else default,
            device=self.device,
            dtype=self.dtype,
        )
        if atol is not None:
            integ.atol = atol
        if rtol is not None:
            integ.rtol = rtol
        return integ

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
    def sample(
        self,
        x: Optional[torch.Tensor] = None,
        dim: int = 10,
        n_steps: int = 50,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
        reset_schedulers: bool = True,
        *,
        mode: Literal["ode", "sde"] = "ode",
        shape: Optional[Tuple[int, ...]] = None,
        ode_method: Optional[str] = None,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        reverse: bool = False,
        sde_method: Optional[str] = None,
        diffusion_form: str = "SBDM",
        diffusion_norm: float = 1.0,
        last_step: Optional[str] = "Mean",
        last_step_size: float = 0.04,
        **model_kwargs,
    ) -> torch.Tensor:
        r"""Unified sampling entrypoint for flow/diffusion models.

        Conforms to the `BaseSampler.sample` contract for the common kwargs.
        Flow-specific options (`mode`, `reverse`, diffusion settings) are
        keyword-only. For full control, call `sample_ode` or `sample_sde`
        directly. The integrator is set at construction via the
        ``integrator`` argument; `ode_method`/`sde_method`/`atol`/`rtol`
        are deprecated per-call overrides.

        ``thin``, ``return_trajectory``, and ``return_diagnostics`` are not
        supported by ODE/SDE solvers (they integrate continuously, not stepwise);
        passing non-default values raises `NotImplementedError`. ``reset_schedulers``
        is honored for `train_eps` / `sample_eps` schedules.

        Raises:
            NotImplementedError: If `thin != 1`, `return_trajectory=True`, or
                `return_diagnostics=True`.
            ValueError: If `mode` is not `"ode"` or `"sde"`.
        """
        if thin != 1:
            raise NotImplementedError("thin is not supported for FlowSampler")
        if return_trajectory or return_diagnostics:
            raise NotImplementedError(
                "FlowSampler does not support trajectories/diagnostics"
            )
        if reset_schedulers:
            self.reset_schedulers()

        if x is None:
            if shape is not None:
                z = torch.randn(*shape, device=self.device, dtype=self.dtype)
            else:
                z = torch.randn(n_samples, dim, device=self.device, dtype=self.dtype)
        else:
            z = x.to(device=self.device, dtype=self.dtype)

        if mode == "ode":
            return self.sample_ode(
                z=z,
                num_steps=n_steps,
                method=ode_method,
                atol=atol,
                rtol=rtol,
                reverse=reverse,
                **model_kwargs,
            )
        if mode == "sde":
            return self.sample_sde(
                z=z,
                num_steps=n_steps,
                method=sde_method,
                diffusion_form=diffusion_form,
                diffusion_norm=diffusion_norm,
                last_step=last_step,
                last_step_size=last_step_size,
                **model_kwargs,
            )
        raise ValueError(f"Unknown mode: {mode}")

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

    def _check_interval(
        self,
        sde: bool = False,
        reverse: bool = False,
        last_step_size: float = 0.0,
        diffusion_form: str = "SBDM",
    ) -> Tuple[float, float]:
        r"""Compute time interval for sampling."""
        t0 = 0.0
        t1 = 1.0
        eps = self.sample_eps

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
                if (diffusion_form == "SBDM" and sde)
                or self.prediction_type != PredictionType.VELOCITY
                else 0
            )
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    @torch.no_grad()
    def sample_ode(
        self,
        z: torch.Tensor,
        num_steps: int = 50,
        method: Optional[str] = None,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        reverse: bool = False,
        **model_kwargs,
    ) -> torch.Tensor:
        r"""
        Sample using probability flow ODE.

        The solver is the constructor ``integrator`` (default: adaptive
        `Dopri5Integrator`).

        Args:
            z: Initial noise tensor of shape (batch_size, ...).
            num_steps: Number of discretization steps (for fixed-step methods).
            method: Deprecated. Pass ``integrator=`` to the constructor.
            atol: Deprecated. Set on the integrator instance.
            rtol: Deprecated. Set on the integrator instance.
            reverse: If True, sample from data to noise.
            **model_kwargs: Additional arguments passed to the model.

        Returns:
            Generated samples tensor.
        """
        z = z.to(device=self.device, dtype=self.dtype)
        integ = self._integrator_for(
            "ode", self._deprecated_call_integrator("ode", method, atol, rtol)
        )
        drift_fn = self._get_drift()

        t0, t1 = self._check_interval(sde=False, reverse=reverse)
        # num_steps is the number of integration steps, so we need num_steps+1 time points
        t = torch.linspace(t0, t1, num_steps + 1, device=self.device, dtype=self.dtype)

        if reverse:

            def wrapped_drift(x, t_val, **kwargs):
                return drift_fn(x, 1.0 - t_val, **kwargs)

        else:
            wrapped_drift = drift_fn

        def fixed_step_drift(x, t_batch):
            return wrapped_drift(x, t_batch, **model_kwargs)

        drift = fixed_step_drift
        if t1 < t0 and integ.error_weights is not None:
            # Adaptive stepping assumes increasing time. Integrate the
            # exact change of variables s = t0 - t on a forward grid;
            # fixed-step integrators handle the raw decreasing grid.
            def reversed_drift(x, s_batch):
                return -fixed_step_drift(x, t0 - s_batch)

            drift = reversed_drift
            t = torch.linspace(
                0.0, t0 - t1, num_steps + 1, device=self.device, dtype=self.dtype
            )

        return integ.integrate(
            state={"x": z},
            step_size=t[1] - t[0],
            n_steps=num_steps,
            drift=drift,
            t=t,
        )["x"]

    @torch.no_grad()
    def sample_sde(
        self,
        z: torch.Tensor,
        num_steps: int = 250,
        method: Optional[str] = None,
        diffusion_form: str = "SBDM",
        diffusion_norm: float = 1.0,
        last_step: Optional[str] = "Mean",
        last_step_size: float = 0.04,
        **model_kwargs,
    ) -> torch.Tensor:
        r"""
        Sample using reverse-time SDE.

        The solver is the constructor ``integrator`` (default:
        `EulerMaruyamaIntegrator`) and must be a
        `BaseSDERungeKuttaIntegrator`.

        Args:
            z: Initial noise tensor of shape (batch_size, ...).
            num_steps: Number of discretization steps.
            method: Deprecated. Pass ``integrator=`` to the constructor.
            diffusion_form: Form of diffusion coefficient. Choices:
                - 'constant': Constant diffusion
                - 'SBDM': Score-based diffusion matching (default)
                - 'sigma': Proportional to noise schedule
                - 'linear': Linear decay
                - 'decreasing': Faster decay towards t=1
                - 'increasing-decreasing': Peak at midpoint
            diffusion_norm: Scaling factor for diffusion.
            last_step: Type of last step ('Mean', 'Tweedie', 'Euler', or None).
            last_step_size: Size of the last step.
            **model_kwargs: Additional arguments passed to the model.

        Returns:
            Generated samples tensor.
        """
        z = z.to(device=self.device, dtype=self.dtype)

        if last_step is None:
            last_step_size = 0.0

        t0, t1 = self._check_interval(
            sde=True, last_step_size=last_step_size, diffusion_form=diffusion_form
        )
        # t needs num_steps+1 points to perform num_steps integration steps
        t = torch.linspace(t0, t1, num_steps + 1, device=self.device, dtype=self.dtype)

        drift_fn = self._get_drift()
        score_fn = self._get_score()

        def diffusion_fn(x, t_val):
            return self.interpolant.compute_diffusion(
                x, t_val, form=diffusion_form, norm=diffusion_norm
            )

        def sde_drift(x, t_val, **kwargs):
            diffusion = diffusion_fn(x, t_val)
            return drift_fn(x, t_val, **kwargs) + diffusion * score_fn(
                x, t_val, **kwargs
            )

        def fixed_sde_drift(x, t_val):
            return sde_drift(x, t_val, **model_kwargs)

        integ = self._integrator_for(
            "sde", self._deprecated_call_integrator("sde", method)
        )
        x = integ.integrate(
            state={"x": z},
            step_size=t[1] - t[0],
            n_steps=num_steps,
            drift=fixed_sde_drift,
            diffusion=diffusion_fn,
            t=t,
        )["x"]

        # Apply last step
        if last_step is not None:
            t_final = torch.ones(x.size(0), device=self.device, dtype=self.dtype) * t1
            x = self._apply_last_step(
                x, t_final, sde_drift, last_step, last_step_size, **model_kwargs
            )

        return x

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
            -N / 2.0 * np.log(2 * np.pi)
            - torch.sum(z.square(), dim=tuple(range(1, z.ndim))) / 2.0
        )


__all__ = ["FlowSampler", "PredictionType"]
