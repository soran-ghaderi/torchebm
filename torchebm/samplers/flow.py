r"""Flow-based sampler for trained generative models.

Supports both ODE (probability flow) and SDE (diffusion) sampling modes
with various numerical integration methods.
"""

from typing import Callable, Literal, Optional, Tuple, Union
import enum

import torch
from torch import nn
import numpy as np

from torchebm.core import BaseSampler
from torchebm.integrators import (
    EulerMaruyamaIntegrator,
    HeunIntegrator,
)
from torchebm.interpolants import (
    BaseInterpolant,
    LinearInterpolant,
    CosineInterpolant,
    VariancePreservingInterpolant,
    expand_t_like_x,
)
from torchebm.losses.loss_utils import get_interpolant

try:
    from torchdiffeq import odeint

    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


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
        sample_eps: Epsilon for sampling time interval.
        dtype: Data type for computations.
        device: Device for computations.
        use_mixed_precision: Whether to use mixed precision.

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
        )
        z = torch.randn(100, 2)
        samples = sampler.sample_ode(z, num_steps=50, method="euler")
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        interpolant: Union[str, BaseInterpolant] = "linear",
        prediction: Literal["velocity", "score", "noise"] = "velocity",
        train_eps: float = 0.0,
        sample_eps: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        use_mixed_precision: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            dtype=dtype,
            device=device,
            use_mixed_precision=use_mixed_precision,
        )
        self.train_eps = train_eps
        self.sample_eps = sample_eps

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

        self.interpolant = BaseSampler.safe_to(
            self.interpolant, device=self.device, dtype=self.dtype
        )

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
        *,
        mode: Literal["ode", "sde"] = "ode",
        shape: Optional[Tuple[int, ...]] = None,
        ode_method: str = "dopri5",
        atol: float = 1e-6,
        rtol: float = 1e-3,
        reverse: bool = False,
        sde_method: str = "euler",
        diffusion_form: str = "SBDM",
        diffusion_norm: float = 1.0,
        last_step: Optional[str] = "Mean",
        last_step_size: float = 0.04,
        **model_kwargs,
    ) -> torch.Tensor:
        r"""
        Unified sampling entrypoint for flow/diffusion models.

        This method exists for API compatibility with `BaseSampler`. For full control,
        prefer calling `sample_ode` or `sample_sde` directly.
        """
        if thin != 1:
            raise ValueError("thin is not supported for FlowSampler")
        if return_trajectory or return_diagnostics:
            raise ValueError("FlowSampler does not support trajectories/diagnostics")

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
        method: str = "dopri5",
        atol: float = 1e-6,
        rtol: float = 1e-3,
        reverse: bool = False,
        **model_kwargs,
    ) -> torch.Tensor:
        r"""
        Sample using probability flow ODE.

        Args:
            z: Initial noise tensor of shape (batch_size, ...).
            num_steps: Number of discretization steps (for fixed-step methods).
            method: ODE solver ('euler', 'heun', 'dopri5', 'dopri8').
            atol: Absolute tolerance for adaptive solvers.
            rtol: Relative tolerance for adaptive solvers.
            reverse: If True, sample from data to noise.
            **model_kwargs: Additional arguments passed to the model.

        Returns:
            Generated samples tensor.
        """
        z = z.to(device=self.device, dtype=self.dtype)
        drift_fn = self._get_drift()

        t0, t1 = self._check_interval(sde=False, reverse=reverse)
        t = torch.linspace(t0, t1, num_steps, device=self.device, dtype=self.dtype)

        if reverse:

            def wrapped_drift(x, t_val, **kwargs):
                return drift_fn(x, torch.ones_like(t_val) * (1 - t_val), **kwargs)

        else:
            wrapped_drift = drift_fn

        def ode_fn(t_val, x):
            t_batch = (
                torch.ones(x.size(0), device=self.device, dtype=self.dtype) * t_val
            )
            return wrapped_drift(x, t_batch, **model_kwargs)

        def fixed_step_drift(x, t_batch):
            return wrapped_drift(x, t_batch, **model_kwargs)

        if method in ["dopri5", "dopri8", "bosh3", "adaptive_heun"]:
            if not HAS_TORCHDIFFEQ:
                raise ImportError("torchdiffeq required for adaptive solvers")
            samples = odeint(ode_fn, z, t, method=method, atol=atol, rtol=rtol)
            return samples[-1]
        if method == "euler":
            integrator = EulerMaruyamaIntegrator(device=self.device, dtype=self.dtype)
            return integrator.integrate(
                state={"x": z},
                model=None,
                step_size=t[1] - t[0],
                n_steps=num_steps,
                drift=fixed_step_drift,
                t=t,
            )["x"]
        if method == "heun":
            integrator = HeunIntegrator(device=self.device, dtype=self.dtype)
            return integrator.integrate(
                state={"x": z},
                model=None,
                step_size=t[1] - t[0],
                n_steps=num_steps,
                drift=fixed_step_drift,
                t=t,
            )["x"]
        raise ValueError(f"Unknown ODE method: {method}")

    @torch.no_grad()
    def sample_sde(
        self,
        z: torch.Tensor,
        num_steps: int = 250,
        method: str = "euler",
        diffusion_form: str = "SBDM",
        diffusion_norm: float = 1.0,
        last_step: Optional[str] = "Mean",
        last_step_size: float = 0.04,
        **model_kwargs,
    ) -> torch.Tensor:
        r"""
        Sample using reverse-time SDE.

        Args:
            z: Initial noise tensor of shape (batch_size, ...).
            num_steps: Number of discretization steps.
            method: SDE solver ('euler', 'heun').
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
        t = torch.linspace(t0, t1, num_steps, device=self.device, dtype=self.dtype)

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

        if method == "euler":
            integrator = EulerMaruyamaIntegrator(device=self.device, dtype=self.dtype)
            x = integrator.integrate(
                state={"x": z},
                model=None,
                step_size=t[1] - t[0],
                n_steps=num_steps,
                drift=fixed_sde_drift,
                diffusion=diffusion_fn,
                t=t,
            )["x"]
        elif method == "heun":
            integrator = HeunIntegrator(device=self.device, dtype=self.dtype)
            x = integrator.integrate(
                state={"x": z},
                model=None,
                step_size=t[1] - t[0],
                n_steps=num_steps,
                drift=fixed_sde_drift,
                diffusion=diffusion_fn,
                t=t,
            )["x"]
        else:
            raise ValueError(f"Unknown SDE method: {method}")

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
            return x / alpha + (sigma**2) / alpha * score
        else:
            return x

    def prior_logp(self, z: torch.Tensor) -> torch.Tensor:
        r"""Compute log probability under standard Gaussian prior."""
        shape = torch.tensor(z.size())
        N = torch.prod(shape[1:])
        return (
            -N / 2.0 * np.log(2 * np.pi)
            - torch.sum(z**2, dim=tuple(range(1, z.ndim))) / 2.0
        )


__all__ = ["FlowSampler", "PredictionType"]
