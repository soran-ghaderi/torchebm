r"""Backward (implicit) Euler-Maruyama integrator."""

import warnings
from typing import Callable, Dict, Optional

import torch

from torchebm.core import BaseSDERungeKuttaIntegrator


class BackwardEulerMaruyamaIntegrator(BaseSDERungeKuttaIntegrator):
    r"""
    Backward (implicit) Euler-Maruyama integrator for Itô SDEs and ODEs.

    The SDE form is:

    \[
    \mathrm{d}x = f(x,t)\,\mathrm{d}t + \sqrt{2D(x,t)}\,\mathrm{d}W_t
    \]

    When `diffusion` is omitted, this reduces to the backward Euler
    method for ODEs.

    Update rule (implicit in the drift, explicit in the diffusion):

    \[
    x_{n+1} = x_n + f(x_{n+1}, t_n + \Delta t)\,\Delta t
              + \sqrt{2D(x_n, t_n)}\,\Delta W_n
    \]

    The implicit equation is solved by fixed-point (Picard) iteration,
    seeded with an explicit Euler predictor.  Iteration stops when the
    max-abs change falls below ``atol`` or after ``max_iter`` iterations.
    Picard converges only when ``step_size`` times the drift Lipschitz
    constant is below 1; for stiffer regimes choose a smaller ``step_size``.

    Args:
        device: Device for computations.
        dtype: Data type for computations.
        max_iter: Maximum fixed-point iterations per step.
        atol: Absolute tolerance for the fixed-point solve and adaptive stepping.
        rtol: Relative tolerance for adaptive stepping.
        max_steps: Maximum number of steps before raising ``RuntimeError``.
        safety: Safety factor for step-size adjustment (< 1).
        min_factor: Minimum step-size shrink factor.
        max_factor: Maximum step-size growth factor.
        max_step_size: Maximum absolute step size during adaptive integration.
        norm: Callable ``norm(tensor) -> scalar`` for local error measurement.

    Example:
        ```python
        from torchebm.integrators import BackwardEulerMaruyamaIntegrator
        import torch

        integrator = BackwardEulerMaruyamaIntegrator()
        state = {"x": torch.randn(100, 2)}
        drift = lambda x, t: -x  # stiff mean-reverting drift
        result = integrator.step(
            state, step_size=0.1, drift=drift, noise_scale=1.0
        )
        ```
    """

    def __init__(self, *args, max_iter: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iter = max_iter

    @property
    def tableau_a(self):
        return ((),)

    @property
    def tableau_b(self):
        return (1.0,)

    @property
    def tableau_c(self):
        return (1.0,)

    def _deterministic_step(
        self,
        x: torch.Tensor,
        step_size: torch.Tensor,
        drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        t: torch.Tensor,
    ) -> torch.Tensor:
        r"""Solve ``x_new = x + h * f(x_new, t + h)`` by fixed-point iteration."""
        t_next = t + step_size
        x_new = x + step_size * drift_fn(x, t)
        for _ in range(self.max_iter):
            x_prev = x_new
            x_new = x + step_size * drift_fn(x_new, t_next)
            if torch.max(torch.abs(x_new - x_prev)).item() < self.atol:
                break
        return x_new

    # -- backward-compat shims for deprecated ``model`` kwarg ----------------

    @staticmethod
    def _resolve_model_to_drift(model, drift):
        """Convert deprecated ``model`` to a ``drift`` callable."""
        if model is not None:
            warnings.warn(
                "Passing 'model' to BackwardEulerMaruyamaIntegrator is deprecated. "
                "Use drift=lambda x, t: -model.gradient(x) instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if drift is None:
                drift = lambda x_, t_: -model.gradient(x_)
        return drift

    def step(
        self,
        state: Dict[str, torch.Tensor],
        step_size: torch.Tensor = None,
        *,
        model=None,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        drift = self._resolve_model_to_drift(model, drift)
        return super().step(state, step_size, drift=drift, **kwargs)

    def integrate(
        self,
        state: Dict[str, torch.Tensor],
        step_size: torch.Tensor = None,
        n_steps: int = None,
        *,
        model=None,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        drift = self._resolve_model_to_drift(model, drift)
        return super().integrate(state, step_size, n_steps, drift=drift, **kwargs)
