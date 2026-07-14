r"""Euler-Maruyama and Backward (implicit) Euler-Maruyama integrators."""

import warnings
from typing import Callable, Dict, Optional

import torch

from torchebm.core import BaseSDERungeKuttaIntegrator


class EulerMaruyamaIntegrator(BaseSDERungeKuttaIntegrator):
    r"""
    Euler-Maruyama integrator for Itô SDEs and ODEs.

    The SDE form is:

    \[
    \mathrm{d}x = f(x,t)\,\mathrm{d}t + \sqrt{2D(x,t)}\,\mathrm{d}W_t
    \]

    When `diffusion` is omitted, this reduces to the Euler method for ODEs.

    Update rule:

    \[
    x_{n+1} = x_n + f(x_n, t_n)\Delta t + \sqrt{2D(x_n,t_n)}\,\Delta W_n
    \]

    Args:
        device: Device for computations.
        dtype: Data type for computations.
        atol: Absolute tolerance for adaptive stepping.
        rtol: Relative tolerance for adaptive stepping.
        max_steps: Maximum number of steps before raising ``RuntimeError``.
        safety: Safety factor for step-size adjustment (< 1).
        min_factor: Minimum step-size shrink factor.
        max_factor: Maximum step-size growth factor.
        max_step_size: Maximum absolute step size during adaptive integration.
        norm: Callable ``norm(tensor) -> scalar`` for local error measurement.

    Example:
        ```python
        from torchebm.integrators import EulerMaruyamaIntegrator
        import torch

        integrator = EulerMaruyamaIntegrator()
        state = {"x": torch.randn(100, 2)}
        drift = lambda x, t: -x  # simple mean-reverting drift
        result = integrator.step(
            state, step_size=0.01, drift=drift, noise_scale=1.0
        )
        ```
    """

    @property
    def tableau_a(self):
        return ((),)

    @property
    def tableau_b(self):
        return (1.0,)

    @property
    def tableau_c(self):
        return (0.0,)


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
    seeded with an explicit Euler predictor.  By default it runs for
    ``solver_max_iter`` iterations.  When ``solver_check_every`` is positive,
    it checks the RMS residual and stops once it is below ``solver_tol``.
    Picard converges only when ``step_size`` times the drift Lipschitz
    constant is below 1; for stiffer regimes choose a smaller ``step_size``.

    Args:
        device: Device for computations.
        dtype: Data type for computations.
        solver_max_iter: Total Picard iterations per implicit stage.
        solver_tol: RMS residual threshold for early termination. Only
            consulted when ``solver_check_every > 0``.
        solver_check_every: When positive, check the residual every ``n``
            iterations and exit early once below ``solver_tol``.
        atol: Absolute tolerance for adaptive stepping.
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

    @property
    def tableau_a(self):
        return ((1.0,),)

    @property
    def tableau_b(self):
        return (1.0,)

    @property
    def tableau_c(self):
        return (1.0,)
