r"""Heun (improved Euler) integrator."""

from torchebm.core import BaseSDERungeKuttaIntegrator


class HeunIntegrator(BaseSDERungeKuttaIntegrator):
    r"""
    Heun integrator (predictor-corrector) for Itô SDEs and ODEs.

    A second-order method that uses a predictor step followed by a corrector:

    \[
    \mathrm{d}x = f(x,t)\,\mathrm{d}t + \sqrt{2D(x,t)}\,\mathrm{d}W_t
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
        from torchebm.integrators import HeunIntegrator
        import torch

        integrator = HeunIntegrator()
        state = {"x": torch.randn(100, 2)}
        t = torch.linspace(0, 1, 50)
        drift = lambda x, t: -x
        result = integrator.integrate(
            state, step_size=0.02, n_steps=50, drift=drift, t=t
        )
        ```
    """

    @property
    def tableau_a(self):
        return ((), (1.0,))

    @property
    def tableau_b(self):
        return (0.5, 0.5)

    @property
    def tableau_c(self):
        return (0.0, 1.0)
