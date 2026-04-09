r"""Backward (implicit) Euler integrator."""

import warnings
from typing import Callable, Dict, Optional

import torch

from torchebm.core import BaseSDERungeKuttaIntegrator


class BackwardEulerIntegrator(BaseSDERungeKuttaIntegrator):
    r"""
    Backward (implicit) Euler integrator for Itô SDEs and ODEs.

    Butcher tableau (1-stage diagonally implicit Runge–Kutta):

    \[
    \begin{array}{c|c}
    1 & 1 \\\hline
      & 1
    \end{array}
    \]

    See https://en.wikipedia.org/wiki/Backward_Euler_method.

    The SDE form is:

    \[
    \mathrm{d}x = f(x,t)\,\mathrm{d}t + \sqrt{2D(x,t)}\,\mathrm{d}W_t
    \]

    Update rule (drift-implicit, diffusion-explicit, matching the SDE-RK
    family in this package):

    \[
    x'_{n+1} = x_n + h\,f(x'_{n+1}, t_n + h)
    \quad\text{(implicit solve)}
    \]

    \[
    x_{n+1} = x'_{n+1} + \sqrt{2\,D(x_n, t_n)}\,\Delta W_n
    \]

    The implicit equation is solved by fixed-point iteration with an
    explicit-Euler warm start.  Convergence requires \(h \cdot L < 1\)
    where \(L\) is the local Lipschitz constant of \(f\); reduce
    ``step_size`` for stiffer problems.  Backward Euler is L-stable, so
    the discretization itself remains bounded for any step size — only
    the fixed-point iteration limits the practical step size.

    Args:
        device: Device for computations.
        dtype: Data type for computations.
        n_iter: Maximum fixed-point iterations per step.
        tol: Convergence tolerance — iterate until
            ``max|x^{k+1} - x^k| <= tol * (1 + max|x^k|)``.

    Example:
        ```python
        from torchebm.integrators import BackwardEulerIntegrator
        import torch

        integrator = BackwardEulerIntegrator()
        state = {"x": torch.randn(100, 2)}
        drift = lambda x, t: -x  # mean-reverting drift
        result = integrator.step(
            state, step_size=0.5, drift=drift, noise_scale=1.0
        )
        ```
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        *,
        n_iter: int = 50,
        tol: float = 1e-6,
        **kwargs,
    ):
        super().__init__(device=device, dtype=dtype, **kwargs)
        self.n_iter = n_iter
        self.tol = tol

    @property
    def tableau_a(self):
        return ((1.0,),)

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
        r"""Implicit-Euler deterministic step.

        Solves \(x_{n+1} = x_n + h\,f(x_{n+1}, t_n + h)\) by fixed-point
        iteration with an explicit-Euler warm start.  Replaces the base
        class's explicit RK driver because the diagonal entry
        \(a_{0,0} = 1\) makes the stage equation implicit.
        """
        t_next = t + step_size
        x_curr = x + step_size * drift_fn(x, t)
        for _ in range(self.n_iter):
            x_next = x + step_size * drift_fn(x_curr, t_next)
            delta = (x_next - x_curr).abs().max()
            if delta <= self.tol * (1.0 + x_curr.abs().max()):
                return x_next
            x_curr = x_next
        return x_curr

    # -- backward-compat shim for deprecated ``model`` kwarg ----------------

    @staticmethod
    def _resolve_model_to_drift(model, drift):
        """Convert deprecated ``model`` to a ``drift`` callable."""
        if model is not None:
            warnings.warn(
                "Passing 'model' to BackwardEulerIntegrator is deprecated. "
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
