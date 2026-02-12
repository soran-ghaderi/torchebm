r"""Heun (improved Euler) integrator."""

import warnings
from typing import Callable, Dict, Optional

import torch

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

    # -- backward-compat shims for deprecated ``model`` kwarg ----------------

    @staticmethod
    def _resolve_model_to_drift(model, drift):
        """Convert deprecated ``model`` to a ``drift`` callable."""
        if model is not None:
            warnings.warn(
                "Passing 'model' to HeunIntegrator is deprecated. "
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
