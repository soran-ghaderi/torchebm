r"""Euler-Maruyama integrator."""

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

    # -- backward-compat shims for deprecated ``model`` kwarg ----------------

    @staticmethod
    def _resolve_model_to_drift(model, drift):
        """Convert deprecated ``model`` to a ``drift`` callable."""
        if model is not None:
            warnings.warn(
                "Passing 'model' to EulerMaruyamaIntegrator is deprecated. "
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
