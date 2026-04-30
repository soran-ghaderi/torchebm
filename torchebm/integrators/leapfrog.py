r"""Symplectic leapfrog (Störmer-Verlet) integrator for Hamiltonian dynamics."""

import warnings
from typing import Callable, Dict, Optional, Union

import torch

from torchebm.core import BaseModel, BaseIntegrator


class LeapfrogIntegrator(BaseIntegrator):
    r"""
    Symplectic leapfrog (Störmer–Verlet) integrator for Hamiltonian dynamics.

    Update rule:

    \[
    p_{t+1/2} = p_t - \frac{\epsilon}{2} \nabla_x U(x_t)
    \]

    \[
    x_{t+1} = x_t + \epsilon p_{t+1/2}
    \]

    \[
    p_{t+1} = p_{t+1/2} - \frac{\epsilon}{2} \nabla_x U(x_{t+1})
    \]

    Args:
        device: Device for computations.
        dtype: Data type for computations.

    Example:
        ```python
        from torchebm.integrators import LeapfrogIntegrator
        import torch

        energy_fn = ...  # an energy model with .gradient()
        integrator = LeapfrogIntegrator()
        state = {"x": torch.randn(100, 2), "p": torch.randn(100, 2)}
        drift = lambda x, t: -energy_fn.gradient(x)
        result = integrator.integrate(state, step_size=0.01, n_steps=10, drift=drift)
        ```
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self._mass_view_ndim: Optional[int] = None
        self._mass_view_shape: Optional[tuple] = None

    def _broadcast_mass(self, mass: torch.Tensor, ndim: int) -> torch.Tensor:
        r"""Reshape ``mass`` for broadcasting against an ``ndim``-D state tensor."""
        if self._mass_view_ndim != ndim:
            self._mass_view_shape = (1,) * (ndim - 1) + (-1,)
            self._mass_view_ndim = ndim
        return mass.view(self._mass_view_shape)

    @staticmethod
    def _resolve_deprecated_to_drift(model, potential_grad, drift):
        r"""Convert deprecated `model` or `potential_grad` to a `drift` callable."""
        if model is not None:
            warnings.warn(
                "Passing 'model' to LeapfrogIntegrator is deprecated. "
                "Use drift=lambda x, t: -model.gradient(x) instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if drift is None:
                drift = lambda x_, t_: -model.gradient(x_)
        if potential_grad is not None:
            warnings.warn(
                "Passing 'potential_grad' to LeapfrogIntegrator is deprecated. "
                "Use drift=lambda x, t: -potential_grad(x) instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if drift is None:
                drift = lambda x_, t_: -potential_grad(x_)
        return drift

    def step(
        self,
        state: Dict[str, torch.Tensor],
        model: Optional[BaseModel] = None,
        step_size: torch.Tensor = None,
        mass: Optional[Union[float, torch.Tensor]] = None,
        *,
        potential_grad: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        safe: bool = False,
    ) -> Dict[str, torch.Tensor]:
        r"""Advance one leapfrog step.

        Args:
            state: Current Hamiltonian state with keys `"x"` and `"p"`.
            model: Deprecated energy model. If provided and `drift` is `None`,
                uses `drift(x, t) = -model.gradient(x)`.
            step_size: Integration step size.
            mass: Optional mass term. Can be a scalar float or tensor.
            potential_grad: Deprecated callable for `\nabla_x U(x)`. If provided
                and `drift` is `None`, uses `drift(x, t) = -potential_grad(x)`.
            drift: Drift/force callable with signature `(x, t) -> force`.
            safe: If `True`, clamps force magnitudes and replaces NaNs by zeros.

        Returns:
            Updated state dictionary with keys `"x"` and `"p"`.
        """
        x = state["x"]
        p = state["p"]

        if not torch.is_tensor(step_size):
            step_size = torch.tensor(step_size, device=x.device, dtype=x.dtype)
        drift = self._resolve_deprecated_to_drift(model, potential_grad, drift)
        drift_fn = self._resolve_drift(drift)

        t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        force = drift_fn(x, t)
        if safe:
            force.clamp_(min=-1e6, max=1e6)

        p_half = p + 0.5 * step_size * force

        if mass is None:
            x_new = x + step_size * p_half
        else:
            if isinstance(mass, float):
                safe_mass = max(mass, 1e-10)
                x_new = x + step_size * p_half / safe_mass
            else:
                safe_mass = torch.clamp(mass, min=1e-10)
                x_new = x + step_size * p_half / self._broadcast_mass(
                    safe_mass, x.ndim
                )

        force_new = drift_fn(x_new, t)
        if safe:
            force_new.clamp_(min=-1e6, max=1e6)
        p_new = p_half + 0.5 * step_size * force_new

        if safe:
            # Unconditional `nan_to_num_` on freshly owned tensors avoids the
            # CPU sync that `if isnan(x).any() or isnan(p).any()` would force
            # every step (Python `or` materialises both 0-d bools to host).
            x_new.nan_to_num_(nan=0.0)
            p_new.nan_to_num_(nan=0.0)
        return {"x": x_new, "p": p_new}

    def integrate(
        self,
        state: Dict[str, torch.Tensor],
        model: Optional[BaseModel] = None,
        step_size: torch.Tensor = None,
        n_steps: int = None,
        mass: Optional[Union[float, torch.Tensor]] = None,
        *,
        potential_grad: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        safe: bool = False,
        inference_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        r"""Integrate Hamiltonian dynamics for multiple leapfrog steps.

        Args:
            state: Initial Hamiltonian state with keys `"x"` and `"p"`.
            model: Deprecated energy model. If provided and `drift` is `None`,
                uses `drift(x, t) = -model.gradient(x)`.
            step_size: Integration step size.
            n_steps: Number of leapfrog steps to apply. Must be positive.
            mass: Optional mass term. Can be a scalar float or tensor.
            potential_grad: Deprecated callable for `\nabla_x U(x)`. If provided
                and `drift` is `None`, uses `drift(x, t) = -potential_grad(x)`.
            drift: Drift/force callable with signature `(x, t) -> force`.
            safe: If `True`, clamps force magnitudes and replaces NaNs by zeros.
            inference_mode: If `True`, runs integration under
                `torch.inference_mode()`.

        Returns:
            Final state dictionary with keys `"x"` and `"p"`.

        Raises:
            ValueError: If `n_steps <= 0`.
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")

        if inference_mode:
            with torch.inference_mode():
                return self.integrate(
                    state, model=model, step_size=step_size,
                    n_steps=n_steps, mass=mass,
                    potential_grad=potential_grad, drift=drift, safe=safe,
                )

        drift = self._resolve_deprecated_to_drift(model, potential_grad, drift)
        drift_fn = self._resolve_drift(drift)

        x = state["x"]
        p = state["p"]

        if not torch.is_tensor(step_size):
            step_size = torch.tensor(step_size, device=x.device, dtype=x.dtype)

        t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        for _ in range(n_steps):
            force = drift_fn(x, t)
            if safe:
                force.clamp_(min=-1e6, max=1e6)

            p_half = p + 0.5 * step_size * force

            if mass is None:
                x = x + step_size * p_half
            else:
                if isinstance(mass, float):
                    safe_mass = max(mass, 1e-10)
                    x = x + step_size * p_half / safe_mass
                else:
                    safe_mass = torch.clamp(mass, min=1e-10)
                    x = x + step_size * p_half / self._broadcast_mass(
                        safe_mass, x.ndim
                    )

            force_new = drift_fn(x, t)
            if safe:
                force_new.clamp_(min=-1e6, max=1e6)
            p = p_half + 0.5 * step_size * force_new

            if safe:
                # Unconditional in-place sanitization — idempotent on clean
                # tensors and avoids the per-substep CPU sync caused by
                # ``isnan(x).any() or isnan(p).any()``.
                x.nan_to_num_(nan=0.0)
                p.nan_to_num_(nan=0.0)

        return {"x": x, "p": p}
