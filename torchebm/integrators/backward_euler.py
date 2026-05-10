r"""Backward (implicit) Euler integrator."""

import warnings
from typing import Callable, Dict, Optional

import torch

from torchebm.core import BaseIntegrator


class BackwardEulerIntegrator(BaseIntegrator):
    r"""
    Backward (implicit) Euler integrator for Itô SDEs and ODEs.

    The SDE form is:

    \[
    \mathrm{d}x = f(x,t)\,\mathrm{d}t + \sqrt{2D(x,t)}\,\mathrm{d}W_t
    \]

    Update rule (drift-implicit, diffusion-explicit):

    \[
    x_{n+1} = x_n + h\,f(x_{n+1}, t_{n+1})
              + \sqrt{2\,D(x_n, t_n)}\,\Delta W_n
    \]

    Each step solves the implicit equation
    \(x_{n+1} - h\,f(x_{n+1}, t_{n+1}) = x_n + \text{noise}\) by fixed-point
    iteration starting from an explicit-Euler warm start.  Convergence of
    the iteration requires \(h \cdot L < 1\) where \(L\) is the local
    Lipschitz constant of \(f\); for stiffer problems reduce ``step_size``.

    Backward Euler is L-stable, making it suitable for stiff systems where
    explicit Euler diverges.  When ``diffusion`` is omitted this reduces to
    the implicit Euler method for ODEs.

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
        drift = lambda x, t: -x  # simple mean-reverting drift
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
    ):
        super().__init__(device=device, dtype=dtype)
        self.n_iter = n_iter
        self.tol = tol

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

    @staticmethod
    def _resolve_diffusion(
        diffusion: Optional[torch.Tensor],
        noise_scale: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        r"""Return a diffusion tensor from explicit value or ``noise_scale``."""
        if diffusion is not None:
            return diffusion
        if noise_scale is not None:
            if not torch.is_tensor(noise_scale):
                noise_scale = torch.tensor(noise_scale, device=device, dtype=dtype)
            return noise_scale ** 2
        return None

    def _solve_implicit(
        self,
        x_prev: torch.Tensor,
        step_size: torch.Tensor,
        drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        t_prev: torch.Tensor,
        t_next: torch.Tensor,
        noise_term: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Solve \(x = x_{prev} + h f(x, t_{next}) + \text{noise}\) by fixed-point."""
        x_curr = x_prev + step_size * drift_fn(x_prev, t_prev)
        if noise_term is not None:
            x_curr = x_curr + noise_term
        for _ in range(self.n_iter):
            x_next = x_prev + step_size * drift_fn(x_curr, t_next)
            if noise_term is not None:
                x_next = x_next + noise_term
            delta = (x_next - x_curr).abs().max()
            if delta <= self.tol * (1.0 + x_curr.abs().max()):
                return x_next
            x_curr = x_next
        return x_curr

    def step(
        self,
        state: Dict[str, torch.Tensor],
        step_size: torch.Tensor = None,
        *,
        model=None,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        diffusion: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        noise_scale: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Advance one backward-Euler step with optional SDE noise.

        Args:
            state: Mapping containing ``"x"`` position tensor.
            step_size: Step size for the integration.
            model: Deprecated energy model. If provided and ``drift`` is
                ``None``, uses ``drift(x, t) = -model.gradient(x)``.
            drift: Explicit drift callable ``f(x, t)``.
            diffusion: Diffusion coefficient \(D(x, t)\) tensor (already
                evaluated at the current state).
            noise: Pre-sampled standard-normal noise tensor.  When ``None``,
                generated internally.
            noise_scale: Scalar whose square is used as \(D\) when
                ``diffusion`` is not given.
            t: Current time tensor (batch,).

        Returns:
            Updated state dict ``{"x": x_new}``.
        """
        x = state["x"]
        if not torch.is_tensor(step_size):
            step_size = torch.tensor(step_size, device=x.device, dtype=x.dtype)
        if t is None:
            t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        drift = self._resolve_model_to_drift(model, drift)
        drift_fn = self._resolve_drift(drift)
        diffusion_val = self._resolve_diffusion(
            diffusion, noise_scale, x.device, x.dtype
        )

        noise_term: Optional[torch.Tensor] = None
        if diffusion_val is not None:
            if noise is None:
                noise = torch.randn_like(x, device=self.device, dtype=self.dtype)
            noise_term = torch.sqrt(2.0 * diffusion_val) * noise * torch.sqrt(step_size)

        x_new = self._solve_implicit(
            x, step_size, drift_fn, t, t + step_size, noise_term,
        )
        return {"x": x_new}

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
        diffusion: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        noise_scale: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        adaptive: Optional[bool] = None,
        inference_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        r"""Integrate the state over a time interval (ODE or SDE).

        Args:
            state: Mapping with key ``"x"``.
            step_size: Uniform step size.
            n_steps: Number of integration steps.
            model: Deprecated energy model.
            drift: Explicit drift callable ``f(x, t)``.
            diffusion: Time-dependent diffusion callable ``D(x, t)``.
            noise_scale: Scalar whose square is used as \(D\) when
                ``diffusion`` is not given.
            t: 1-D time grid.  Built from ``step_size`` when ``None``.
            adaptive: Accepted for API consistency with the SDE-RK base.
                Backward Euler has no embedded error pair, so ``True`` is
                rejected; ``False`` and ``None`` use fixed-step integration.
            inference_mode: When ``True``, wraps computation in
                ``torch.inference_mode()`` for faster execution without
                gradient tracking.

        Returns:
            Updated state dict ``{"x": x_final}``.
        """
        if adaptive:
            raise ValueError(
                "BackwardEulerIntegrator does not support adaptive stepping "
                "(no embedded error pair). Pass adaptive=False or omit the kwarg."
            )

        if inference_mode:
            with torch.inference_mode():
                return self.integrate(
                    state, step_size, n_steps,
                    model=model, drift=drift, diffusion=diffusion,
                    noise_scale=noise_scale, t=t,
                )

        if n_steps is None or n_steps <= 0:
            raise ValueError("n_steps must be positive")

        drift = self._resolve_model_to_drift(model, drift)
        drift_fn = self._resolve_drift(drift)

        x = state["x"]
        if not torch.is_tensor(step_size):
            step_size = torch.tensor(step_size, device=x.device, dtype=x.dtype)

        if t is None:
            t_grid = (
                torch.arange(n_steps + 1, device=x.device, dtype=x.dtype)
                * step_size
            )
        else:
            if t.ndim != 1 or t.numel() < 2:
                raise ValueError("t must be a 1D tensor with length >= 2")
            t_grid = t

        has_diffusion_fn = diffusion is not None
        ns_const = self._resolve_diffusion(
            None, noise_scale, x.device, x.dtype
        ) if not has_diffusion_fn else None

        n = t_grid.numel() - 1
        batch_size = x.size(0)
        for i in range(n):
            dt = t_grid[i + 1] - t_grid[i]
            t_prev = t_grid[i].expand(batch_size)
            t_next = t_grid[i + 1].expand(batch_size)
            diff_val = diffusion(x, t_prev) if has_diffusion_fn else ns_const
            if diff_val is not None:
                dw = torch.randn_like(x) * torch.sqrt(dt)
                noise_term = torch.sqrt(2.0 * diff_val) * dw
            else:
                noise_term = None
            x = self._solve_implicit(x, dt, drift_fn, t_prev, t_next, noise_term)
        return {"x": x}
