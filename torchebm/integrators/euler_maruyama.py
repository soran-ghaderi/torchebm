r"""Euler-Maruyama integrator."""

from typing import Callable, Dict, Optional

import torch

from torchebm.core import BaseIntegrator, BaseModel
from torchebm.integrators import _integrate_time_grid

# def _integrate_time_grid(
#     x: torch.Tensor,
#     t: torch.Tensor,
#     step_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
# ) -> torch.Tensor:
#     if t.ndim != 1:
#         raise ValueError("t must be a 1D tensor")
#     if t.numel() < 2:
#         raise ValueError("t must have length >= 2")
#     for i in range(t.numel() - 1):
#         dt = t[i + 1] - t[i]
#         t_batch = t[i].expand(x.size(0))
#         x = step_fn(x, t_batch, dt)
#     return x


class EulerMaruyamaIntegrator(BaseIntegrator):
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
            state, model=None, step_size=0.01, drift=drift, noise_scale=1.0
        )
        ```
    """

    def step(
        self,
        state: Dict[str, torch.Tensor],
        model: Optional[BaseModel],
        step_size: torch.Tensor,
        *,
        drift: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        diffusion: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        noise_scale: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = state["x"]
        if not torch.is_tensor(step_size):
            step_size = torch.tensor(step_size, device=x.device, dtype=x.dtype)

        if t is None:
            t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        if drift is None:
            if model is None:
                raise ValueError(
                    "Either `model` must be provided or `drift` must be set."
                )
            drift = lambda x_, t_: -model.gradient(x_)

        if diffusion is None and noise_scale is not None:
            if not torch.is_tensor(noise_scale):
                noise_scale = torch.tensor(noise_scale, device=x.device, dtype=x.dtype)
            diffusion = noise_scale**2

        drift_term = drift(x, t) * step_size

        if diffusion is None:
            return {"x": x + drift_term}

        if noise is None:
            noise = torch.randn_like(x, device=self.device, dtype=self.dtype)

        dw = noise * torch.sqrt(step_size)
        return {"x": x + drift_term + torch.sqrt(2.0 * diffusion) * dw}

    def integrate(
        self,
        state: Dict[str, torch.Tensor],
        model: Optional[BaseModel],
        step_size: torch.Tensor,
        n_steps: int,
        *,
        drift: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        diffusion: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        noise_scale: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if t is None:
            if not torch.is_tensor(step_size):
                step_size = torch.tensor(
                    step_size, device=state["x"].device, dtype=state["x"].dtype
                )
            t = (
                torch.arange(n_steps, device=state["x"].device, dtype=state["x"].dtype)
                * step_size
            )
        if t.ndim != 1 or t.numel() != n_steps:
            raise ValueError("t must be a 1D tensor with length n_steps")

        x0 = state["x"]

        def _step_fn(x, t_batch, dt):
            diffusion_t = diffusion(x, t_batch) if diffusion is not None else None
            return self.step(
                state={"x": x},
                model=model,
                step_size=dt,
                drift=drift,
                diffusion=diffusion_t,
                noise_scale=noise_scale,
                t=t_batch,
            )["x"]

        return {"x": _integrate_time_grid(x0, t, _step_fn)}




# __all__ = [
#     "EulerMaruyamaIntegrator",
#     "HeunIntegrator",
# ]
