from typing import Callable, Dict, Optional

import torch

from torchebm.core import BaseIntegrator, BaseModel
from torchebm.integrators import _integrate_time_grid


class HeunIntegrator(BaseIntegrator):
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
            state, model=None, step_size=0.02, n_steps=50, drift=drift, t=t
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
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        noise_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = state["x"]
        if not torch.is_tensor(step_size):
            step_size = torch.tensor(step_size, device=x.device, dtype=x.dtype)

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

        # Heun predictor-corrector for drift term
        k1 = drift(x, t)
        x_pred = x + step_size * k1
        k2 = drift(x_pred, t + step_size)
        x_new = x + 0.5 * step_size * (k1 + k2)

        # Add stochastic term after deterministic update
        if diffusion is not None:
            if noise is None:
                noise = torch.randn_like(x, device=self.device, dtype=self.dtype)
            dw = noise * torch.sqrt(step_size)
            x_new = x_new + torch.sqrt(2.0 * diffusion) * dw

        return {"x": x_new}


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
        t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
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
                t=t_batch,
                noise_scale=noise_scale,
            )["x"]

        return {"x": _integrate_time_grid(x0, t, _step_fn)}
