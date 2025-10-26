"""Stochastic differential equation integrators."""

from typing import Dict, Optional

import torch

from torchebm.core import BaseModel, Integrator


class EulerMaruyamaIntegrator(Integrator):
    r"""
    Euler-Maruyama integrator for overdamped Langevin dynamics.

    Update rule:

    \[
    x_{t+1} = x_t - \eta \nabla_x U(x_t) + \sqrt{2\eta} \sigma \mathcal{N}(0, I)
    \]

    Args:
        step_size: Step size for gradient descent.
        noise_scale: Scale of Gaussian noise injection.
        noise: Optional pre-sampled noise tensor; if not provided, sampled internally.
    """

    def step(
        self,
        state: Dict[str, torch.Tensor],
        model: BaseModel,
        step_size: torch.Tensor,
        noise_scale: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = state["x"]

        grad = model.gradient(x)

        if noise is None:
            noise = torch.randn_like(x, device=self.device, dtype=self.dtype)

        if not torch.is_tensor(step_size):
            step_size = torch.tensor(step_size, device=x.device, dtype=x.dtype)
        if not torch.is_tensor(noise_scale):
            noise_scale = torch.tensor(noise_scale, device=x.device, dtype=x.dtype)

        x_new = (
            x - step_size * grad + torch.sqrt(2.0 * step_size) * (noise_scale * noise)
        )
        return {"x": x_new}

    def integrate(
        self,
        state: Dict[str, torch.Tensor],
        model: BaseModel,
        step_size: torch.Tensor,
        n_steps: int,
        noise_scale: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")

        x = state["x"]
        for _ in range(n_steps):
            state = self.step(
                state={"x": x},
                model=model,
                step_size=step_size,
                noise_scale=noise_scale,
            )
            x = state["x"]
        return {"x": x}
