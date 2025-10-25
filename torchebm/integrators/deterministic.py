"""Deterministic ordinary differential equation integrators."""

from typing import Dict, Optional, Union

import torch

from torchebm.core import BaseModel, Integrator


class LeapfrogIntegrator(Integrator):
    r"""
    Symplectic leapfrog (Störmer–Verlet) integrator for Hamiltonian dynamics.

    Performs `n_steps` leapfrog steps per call.

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
        n_steps: Default number of leapfrog steps per call.
        device: Device for computations.
        dtype: Data type for computations.
    """

    def __init__(
        self,
        n_steps: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        self.n_steps = int(n_steps)

    def step(
        self,
        state: Dict[str, torch.Tensor],
        model: BaseModel,
        step_size: torch.Tensor,
        n_steps: int,
        mass: Optional[Union[float, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        x = state["x"]
        p = state["p"]

        if not torch.is_tensor(step_size):
            step_size = torch.tensor(step_size, device=x.device, dtype=x.dtype)

        for _ in range(n_steps):
            grad = model.gradient(x)
            grad = torch.clamp(grad, min=-1e6, max=1e6)

            # half-step momentum
            p_half = p - 0.5 * step_size * grad

            # full-step position update
            if mass is None:
                x_new = x + step_size * p_half
            else:
                if isinstance(mass, float):
                    safe_mass = max(mass, 1e-10)
                    x_new = x + step_size * p_half / safe_mass
                else:
                    safe_mass = torch.clamp(mass, min=1e-10)
                    x_new = x + step_size * p_half / safe_mass.view(
                        *([1] * (len(x.shape) - 1)), -1
                    )

            # half-step momentum update at new position
            grad_new = model.gradient(x_new)
            grad_new = torch.clamp(grad_new, min=-1e6, max=1e6)
            p_new = p_half - 0.5 * step_size * grad_new

            x, p = x_new, p_new

        # handling NaNs
        if torch.isnan(x).any() or torch.isnan(p).any():
            x = torch.nan_to_num(x, nan=0.0)
            p = torch.nan_to_num(p, nan=0.0)

        return {"x": x, "p": p}
