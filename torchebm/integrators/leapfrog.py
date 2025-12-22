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
        from torchebm.core import DoubleWellEnergy
        import torch

        energy = DoubleWellEnergy()
        integrator = LeapfrogIntegrator()
        state = {"x": torch.randn(100, 2), "p": torch.randn(100, 2)}
        result = integrator.integrate(state, energy, step_size=0.01, n_steps=10)
        ```
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)

    def step(
        self,
        state: Dict[str, torch.Tensor],
        model: Optional[BaseModel],
        step_size: torch.Tensor,
        mass: Optional[Union[float, torch.Tensor]] = None,
        *,
        potential_grad: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        x = state["x"]
        p = state["p"]

        if not torch.is_tensor(step_size):
            step_size = torch.tensor(step_size, device=x.device, dtype=x.dtype)

        if potential_grad is None:
            if model is None:
                raise ValueError(
                    "Either `model` must be provided or `potential_grad` must be set."
                )
            potential_grad = model.gradient

        grad = potential_grad(x)
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
        grad_new = potential_grad(x_new)
        grad_new = torch.clamp(grad_new, min=-1e6, max=1e6)
        p_new = p_half - 0.5 * step_size * grad_new

        # handling NaNs
        if torch.isnan(x_new).any() or torch.isnan(p_new).any():
            x_new = torch.nan_to_num(x_new, nan=0.0)
            p_new = torch.nan_to_num(p_new, nan=0.0)
        return {"x": x_new, "p": p_new}

    def integrate(
        self,
        state: Dict[str, torch.Tensor],
        model: Optional[BaseModel],
        step_size: torch.Tensor,
        n_steps: int,
        mass: Optional[Union[float, torch.Tensor]] = None,
        *,
        potential_grad: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")

        x = state["x"]
        p = state["p"]

        for _ in range(n_steps):
            state = self.step(
                state={"x": x, "p": p},
                model=model,
                step_size=step_size,
                mass=mass,
                potential_grad=potential_grad,
            )
            x, p = state["x"], state["p"]

        return {"x": x, "p": p}
