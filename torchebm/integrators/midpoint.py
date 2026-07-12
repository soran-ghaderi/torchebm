r"""Explicit midpoint second-order Runge-Kutta integrator."""

from typing import Tuple

from torchebm.core import BaseRungeKuttaIntegrator


class MidpointIntegrator(BaseRungeKuttaIntegrator):
    r"""Explicit midpoint (RK2) integrator.

    This fixed-step, two-stage method evaluates the drift at the midpoint
    predicted by an Euler half-step:

    \[
    k_1 = f(x_n, t_n), \qquad
    k_2 = f\!\left(x_n + \tfrac{h}{2}k_1, t_n + \tfrac{h}{2}\right)
    \]

    \[
    x_{n+1} = x_n + h k_2
    \]

    Args:
        device: Device for computations.
        dtype: Data type for computations.
    """

    @property
    def tableau_a(self) -> Tuple[Tuple[float, ...], ...]:
        return ((), (0.5,))

    @property
    def tableau_b(self) -> Tuple[float, ...]:
        return (0.0, 1.0)

    @property
    def tableau_c(self) -> Tuple[float, ...]:
        return (0.0, 0.5)
