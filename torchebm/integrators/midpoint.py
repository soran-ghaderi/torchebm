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

    The Butcher tableau:

    \[
    \begin{array}{c|cc}
    0 \\
    \tfrac{1}{2} & \tfrac{1}{2} \\
    \hline
    & 0 & 1
    \end{array}
    \]

    Args:
        device: Device for computations.
        dtype: Data type for computations.

    Example:
        ```python
        from torchebm.integrators import MidpointIntegrator
        import torch

        integrator = MidpointIntegrator()
        state = {"x": torch.randn(100, 2)}
        drift = lambda x, t: -x
        result = integrator.integrate(
            state, step_size=0.01, n_steps=100, drift=drift,
        )
        ```
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
