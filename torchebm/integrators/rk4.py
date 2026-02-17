r"""Classical 4th-order Runge-Kutta integrator (fixed step)."""

from typing import Tuple

from torchebm.core import BaseRungeKuttaIntegrator


class RK4Integrator(BaseRungeKuttaIntegrator):
    r"""Classical 4th-order Runge-Kutta integrator.

    A 4-stage, 4th-order explicit method. Fixed-step only (no embedded
    error pair).

    The update rule computes four stages and combines them:

    \[
    k_1 = f(x_n,\, t_n)
    \]

    \[
    k_2 = f\!\bigl(x_n + \tfrac{h}{2}\,k_1,\; t_n + \tfrac{h}{2}\bigr)
    \]

    \[
    k_3 = f\!\bigl(x_n + \tfrac{h}{2}\,k_2,\; t_n + \tfrac{h}{2}\bigr)
    \]

    \[
    k_4 = f(x_n + h\,k_3,\; t_n + h)
    \]

    \[
    x_{n+1} = x_n + \tfrac{h}{6}\bigl(k_1 + 2k_2 + 2k_3 + k_4\bigr)
    \]

    The Butcher tableau:

    \[
    \begin{array}{c|cccc}
    0 \\
    \tfrac{1}{2} & \tfrac{1}{2} \\
    \tfrac{1}{2} & 0 & \tfrac{1}{2} \\
    1 & 0 & 0 & 1 \\
    \hline
    & \tfrac{1}{6} & \tfrac{1}{3} & \tfrac{1}{3} & \tfrac{1}{6}
    \end{array}
    \]

    Reference:
        Kutta, W. (1901). Beitrag zur naherungsweisen Integration
        totaler Differentialgleichungen. Z. Math. Phys., 46, 435--453.

    Args:
        device: Device for computations.
        dtype: Data type for computations.

    Example:
        ```python
        from torchebm.integrators import RK4Integrator
        import torch

        integrator = RK4Integrator()
        state = {"x": torch.randn(100, 2)}
        drift = lambda x, t: -x
        result = integrator.integrate(
            state, step_size=0.01, n_steps=100, drift=drift,
        )
        ```
    """

    @property
    def tableau_a(self) -> Tuple[Tuple[float, ...], ...]:
        return (
            (),
            (1 / 2,),
            (0.0, 1 / 2),
            (0.0, 0.0, 1.0),
        )

    @property
    def tableau_b(self) -> Tuple[float, ...]:
        return (1 / 6, 1 / 3, 1 / 3, 1 / 6)

    @property
    def tableau_c(self) -> Tuple[float, ...]:
        return (0.0, 1 / 2, 1 / 2, 1.0)
