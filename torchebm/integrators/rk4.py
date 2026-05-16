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

class RK438Integrator(BaseRungeKuttaIntegrator):
    r"""Kutta's 3/8-rule 4th-order Runge-Kutta integrator.

    A 4-stage, 4th-order explicit method, presented by Kutta (1901) as an
    alternative to the classical RK4.  Like the classical RK4 it is
    fixed-step only (no embedded error pair), but its coefficients
    distribute the stage evaluations differently — the nodes are at
    \(0, \tfrac{1}{3}, \tfrac{2}{3}, 1\) instead of \(0, \tfrac{1}{2},
    \tfrac{1}{2}, 1\).  In exchange for a slightly larger truncation-error
    constant, the method exhibits smaller error coefficients on certain
    stiff problems and is sometimes preferred for its symmetry.

    The update rule computes four stages and combines them:

    \[
    k_1 = f(x_n,\, t_n)
    \]

    \[
    k_2 = f\!\bigl(x_n + \tfrac{h}{3}\,k_1,\; t_n + \tfrac{h}{3}\bigr)
    \]

    \[
    k_3 = f\!\bigl(x_n - \tfrac{h}{3}\,k_1 + h\,k_2,\;
                    t_n + \tfrac{2h}{3}\bigr)
    \]

    \[
    k_4 = f\!\bigl(x_n + h\,k_1 - h\,k_2 + h\,k_3,\; t_n + h\bigr)
    \]

    \[
    x_{n+1} = x_n + \tfrac{h}{8}\bigl(k_1 + 3k_2 + 3k_3 + k_4\bigr)
    \]

    The Butcher tableau:

    \[
    \begin{array}{c|cccc}
    0 \\
    \tfrac{1}{3} & \tfrac{1}{3} \\
    \tfrac{2}{3} & -\tfrac{1}{3} & 1 \\
    1 & 1 & -1 & 1 \\
    \hline
    & \tfrac{1}{8} & \tfrac{3}{8} & \tfrac{3}{8} & \tfrac{1}{8}
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
        from torchebm.integrators import RK438Integrator
        import torch

        integrator = RK438Integrator()
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
            (1 / 3,),
            (-1 / 3, 1.0),
            (1.0, -1.0, 1.0),
        )

    @property
    def tableau_b(self) -> Tuple[float, ...]:
        return (1 / 8, 3 / 8, 3 / 8, 1 / 8)

    @property
    def tableau_c(self) -> Tuple[float, ...]:
        return (0.0, 1 / 3, 2 / 3, 1.0)