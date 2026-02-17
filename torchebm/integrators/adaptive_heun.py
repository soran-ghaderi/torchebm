r"""Adaptive Heun (Heun-Euler 2(1)) integrator with embedded error estimate."""

from typing import Tuple

from torchebm.core import BaseRungeKuttaIntegrator


class AdaptiveHeunIntegrator(BaseRungeKuttaIntegrator):
    r"""Adaptive Heun (Heun-Euler) 2(1) explicit Runge-Kutta integrator.

    A 2-stage, 2nd-order method with an embedded 1st-order (Euler) solution
    for local error estimation. When `adaptive=True` (the default for
    `integrate()` since `error_weights` is defined), the step size is
    adjusted automatically.

    Fixed-step usage is available through `step()` (always fixed) or by
    passing `adaptive=False` to `integrate()`.

    The update rule (predictor-corrector form):

    \[
    k_1 = f(x_n,\, t_n)
    \]

    \[
    k_2 = f(x_n + h\,k_1,\; t_n + h)
    \]

    \[
    x_{n+1} = x_n + \tfrac{h}{2}\bigl(k_1 + k_2\bigr)
    \]

    The Butcher tableau:

    \[
    \begin{array}{c|cc}
    0 \\
    1 & 1 \\
    \hline
    & \tfrac{1}{2} & \tfrac{1}{2}
    \end{array}
    \]

    The embedded 1st-order solution is the Euler method with weights
    \((1, 0)\). The error estimate is computed directly as:

    \[
    e = h \sum_i \hat{e}_i\,k_i, \qquad
    \hat{e}_i = b_i - \hat{b}_i = \bigl(\tfrac{1}{2} - 1,\;
    \tfrac{1}{2} - 0\bigr) = \bigl(-\tfrac{1}{2},\;\tfrac{1}{2}\bigr)
    \]

    Also known as the "improved Euler method", "modified Euler method",
    or "explicit trapezoidal rule". Should not be confused with Heun's
    third-order method.

    Reference:
        Heun, K. (1900). Neue Methoden zur approximativen Integration der
        Differentialgleichungen einer unabhangigen Veranderlichen.
        Z. Math. Phys., 45, 23--38.

    See also `diffrax.Heun` in:
        Kidger, P. (2021). On Neural Differential Equations. PhD thesis,
        University of Oxford.

    Args:
        atol: Absolute tolerance for adaptive stepping.
        rtol: Relative tolerance for adaptive stepping.
        max_steps: Maximum number of accepted steps before raising.
        safety: Safety factor for step-size adjustment (< 1).
        min_factor: Minimum step-size shrink factor.
        max_factor: Maximum step-size growth factor.
        max_step_size: Maximum absolute step size during adaptive integration.
        norm: Callable `norm(tensor) -> scalar` for local error measurement.
        device: Device for computations.
        dtype: Data type for computations.

    Example:
        ```python
        from torchebm.integrators import AdaptiveHeunIntegrator
        import torch

        integrator = AdaptiveHeunIntegrator(atol=1e-3, rtol=1e-2)
        state = {"x": torch.randn(100, 2)}
        drift = lambda x, t: -x
        result = integrator.integrate(
            state, step_size=0.1, n_steps=10, drift=drift,
        )
        ```
    """

    @property
    def tableau_a(self) -> Tuple[Tuple[float, ...], ...]:
        return (
            (),
            (1.0,),
        )

    @property
    def tableau_b(self) -> Tuple[float, ...]:
        return (0.5, 0.5)

    @property
    def tableau_c(self) -> Tuple[float, ...]:
        return (0.0, 1.0)

    @property
    def error_weights(self) -> Tuple[float, ...]:
        # e_i = b_i - b_hat_i  where b_hat = (1, 0) is Euler
        # From diffrax: b_error = [0.5, -0.5]
        return (0.5, -0.5)

    @property
    def order(self) -> int:
        return 2
