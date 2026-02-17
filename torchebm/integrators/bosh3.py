r"""Bogacki-Shampine 3(2) integrator with adaptive step-size control (FSAL)."""

from typing import Tuple

from torchebm.core import BaseRungeKuttaIntegrator


class Bosh3Integrator(BaseRungeKuttaIntegrator):
    r"""Bogacki-Shampine 3(2) explicit Runge-Kutta integrator.

    A 3-stage, 3rd-order method with an embedded 2nd-order solution for
    local error estimation. The method has the FSAL (First Same As Last)
    property: the drift evaluation at the accepted solution is reused as
    the first stage of the next step, giving effectively 3 function
    evaluations per accepted step.

    When `adaptive=True` (the default for `integrate()` since
    `error_weights` is defined), the step size is adjusted automatically
    to satisfy the tolerance `atol + rtol * max(|x|, |x_new|)`.

    Fixed-step usage is available through `step()` (always fixed) or by
    passing `adaptive=False` to `integrate()`.

    The update rule:

    \[
    k_1 = f(x_n,\, t_n)
    \]

    \[
    k_2 = f\!\bigl(x_n + \tfrac{h}{2}\,k_1,\; t_n + \tfrac{h}{2}\bigr)
    \]

    \[
    k_3 = f\!\bigl(x_n + \tfrac{3h}{4}\,k_2,\; t_n + \tfrac{3h}{4}\bigr)
    \]

    \[
    x_{n+1} = x_n + h\bigl(\tfrac{2}{9}\,k_1 + \tfrac{1}{3}\,k_2
    + \tfrac{4}{9}\,k_3\bigr)
    \]

    The Butcher tableau:

    \[
    \begin{array}{c|ccc}
    0 \\
    \tfrac{1}{2} & \tfrac{1}{2} \\
    \tfrac{3}{4} & 0 & \tfrac{3}{4} \\
    \hline
    & \tfrac{2}{9} & \tfrac{1}{3} & \tfrac{4}{9}
    \end{array}
    \]

    The 3rd-order solution weights are \(b = (\tfrac{2}{9}, \tfrac{1}{3},
    \tfrac{4}{9})\). The embedded 2nd-order solution uses

    \[
    \hat{b} = \bigl(\tfrac{7}{24},\;\tfrac{1}{4},\;
    \tfrac{1}{3},\;\tfrac{1}{8}\bigr)
    \]

    where the 4th entry corresponds to the FSAL evaluation at the new
    solution point.

    Also sometimes known as "Ralston's third-order method".

    Reference:
        Bogacki, P. and Shampine, L. F. (1989). A 3(2) pair of
        Runge-Kutta formulas. Applied Mathematics Letters, 2(4), 321--325.

    See also `diffrax.Bosh3` in:
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
        from torchebm.integrators import Bosh3Integrator
        import torch

        integrator = Bosh3Integrator(atol=1e-4, rtol=1e-2)
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
            (1 / 2,),
            (0.0, 3 / 4),
        )

    @property
    def tableau_b(self) -> Tuple[float, ...]:
        return (2 / 9, 1 / 3, 4 / 9)

    @property
    def tableau_c(self) -> Tuple[float, ...]:
        return (0.0, 1 / 2, 3 / 4)

    @property
    def error_weights(self) -> Tuple[float, ...]:
        # e_i = b_i - b_hat_i  (4 entries: 3 stages + 1 FSAL)
        #
        # 3rd-order weights  b     = (2/9,   1/3,  4/9,  0  )
        # 2nd-order weights  b_hat = (7/24,  1/4,  1/3,  1/8)
        # error              e     = b - b_hat
        #
        # From diffrax:
        #   b_error = [2/9 - 7/24, 1/3 - 1/4, 4/9 - 1/3, -1/8]
        return (
            2 / 9 - 7 / 24,
            1 / 3 - 1 / 4,
            4 / 9 - 1 / 3,
            -1 / 8,
        )

    @property
    def order(self) -> int:
        return 3

    @property
    def fsal(self) -> bool:
        return True
