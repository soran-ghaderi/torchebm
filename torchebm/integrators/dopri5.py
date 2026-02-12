r"""Dormand-Prince 5(4) integrator with optional adaptive step-size control."""

from typing import Tuple

from torchebm.core import BaseRungeKuttaIntegrator


class Dopri5Integrator(BaseRungeKuttaIntegrator):
    r"""Dormand-Prince 5(4) explicit Runge-Kutta integrator.

    A 6-stage, 5th-order method with an embedded 4th-order solution for
    local error estimation.  When ``adaptive=True`` (the default for
    `integrate`), the step size is adjusted automatically to satisfy
    the tolerance ``atol + rtol * max(|x|, |x_new|)``.

    Fixed-step usage is available through :meth:`step` (always fixed) or by
    passing ``adaptive=False`` to :meth:`integrate`.

    The Butcher tableau is the standard Dormand-Prince pair:

    \[
    \begin{array}{c|cccccc}
    0 \\
    \tfrac{1}{5} & \tfrac{1}{5} \\
    \tfrac{3}{10} & \tfrac{3}{40} & \tfrac{9}{40} \\
    \tfrac{4}{5} & \tfrac{44}{45} & -\tfrac{56}{15} & \tfrac{32}{9} \\
    \tfrac{8}{9} & \tfrac{19372}{6561} & -\tfrac{25360}{2187}
        & \tfrac{64448}{6561} & -\tfrac{212}{729} \\
    1 & \tfrac{9017}{3168} & -\tfrac{355}{33}
        & \tfrac{46732}{5247} & \tfrac{49}{176} & -\tfrac{5103}{18656} \\
    \hline
    & \tfrac{35}{384} & 0 & \tfrac{500}{1113}
        & \tfrac{125}{192} & -\tfrac{2187}{6784} & \tfrac{11}{84}
    \end{array}
    \]

    Args:
        atol: Absolute tolerance for adaptive stepping.
        rtol: Relative tolerance for adaptive stepping.
        max_steps: Maximum number of accepted steps before raising.
        safety: Safety factor for step-size adjustment (< 1).
        min_factor: Minimum step-size shrink factor.
        max_factor: Maximum step-size growth factor.
        max_step_size: Maximum absolute step size during adaptive integration.
        norm: Callable ``norm(tensor) -> scalar`` for local error measurement.
        device: Device for computations.
        dtype: Data type for computations.

    Example:
        ```python
        from torchebm.integrators import Dopri5Integrator
        import torch

        integrator = Dopri5Integrator(atol=1e-5, rtol=1e-3)
        state = {"x": torch.randn(100, 2)}
        drift = lambda x, t: -x
        result = integrator.integrate(
            state, model=None, step_size=0.1, n_steps=10, drift=drift,
        )
        ```
    """

    @property
    def tableau_a(self) -> Tuple[Tuple[float, ...], ...]:
        return (
            (),
            (1 / 5,),
            (3 / 40, 9 / 40),
            (44 / 45, -56 / 15, 32 / 9),
            (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
            (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
        )

    @property
    def tableau_b(self) -> Tuple[float, ...]:
        return (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84)

    @property
    def tableau_c(self) -> Tuple[float, ...]:
        return (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0)

    @property
    def error_weights(self) -> Tuple[float, ...]:
        # e_i = b_i - b_hat_i  (7 entries, including FSAL stage)
        return (
            71 / 57600,
            0.0,
            -71 / 16695,
            71 / 1920,
            -17253 / 339200,
            22 / 525,
            -1 / 40,
        )

    @property
    def order(self) -> int:
        return 5

    @property
    def fsal(self) -> bool:
        return True
