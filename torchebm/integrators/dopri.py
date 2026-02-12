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


class Dopri8Integrator(BaseRungeKuttaIntegrator):
    r"""Dormand-Prince 8(7) explicit Runge-Kutta integrator.

    A 13-stage, 8th-order method with an embedded 7th-order solution for
    local error estimation (FSAL).  When ``adaptive=True`` (the default for
    `integrate`), the step size is adjusted automatically to satisfy
    the tolerance ``atol + rtol * max(|x|, |x_new|)``.

    Fixed-step usage is available through :meth:`step` (always fixed) or by
    passing ``adaptive=False`` to :meth:`integrate`.

    Coefficients from:

    \[
    \text{Prince, P. J. and Dormand, J. R. (1981).
    High order embedded Runge--Kutta formulae.
    J. Comp. Appl. Math, 7(1), 67--75.}
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
        from torchebm.integrators import Dopri8Integrator
        import torch

        integrator = Dopri8Integrator(atol=1e-8, rtol=1e-6)
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
            (1 / 18,),
            (1 / 48, 1 / 16),
            (1 / 32, 0, 3 / 32),
            (5 / 16, 0, -75 / 64, 75 / 64),
            (3 / 80, 0, 0, 3 / 16, 3 / 20),
            (
                29443841 / 614563906, 0, 0,
                77736538 / 692538347, -28693883 / 1125000000,
                23124283 / 1800000000,
            ),
            (
                16016141 / 946692911, 0, 0,
                61564180 / 158732637, 22789713 / 633445777,
                545815736 / 2771057229, -180193667 / 1043307555,
            ),
            (
                39632708 / 573591083, 0, 0,
                -433636366 / 683701615, -421739975 / 2616292301,
                100302831 / 723423059, 790204164 / 839813087,
                800635310 / 3783071287,
            ),
            (
                246121993 / 1340847787, 0, 0,
                -37695042795 / 15268766246, -309121744 / 1061227803,
                -12992083 / 490766935, 6005943493 / 2108947869,
                393006217 / 1396673457, 123872331 / 1001029789,
            ),
            (
                -1028468189 / 846180014, 0, 0,
                8478235783 / 508512852, 1311729495 / 1432422823,
                -10304129995 / 1701304382, -48777925059 / 3047939560,
                15336726248 / 1032824649, -45442868181 / 3398467696,
                3065993473 / 597172653,
            ),
            (
                185892177 / 718116043, 0, 0,
                -3185094517 / 667107341, -477755414 / 1098053517,
                -703635378 / 230739211, 5731566787 / 1027545527,
                5232866602 / 850066563, -4093664535 / 808688257,
                3962137247 / 1805957418, 65686358 / 487910083,
            ),
            (
                403863854 / 491063109, 0, 0,
                -5068492393 / 434740067, -411421997 / 543043805,
                652783627 / 914296604, 11173962825 / 925320556,
                -13158990841 / 6184727034, 3936647629 / 1978049680,
                -160528059 / 685178525, 248638103 / 1413531060,
                0,
            ),
        )

    @property
    def tableau_b(self) -> Tuple[float, ...]:
        return (
            14005451 / 335480064, 0, 0, 0, 0,
            -59238493 / 1068277825, 181606767 / 758867731,
            561292985 / 797845732, -1041891430 / 1371343529,
            760417239 / 1151165299, 118820643 / 751138087,
            -528747749 / 2220607170, 1 / 4,
        )

    @property
    def tableau_c(self) -> Tuple[float, ...]:
        return (
            0.0, 1 / 18, 1 / 12, 1 / 8, 5 / 16, 3 / 8,
            59 / 400, 93 / 200, 5490023248 / 9719169821,
            13 / 20, 1201146811 / 1299019798, 1.0, 1.0,
        )

    @property
    def error_weights(self) -> Tuple[float, ...]:
        return (
            14005451 / 335480064 - 13451932 / 455176623,
            0,
            0,
            0,
            0,
            -59238493 / 1068277825 + 808719846 / 976000145,
            181606767 / 758867731 - 1757004468 / 5645159321,
            561292985 / 797845732 - 656045339 / 265891186,
            -1041891430 / 1371343529 + 3867574721 / 1518517206,
            760417239 / 1151165299 - 465885868 / 322736535,
            118820643 / 751138087 - 53011238 / 667516719,
            -528747749 / 2220607170 - 2 / 45,
            1 / 4,
            0,
        )

    @property
    def order(self) -> int:
        return 8

    @property
    def fsal(self) -> bool:
        return True