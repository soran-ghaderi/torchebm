# `torchebm.integrators`

Integrators for solving differential equations in energy-based models.

## `AdaptiveHeunIntegrator`

Bases: `BaseRungeKuttaIntegrator`

Adaptive Heun (Heun-Euler) 2(1) explicit Runge-Kutta integrator.

A 2-stage, 2nd-order method with an embedded 1st-order (Euler) solution for local error estimation. When `adaptive=True` (the default for `integrate()` since `error_weights` is defined), the step size is adjusted automatically.

Fixed-step usage is available through `step()` (always fixed) or by passing `adaptive=False` to `integrate()`.

The update rule (predictor-corrector form):

[ k_1 = f(x_n,, t_n) ]

[ k_2 = f(x_n + h,k_1,; t_n + h) ]

[ x\_{n+1} = x_n + \\tfrac{h}{2}\\bigl(k_1 + k_2\\bigr) ]

The Butcher tableau:

[ \\begin{array}{c|cc} 0 \\ 1 & 1 \\ \\hline & \\tfrac{1}{2} & \\tfrac{1}{2} \\end{array} ]

The embedded 1st-order solution is the Euler method with weights ((1, 0)). The error estimate is computed directly as:

[ e = h \\sum_i \\hat{e}\_i,k_i, \\qquad \\hat{e}\_i = b_i - \\hat{b}\_i = \\bigl(\\tfrac{1}{2} - 1,; \\tfrac{1}{2} - 0\\bigr) = \\bigl(-\\tfrac{1}{2},;\\tfrac{1}{2}\\bigr) ]

Also known as the "improved Euler method", "modified Euler method", or "explicit trapezoidal rule". Should not be confused with Heun's third-order method.

Reference

Heun, K. (1900). Neue Methoden zur approximativen Integration der Differentialgleichungen einer unabhangigen Veranderlichen. Z. Math. Phys., 45, 23--38.

See also `diffrax.Heun` in: Kidger, P. (2021). On Neural Differential Equations. PhD thesis, University of Oxford.

Parameters:

| Name            | Type                                   | Description                                                  | Default        |
| --------------- | -------------------------------------- | ------------------------------------------------------------ | -------------- |
| `atol`          | `float`                                | Absolute tolerance for adaptive stepping.                    | `1e-06`        |
| `rtol`          | `float`                                | Relative tolerance for adaptive stepping.                    | `0.001`        |
| `max_steps`     | `int`                                  | Maximum number of accepted steps before raising.             | `10000`        |
| `safety`        | `float`                                | Safety factor for step-size adjustment (< 1).                | `0.9`          |
| `min_factor`    | `float`                                | Minimum step-size shrink factor.                             | `0.2`          |
| `max_factor`    | `float`                                | Maximum step-size growth factor.                             | `10.0`         |
| `max_step_size` | `float`                                | Maximum absolute step size during adaptive integration.      | `float('inf')` |
| `norm`          | `Optional[Callable[[Tensor], Tensor]]` | Callable norm(tensor) -> scalar for local error measurement. | `None`         |
| `device`        | `Optional[device]`                     | Device for computations.                                     | `None`         |
| `dtype`         | `Optional[dtype]`                      | Data type for computations.                                  | `None`         |

Example

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

Source code in `torchebm/integrators/adaptive_heun.py`

````python
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
````

## `Bosh3Integrator`

Bases: `BaseRungeKuttaIntegrator`

Bogacki-Shampine 3(2) explicit Runge-Kutta integrator.

A 3-stage, 3rd-order method with an embedded 2nd-order solution for local error estimation. The method has the FSAL (First Same As Last) property: the drift evaluation at the accepted solution is reused as the first stage of the next step, giving effectively 3 function evaluations per accepted step.

When `adaptive=True` (the default for `integrate()` since `error_weights` is defined), the step size is adjusted automatically to satisfy the tolerance `atol + rtol * max(|x|, |x_new|)`.

Fixed-step usage is available through `step()` (always fixed) or by passing `adaptive=False` to `integrate()`.

The update rule:

[ k_1 = f(x_n,, t_n) ]

[ k_2 = f!\\bigl(x_n + \\tfrac{h}{2},k_1,; t_n + \\tfrac{h}{2}\\bigr) ]

[ k_3 = f!\\bigl(x_n + \\tfrac{3h}{4},k_2,; t_n + \\tfrac{3h}{4}\\bigr) ]

\[ x\_{n+1} = x_n + h\\bigl(\\tfrac{2}{9},k_1 + \\tfrac{1}{3},k_2

- \\tfrac{4}{9},k_3\\bigr) \]

The Butcher tableau:

[ \\begin{array}{c|ccc} 0 \\ \\tfrac{1}{2} & \\tfrac{1}{2} \\ \\tfrac{3}{4} & 0 & \\tfrac{3}{4} \\ \\hline & \\tfrac{2}{9} & \\tfrac{1}{3} & \\tfrac{4}{9} \\end{array} ]

The 3rd-order solution weights are (b = (\\tfrac{2}{9}, \\tfrac{1}{3}, \\tfrac{4}{9})). The embedded 2nd-order solution uses

[ \\hat{b} = \\bigl(\\tfrac{7}{24},;\\tfrac{1}{4},; \\tfrac{1}{3},;\\tfrac{1}{8}\\bigr) ]

where the 4th entry corresponds to the FSAL evaluation at the new solution point.

Also sometimes known as "Ralston's third-order method".

Reference

Bogacki, P. and Shampine, L. F. (1989). A 3(2) pair of Runge-Kutta formulas. Applied Mathematics Letters, 2(4), 321--325.

See also `diffrax.Bosh3` in: Kidger, P. (2021). On Neural Differential Equations. PhD thesis, University of Oxford.

Parameters:

| Name            | Type                                   | Description                                                  | Default        |
| --------------- | -------------------------------------- | ------------------------------------------------------------ | -------------- |
| `atol`          | `float`                                | Absolute tolerance for adaptive stepping.                    | `1e-06`        |
| `rtol`          | `float`                                | Relative tolerance for adaptive stepping.                    | `0.001`        |
| `max_steps`     | `int`                                  | Maximum number of accepted steps before raising.             | `10000`        |
| `safety`        | `float`                                | Safety factor for step-size adjustment (< 1).                | `0.9`          |
| `min_factor`    | `float`                                | Minimum step-size shrink factor.                             | `0.2`          |
| `max_factor`    | `float`                                | Maximum step-size growth factor.                             | `10.0`         |
| `max_step_size` | `float`                                | Maximum absolute step size during adaptive integration.      | `float('inf')` |
| `norm`          | `Optional[Callable[[Tensor], Tensor]]` | Callable norm(tensor) -> scalar for local error measurement. | `None`         |
| `device`        | `Optional[device]`                     | Device for computations.                                     | `None`         |
| `dtype`         | `Optional[dtype]`                      | Data type for computations.                                  | `None`         |

Example

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

Source code in `torchebm/integrators/bosh3.py`

````python
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
````

## `Dopri5Integrator`

Bases: `BaseRungeKuttaIntegrator`

Dormand-Prince 5(4) explicit Runge-Kutta integrator.

A 6-stage, 5th-order method with an embedded 4th-order solution for local error estimation and FSAL (First Same As Last) property. When `adaptive=True` (the default for `integrate()` since `error_weights` is defined), the step size is adjusted automatically to satisfy the tolerance `atol + rtol * max(|x|, |x_new|)`.

Fixed-step usage is available through `step()` (always fixed) or by passing `adaptive=False` to `integrate()`.

For an (s)-stage explicit Runge-Kutta method, the general update is:

[ k_i = f!\\bigl(x_n + h \\sum\_{j=1}^{i-1} a\_{ij},k_j,; t_n + c_i,h\\bigr), \\quad i = 1, \\ldots, s ]

[ x\_{n+1} = x_n + h \\sum\_{i=1}^{s} b_i,k_i ]

The Butcher tableau is the standard Dormand-Prince 5(4) pair:

[ \\begin{array}{c|cccccc} 0 \\ \\tfrac{1}{5} & \\tfrac{1}{5} \\ \\tfrac{3}{10} & \\tfrac{3}{40} & \\tfrac{9}{40} \\ \\tfrac{4}{5} & \\tfrac{44}{45} & -\\tfrac{56}{15} & \\tfrac{32}{9} \\ \\tfrac{8}{9} & \\tfrac{19372}{6561} & -\\tfrac{25360}{2187} & \\tfrac{64448}{6561} & -\\tfrac{212}{729} \\ 1 & \\tfrac{9017}{3168} & -\\tfrac{355}{33} & \\tfrac{46732}{5247} & \\tfrac{49}{176} & -\\tfrac{5103}{18656} \\ \\hline & \\tfrac{35}{384} & 0 & \\tfrac{500}{1113} & \\tfrac{125}{192} & -\\tfrac{2187}{6784} & \\tfrac{11}{84} \\end{array} ]

Reference

Dormand, J. R. and Prince, P. J. (1980). A family of embedded Runge-Kutta formulae. Journal of Computational and Applied Mathematics, 6(1), 19--26.

Parameters:

| Name            | Type                                   | Description                                                  | Default        |
| --------------- | -------------------------------------- | ------------------------------------------------------------ | -------------- |
| `atol`          | `float`                                | Absolute tolerance for adaptive stepping.                    | `1e-06`        |
| `rtol`          | `float`                                | Relative tolerance for adaptive stepping.                    | `0.001`        |
| `max_steps`     | `int`                                  | Maximum number of accepted steps before raising.             | `10000`        |
| `safety`        | `float`                                | Safety factor for step-size adjustment (< 1).                | `0.9`          |
| `min_factor`    | `float`                                | Minimum step-size shrink factor.                             | `0.2`          |
| `max_factor`    | `float`                                | Maximum step-size growth factor.                             | `10.0`         |
| `max_step_size` | `float`                                | Maximum absolute step size during adaptive integration.      | `float('inf')` |
| `norm`          | `Optional[Callable[[Tensor], Tensor]]` | Callable norm(tensor) -> scalar for local error measurement. | `None`         |
| `device`        | `Optional[device]`                     | Device for computations.                                     | `None`         |
| `dtype`         | `Optional[dtype]`                      | Data type for computations.                                  | `None`         |

Example

```python
from torchebm.integrators import Dopri5Integrator
import torch

integrator = Dopri5Integrator(atol=1e-5, rtol=1e-3)
state = {"x": torch.randn(100, 2)}
drift = lambda x, t: -x
result = integrator.integrate(
    state, step_size=0.1, n_steps=10, drift=drift,
)
```

Source code in `torchebm/integrators/dopri.py`

````python
class Dopri5Integrator(BaseRungeKuttaIntegrator):
    r"""Dormand-Prince 5(4) explicit Runge-Kutta integrator.

    A 6-stage, 5th-order method with an embedded 4th-order solution for
    local error estimation and FSAL (First Same As Last) property. When
    `adaptive=True` (the default for `integrate()` since `error_weights`
    is defined), the step size is adjusted automatically to satisfy
    the tolerance `atol + rtol * max(|x|, |x_new|)`.

    Fixed-step usage is available through `step()` (always fixed) or by
    passing `adaptive=False` to `integrate()`.

    For an \(s\)-stage explicit Runge-Kutta method, the general update is:

    \[
    k_i = f\!\bigl(x_n + h \sum_{j=1}^{i-1} a_{ij}\,k_j,\;
    t_n + c_i\,h\bigr), \quad i = 1, \ldots, s
    \]

    \[
    x_{n+1} = x_n + h \sum_{i=1}^{s} b_i\,k_i
    \]

    The Butcher tableau is the standard Dormand-Prince 5(4) pair:

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

    Reference:
        Dormand, J. R. and Prince, P. J. (1980). A family of embedded
        Runge-Kutta formulae. Journal of Computational and Applied
        Mathematics, 6(1), 19--26.

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
        from torchebm.integrators import Dopri5Integrator
        import torch

        integrator = Dopri5Integrator(atol=1e-5, rtol=1e-3)
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
````

## `Dopri8Integrator`

Bases: `BaseRungeKuttaIntegrator`

Dormand-Prince 8(7) explicit Runge-Kutta integrator.

A 13-stage, 8th-order method with an embedded 7th-order solution for local error estimation and FSAL (First Same As Last) property. When `adaptive=True` (the default for `integrate()` since `error_weights` is defined), the step size is adjusted automatically to satisfy the tolerance `atol + rtol * max(|x|, |x_new|)`.

Fixed-step usage is available through `step()` (always fixed) or by passing `adaptive=False` to `integrate()`.

For an (s)-stage explicit Runge-Kutta method, the general update is:

[ k_i = f!\\bigl(x_n + h \\sum\_{j=1}^{i-1} a\_{ij},k_j,; t_n + c_i,h\\bigr), \\quad i = 1, \\ldots, s ]

[ x\_{n+1} = x_n + h \\sum\_{i=1}^{s} b_i,k_i ]

Reference

Prince, P. J. and Dormand, J. R. (1981). High order embedded Runge-Kutta formulae. Journal of Computational and Applied Mathematics, 7(1), 67--75.

Parameters:

| Name            | Type                                   | Description                                                  | Default        |
| --------------- | -------------------------------------- | ------------------------------------------------------------ | -------------- |
| `atol`          | `float`                                | Absolute tolerance for adaptive stepping.                    | `1e-06`        |
| `rtol`          | `float`                                | Relative tolerance for adaptive stepping.                    | `0.001`        |
| `max_steps`     | `int`                                  | Maximum number of accepted steps before raising.             | `10000`        |
| `safety`        | `float`                                | Safety factor for step-size adjustment (< 1).                | `0.9`          |
| `min_factor`    | `float`                                | Minimum step-size shrink factor.                             | `0.2`          |
| `max_factor`    | `float`                                | Maximum step-size growth factor.                             | `10.0`         |
| `max_step_size` | `float`                                | Maximum absolute step size during adaptive integration.      | `float('inf')` |
| `norm`          | `Optional[Callable[[Tensor], Tensor]]` | Callable norm(tensor) -> scalar for local error measurement. | `None`         |
| `device`        | `Optional[device]`                     | Device for computations.                                     | `None`         |
| `dtype`         | `Optional[dtype]`                      | Data type for computations.                                  | `None`         |

Example

```python
from torchebm.integrators import Dopri8Integrator
import torch

integrator = Dopri8Integrator(atol=1e-8, rtol=1e-6)
state = {"x": torch.randn(100, 2)}
drift = lambda x, t: -x
result = integrator.integrate(
    state, step_size=0.1, n_steps=10, drift=drift,
)
```

Source code in `torchebm/integrators/dopri.py`

````python
class Dopri8Integrator(BaseRungeKuttaIntegrator):
    r"""Dormand-Prince 8(7) explicit Runge-Kutta integrator.

    A 13-stage, 8th-order method with an embedded 7th-order solution for
    local error estimation and FSAL (First Same As Last) property. When
    `adaptive=True` (the default for `integrate()` since `error_weights`
    is defined), the step size is adjusted automatically to satisfy
    the tolerance `atol + rtol * max(|x|, |x_new|)`.

    Fixed-step usage is available through `step()` (always fixed) or by
    passing `adaptive=False` to `integrate()`.

    For an \(s\)-stage explicit Runge-Kutta method, the general update is:

    \[
    k_i = f\!\bigl(x_n + h \sum_{j=1}^{i-1} a_{ij}\,k_j,\;
    t_n + c_i\,h\bigr), \quad i = 1, \ldots, s
    \]

    \[
    x_{n+1} = x_n + h \sum_{i=1}^{s} b_i\,k_i
    \]

    Reference:
        Prince, P. J. and Dormand, J. R. (1981). High order embedded
        Runge-Kutta formulae. Journal of Computational and Applied
        Mathematics, 7(1), 67--75.

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
        from torchebm.integrators import Dopri8Integrator
        import torch

        integrator = Dopri8Integrator(atol=1e-8, rtol=1e-6)
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
````

## `EulerMaruyamaIntegrator`

Bases: `BaseSDERungeKuttaIntegrator`

Euler-Maruyama integrator for Itô SDEs and ODEs.

The SDE form is:

[ \\mathrm{d}x = f(x,t),\\mathrm{d}t + \\sqrt{2D(x,t)},\\mathrm{d}W_t ]

When `diffusion` is omitted, this reduces to the Euler method for ODEs.

Update rule:

[ x\_{n+1} = x_n + f(x_n, t_n)\\Delta t + \\sqrt{2D(x_n,t_n)},\\Delta W_n ]

Parameters:

| Name            | Type                                   | Description                                                  | Default        |
| --------------- | -------------------------------------- | ------------------------------------------------------------ | -------------- |
| `device`        | `Optional[device]`                     | Device for computations.                                     | `None`         |
| `dtype`         | `Optional[dtype]`                      | Data type for computations.                                  | `None`         |
| `atol`          | `float`                                | Absolute tolerance for adaptive stepping.                    | `1e-06`        |
| `rtol`          | `float`                                | Relative tolerance for adaptive stepping.                    | `0.001`        |
| `max_steps`     | `int`                                  | Maximum number of steps before raising RuntimeError.         | `10000`        |
| `safety`        | `float`                                | Safety factor for step-size adjustment (< 1).                | `0.9`          |
| `min_factor`    | `float`                                | Minimum step-size shrink factor.                             | `0.2`          |
| `max_factor`    | `float`                                | Maximum step-size growth factor.                             | `10.0`         |
| `max_step_size` | `float`                                | Maximum absolute step size during adaptive integration.      | `float('inf')` |
| `norm`          | `Optional[Callable[[Tensor], Tensor]]` | Callable norm(tensor) -> scalar for local error measurement. | `None`         |

Example

```python
from torchebm.integrators import EulerMaruyamaIntegrator
import torch

integrator = EulerMaruyamaIntegrator()
state = {"x": torch.randn(100, 2)}
drift = lambda x, t: -x  # simple mean-reverting drift
result = integrator.step(
    state, step_size=0.01, drift=drift, noise_scale=1.0
)
```

Source code in `torchebm/integrators/euler_maruyama.py`

````python
class EulerMaruyamaIntegrator(BaseSDERungeKuttaIntegrator):
    r"""
    Euler-Maruyama integrator for Itô SDEs and ODEs.

    The SDE form is:

    \[
    \mathrm{d}x = f(x,t)\,\mathrm{d}t + \sqrt{2D(x,t)}\,\mathrm{d}W_t
    \]

    When `diffusion` is omitted, this reduces to the Euler method for ODEs.

    Update rule:

    \[
    x_{n+1} = x_n + f(x_n, t_n)\Delta t + \sqrt{2D(x_n,t_n)}\,\Delta W_n
    \]

    Args:
        device: Device for computations.
        dtype: Data type for computations.
        atol: Absolute tolerance for adaptive stepping.
        rtol: Relative tolerance for adaptive stepping.
        max_steps: Maximum number of steps before raising ``RuntimeError``.
        safety: Safety factor for step-size adjustment (< 1).
        min_factor: Minimum step-size shrink factor.
        max_factor: Maximum step-size growth factor.
        max_step_size: Maximum absolute step size during adaptive integration.
        norm: Callable ``norm(tensor) -> scalar`` for local error measurement.

    Example:
        ```python
        from torchebm.integrators import EulerMaruyamaIntegrator
        import torch

        integrator = EulerMaruyamaIntegrator()
        state = {"x": torch.randn(100, 2)}
        drift = lambda x, t: -x  # simple mean-reverting drift
        result = integrator.step(
            state, step_size=0.01, drift=drift, noise_scale=1.0
        )
        ```
    """

    @property
    def tableau_a(self):
        return ((),)

    @property
    def tableau_b(self):
        return (1.0,)

    @property
    def tableau_c(self):
        return (0.0,)

    # -- backward-compat shims for deprecated ``model`` kwarg ----------------

    @staticmethod
    def _resolve_model_to_drift(model, drift):
        """Convert deprecated ``model`` to a ``drift`` callable."""
        if model is not None:
            warnings.warn(
                "Passing 'model' to EulerMaruyamaIntegrator is deprecated. "
                "Use drift=lambda x, t: -model.gradient(x) instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if drift is None:
                drift = lambda x_, t_: -model.gradient(x_)
        return drift

    def step(
        self,
        state: Dict[str, torch.Tensor],
        step_size: torch.Tensor = None,
        *,
        model=None,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        drift = self._resolve_model_to_drift(model, drift)
        return super().step(state, step_size, drift=drift, **kwargs)

    def integrate(
        self,
        state: Dict[str, torch.Tensor],
        step_size: torch.Tensor = None,
        n_steps: int = None,
        *,
        model=None,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        drift = self._resolve_model_to_drift(model, drift)
        return super().integrate(state, step_size, n_steps, drift=drift, **kwargs)
````

## `HeunIntegrator`

Bases: `BaseSDERungeKuttaIntegrator`

Heun integrator (predictor-corrector) for Itô SDEs and ODEs.

A second-order method that uses a predictor step followed by a corrector:

[ \\mathrm{d}x = f(x,t),\\mathrm{d}t + \\sqrt{2D(x,t)},\\mathrm{d}W_t ]

Parameters:

| Name            | Type                                   | Description                                                  | Default        |
| --------------- | -------------------------------------- | ------------------------------------------------------------ | -------------- |
| `device`        | `Optional[device]`                     | Device for computations.                                     | `None`         |
| `dtype`         | `Optional[dtype]`                      | Data type for computations.                                  | `None`         |
| `atol`          | `float`                                | Absolute tolerance for adaptive stepping.                    | `1e-06`        |
| `rtol`          | `float`                                | Relative tolerance for adaptive stepping.                    | `0.001`        |
| `max_steps`     | `int`                                  | Maximum number of steps before raising RuntimeError.         | `10000`        |
| `safety`        | `float`                                | Safety factor for step-size adjustment (< 1).                | `0.9`          |
| `min_factor`    | `float`                                | Minimum step-size shrink factor.                             | `0.2`          |
| `max_factor`    | `float`                                | Maximum step-size growth factor.                             | `10.0`         |
| `max_step_size` | `float`                                | Maximum absolute step size during adaptive integration.      | `float('inf')` |
| `norm`          | `Optional[Callable[[Tensor], Tensor]]` | Callable norm(tensor) -> scalar for local error measurement. | `None`         |

Example

```python
from torchebm.integrators import HeunIntegrator
import torch

integrator = HeunIntegrator()
state = {"x": torch.randn(100, 2)}
t = torch.linspace(0, 1, 50)
drift = lambda x, t: -x
result = integrator.integrate(
    state, step_size=0.02, n_steps=50, drift=drift, t=t
)
```

Source code in `torchebm/integrators/heun.py`

````python
class HeunIntegrator(BaseSDERungeKuttaIntegrator):
    r"""
    Heun integrator (predictor-corrector) for Itô SDEs and ODEs.

    A second-order method that uses a predictor step followed by a corrector:

    \[
    \mathrm{d}x = f(x,t)\,\mathrm{d}t + \sqrt{2D(x,t)}\,\mathrm{d}W_t
    \]

    Args:
        device: Device for computations.
        dtype: Data type for computations.
        atol: Absolute tolerance for adaptive stepping.
        rtol: Relative tolerance for adaptive stepping.
        max_steps: Maximum number of steps before raising ``RuntimeError``.
        safety: Safety factor for step-size adjustment (< 1).
        min_factor: Minimum step-size shrink factor.
        max_factor: Maximum step-size growth factor.
        max_step_size: Maximum absolute step size during adaptive integration.
        norm: Callable ``norm(tensor) -> scalar`` for local error measurement.

    Example:
        ```python
        from torchebm.integrators import HeunIntegrator
        import torch

        integrator = HeunIntegrator()
        state = {"x": torch.randn(100, 2)}
        t = torch.linspace(0, 1, 50)
        drift = lambda x, t: -x
        result = integrator.integrate(
            state, step_size=0.02, n_steps=50, drift=drift, t=t
        )
        ```
    """

    @property
    def tableau_a(self):
        return ((), (1.0,))

    @property
    def tableau_b(self):
        return (0.5, 0.5)

    @property
    def tableau_c(self):
        return (0.0, 1.0)

    # -- backward-compat shims for deprecated ``model`` kwarg ----------------

    @staticmethod
    def _resolve_model_to_drift(model, drift):
        """Convert deprecated ``model`` to a ``drift`` callable."""
        if model is not None:
            warnings.warn(
                "Passing 'model' to HeunIntegrator is deprecated. "
                "Use drift=lambda x, t: -model.gradient(x) instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if drift is None:
                drift = lambda x_, t_: -model.gradient(x_)
        return drift

    def step(
        self,
        state: Dict[str, torch.Tensor],
        step_size: torch.Tensor = None,
        *,
        model=None,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        drift = self._resolve_model_to_drift(model, drift)
        return super().step(state, step_size, drift=drift, **kwargs)

    def integrate(
        self,
        state: Dict[str, torch.Tensor],
        step_size: torch.Tensor = None,
        n_steps: int = None,
        *,
        model=None,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        drift = self._resolve_model_to_drift(model, drift)
        return super().integrate(state, step_size, n_steps, drift=drift, **kwargs)
````

## `LeapfrogIntegrator`

Bases: `BaseIntegrator`

Symplectic leapfrog (Störmer–Verlet) integrator for Hamiltonian dynamics.

Update rule:

[ p\_{t+1/2} = p_t - \\frac{\\epsilon}{2} \\nabla_x U(x_t) ]

[ x\_{t+1} = x_t + \\epsilon p\_{t+1/2} ]

[ p\_{t+1} = p\_{t+1/2} - \\frac{\\epsilon}{2} \\nabla_x U(x\_{t+1}) ]

Parameters:

| Name     | Type               | Description                 | Default |
| -------- | ------------------ | --------------------------- | ------- |
| `device` | `Optional[device]` | Device for computations.    | `None`  |
| `dtype`  | `Optional[dtype]`  | Data type for computations. | `None`  |

Example

```python
from torchebm.integrators import LeapfrogIntegrator
import torch

energy_fn = ...  # an energy model with .gradient()
integrator = LeapfrogIntegrator()
state = {"x": torch.randn(100, 2), "p": torch.randn(100, 2)}
drift = lambda x, t: -energy_fn.gradient(x)
result = integrator.integrate(state, step_size=0.01, n_steps=10, drift=drift)
```

Source code in `torchebm/integrators/leapfrog.py`

````python
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
        import torch

        energy_fn = ...  # an energy model with .gradient()
        integrator = LeapfrogIntegrator()
        state = {"x": torch.randn(100, 2), "p": torch.randn(100, 2)}
        drift = lambda x, t: -energy_fn.gradient(x)
        result = integrator.integrate(state, step_size=0.01, n_steps=10, drift=drift)
        ```
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)

    @staticmethod
    def _resolve_deprecated_to_drift(model, potential_grad, drift):
        r"""Convert deprecated `model` or `potential_grad` to a `drift` callable."""
        if model is not None:
            warnings.warn(
                "Passing 'model' to LeapfrogIntegrator is deprecated. "
                "Use drift=lambda x, t: -model.gradient(x) instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if drift is None:
                drift = lambda x_, t_: -model.gradient(x_)
        if potential_grad is not None:
            warnings.warn(
                "Passing 'potential_grad' to LeapfrogIntegrator is deprecated. "
                "Use drift=lambda x, t: -potential_grad(x) instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if drift is None:
                drift = lambda x_, t_: -potential_grad(x_)
        return drift

    def step(
        self,
        state: Dict[str, torch.Tensor],
        model: Optional[BaseModel] = None,
        step_size: torch.Tensor = None,
        mass: Optional[Union[float, torch.Tensor]] = None,
        *,
        potential_grad: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        safe: bool = False,
    ) -> Dict[str, torch.Tensor]:
        r"""Advance one leapfrog step.

        Args:
            state: Current Hamiltonian state with keys `"x"` and `"p"`.
            model: Deprecated energy model. If provided and `drift` is `None`,
                uses `drift(x, t) = -model.gradient(x)`.
            step_size: Integration step size.
            mass: Optional mass term. Can be a scalar float or tensor.
            potential_grad: Deprecated callable for `\nabla_x U(x)`. If provided
                and `drift` is `None`, uses `drift(x, t) = -potential_grad(x)`.
            drift: Drift/force callable with signature `(x, t) -> force`.
            safe: If `True`, clamps force magnitudes and replaces NaNs by zeros.

        Returns:
            Updated state dictionary with keys `"x"` and `"p"`.
        """
        x = state["x"]
        p = state["p"]

        if not torch.is_tensor(step_size):
            step_size = torch.tensor(step_size, device=x.device, dtype=x.dtype)
        drift = self._resolve_deprecated_to_drift(model, potential_grad, drift)
        drift_fn = self._resolve_drift(drift)

        t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        force = drift_fn(x, t)
        if safe:
            force = torch.clamp(force, min=-1e6, max=1e6)

        p_half = p + 0.5 * step_size * force

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

        force_new = drift_fn(x_new, t)
        if safe:
            force_new = torch.clamp(force_new, min=-1e6, max=1e6)
        p_new = p_half + 0.5 * step_size * force_new

        if safe and (torch.isnan(x_new).any() or torch.isnan(p_new).any()):
            x_new = torch.nan_to_num(x_new, nan=0.0)
            p_new = torch.nan_to_num(p_new, nan=0.0)
        return {"x": x_new, "p": p_new}

    def integrate(
        self,
        state: Dict[str, torch.Tensor],
        model: Optional[BaseModel] = None,
        step_size: torch.Tensor = None,
        n_steps: int = None,
        mass: Optional[Union[float, torch.Tensor]] = None,
        *,
        potential_grad: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        drift: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        safe: bool = False,
        inference_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        r"""Integrate Hamiltonian dynamics for multiple leapfrog steps.

        Args:
            state: Initial Hamiltonian state with keys `"x"` and `"p"`.
            model: Deprecated energy model. If provided and `drift` is `None`,
                uses `drift(x, t) = -model.gradient(x)`.
            step_size: Integration step size.
            n_steps: Number of leapfrog steps to apply. Must be positive.
            mass: Optional mass term. Can be a scalar float or tensor.
            potential_grad: Deprecated callable for `\nabla_x U(x)`. If provided
                and `drift` is `None`, uses `drift(x, t) = -potential_grad(x)`.
            drift: Drift/force callable with signature `(x, t) -> force`.
            safe: If `True`, clamps force magnitudes and replaces NaNs by zeros.
            inference_mode: If `True`, runs integration under
                `torch.inference_mode()`.

        Returns:
            Final state dictionary with keys `"x"` and `"p"`.

        Raises:
            ValueError: If `n_steps <= 0`.
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")

        if inference_mode:
            with torch.inference_mode():
                return self.integrate(
                    state, model=model, step_size=step_size,
                    n_steps=n_steps, mass=mass,
                    potential_grad=potential_grad, drift=drift, safe=safe,
                )

        drift = self._resolve_deprecated_to_drift(model, potential_grad, drift)
        drift_fn = self._resolve_drift(drift)

        x = state["x"]
        p = state["p"]

        if not torch.is_tensor(step_size):
            step_size = torch.tensor(step_size, device=x.device, dtype=x.dtype)

        t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        for _ in range(n_steps):
            force = drift_fn(x, t)
            if safe:
                force = torch.clamp(force, min=-1e6, max=1e6)

            p_half = p + 0.5 * step_size * force

            if mass is None:
                x = x + step_size * p_half
            else:
                if isinstance(mass, float):
                    safe_mass = max(mass, 1e-10)
                    x = x + step_size * p_half / safe_mass
                else:
                    safe_mass = torch.clamp(mass, min=1e-10)
                    x = x + step_size * p_half / safe_mass.view(
                        *([1] * (len(x.shape) - 1)), -1
                    )

            force_new = drift_fn(x, t)
            if safe:
                force_new = torch.clamp(force_new, min=-1e6, max=1e6)
            p = p_half + 0.5 * step_size * force_new

            if safe and (torch.isnan(x).any() or torch.isnan(p).any()):
                x = torch.nan_to_num(x, nan=0.0)
                p = torch.nan_to_num(p, nan=0.0)

        return {"x": x, "p": p}
````

### `integrate(state, model=None, step_size=None, n_steps=None, mass=None, *, potential_grad=None, drift=None, safe=False, inference_mode=False)`

Integrate Hamiltonian dynamics for multiple leapfrog steps.

Parameters:

| Name             | Type                                           | Description                                                                                                   | Default    |
| ---------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------- |
| `state`          | `Dict[str, Tensor]`                            | Initial Hamiltonian state with keys "x" and "p".                                                              | *required* |
| `model`          | `Optional[BaseModel]`                          | Deprecated energy model. If provided and drift is None, uses drift(x, t) = -model.gradient(x).                | `None`     |
| `step_size`      | `Tensor`                                       | Integration step size.                                                                                        | `None`     |
| `n_steps`        | `int`                                          | Number of leapfrog steps to apply. Must be positive.                                                          | `None`     |
| `mass`           | `Optional[Union[float, Tensor]]`               | Optional mass term. Can be a scalar float or tensor.                                                          | `None`     |
| `potential_grad` | `Optional[Callable[[Tensor], Tensor]]`         | Deprecated callable for \\nabla_x U(x). If provided and drift is None, uses drift(x, t) = -potential_grad(x). | `None`     |
| `drift`          | `Optional[Callable[[Tensor, Tensor], Tensor]]` | Drift/force callable with signature (x, t) -> force.                                                          | `None`     |
| `safe`           | `bool`                                         | If True, clamps force magnitudes and replaces NaNs by zeros.                                                  | `False`    |
| `inference_mode` | `bool`                                         | If True, runs integration under torch.inference_mode().                                                       | `False`    |

Returns:

| Type                | Description                                   |
| ------------------- | --------------------------------------------- |
| `Dict[str, Tensor]` | Final state dictionary with keys "x" and "p". |

Raises:

| Type         | Description       |
| ------------ | ----------------- |
| `ValueError` | If n_steps \<= 0. |

Source code in `torchebm/integrators/leapfrog.py`

```python
def integrate(
    self,
    state: Dict[str, torch.Tensor],
    model: Optional[BaseModel] = None,
    step_size: torch.Tensor = None,
    n_steps: int = None,
    mass: Optional[Union[float, torch.Tensor]] = None,
    *,
    potential_grad: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    drift: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    safe: bool = False,
    inference_mode: bool = False,
) -> Dict[str, torch.Tensor]:
    r"""Integrate Hamiltonian dynamics for multiple leapfrog steps.

    Args:
        state: Initial Hamiltonian state with keys `"x"` and `"p"`.
        model: Deprecated energy model. If provided and `drift` is `None`,
            uses `drift(x, t) = -model.gradient(x)`.
        step_size: Integration step size.
        n_steps: Number of leapfrog steps to apply. Must be positive.
        mass: Optional mass term. Can be a scalar float or tensor.
        potential_grad: Deprecated callable for `\nabla_x U(x)`. If provided
            and `drift` is `None`, uses `drift(x, t) = -potential_grad(x)`.
        drift: Drift/force callable with signature `(x, t) -> force`.
        safe: If `True`, clamps force magnitudes and replaces NaNs by zeros.
        inference_mode: If `True`, runs integration under
            `torch.inference_mode()`.

    Returns:
        Final state dictionary with keys `"x"` and `"p"`.

    Raises:
        ValueError: If `n_steps <= 0`.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    if inference_mode:
        with torch.inference_mode():
            return self.integrate(
                state, model=model, step_size=step_size,
                n_steps=n_steps, mass=mass,
                potential_grad=potential_grad, drift=drift, safe=safe,
            )

    drift = self._resolve_deprecated_to_drift(model, potential_grad, drift)
    drift_fn = self._resolve_drift(drift)

    x = state["x"]
    p = state["p"]

    if not torch.is_tensor(step_size):
        step_size = torch.tensor(step_size, device=x.device, dtype=x.dtype)

    t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

    for _ in range(n_steps):
        force = drift_fn(x, t)
        if safe:
            force = torch.clamp(force, min=-1e6, max=1e6)

        p_half = p + 0.5 * step_size * force

        if mass is None:
            x = x + step_size * p_half
        else:
            if isinstance(mass, float):
                safe_mass = max(mass, 1e-10)
                x = x + step_size * p_half / safe_mass
            else:
                safe_mass = torch.clamp(mass, min=1e-10)
                x = x + step_size * p_half / safe_mass.view(
                    *([1] * (len(x.shape) - 1)), -1
                )

        force_new = drift_fn(x, t)
        if safe:
            force_new = torch.clamp(force_new, min=-1e6, max=1e6)
        p = p_half + 0.5 * step_size * force_new

        if safe and (torch.isnan(x).any() or torch.isnan(p).any()):
            x = torch.nan_to_num(x, nan=0.0)
            p = torch.nan_to_num(p, nan=0.0)

    return {"x": x, "p": p}
```

### `step(state, model=None, step_size=None, mass=None, *, potential_grad=None, drift=None, safe=False)`

Advance one leapfrog step.

Parameters:

| Name             | Type                                           | Description                                                                                                   | Default    |
| ---------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------- |
| `state`          | `Dict[str, Tensor]`                            | Current Hamiltonian state with keys "x" and "p".                                                              | *required* |
| `model`          | `Optional[BaseModel]`                          | Deprecated energy model. If provided and drift is None, uses drift(x, t) = -model.gradient(x).                | `None`     |
| `step_size`      | `Tensor`                                       | Integration step size.                                                                                        | `None`     |
| `mass`           | `Optional[Union[float, Tensor]]`               | Optional mass term. Can be a scalar float or tensor.                                                          | `None`     |
| `potential_grad` | `Optional[Callable[[Tensor], Tensor]]`         | Deprecated callable for \\nabla_x U(x). If provided and drift is None, uses drift(x, t) = -potential_grad(x). | `None`     |
| `drift`          | `Optional[Callable[[Tensor, Tensor], Tensor]]` | Drift/force callable with signature (x, t) -> force.                                                          | `None`     |
| `safe`           | `bool`                                         | If True, clamps force magnitudes and replaces NaNs by zeros.                                                  | `False`    |

Returns:

| Type                | Description                                     |
| ------------------- | ----------------------------------------------- |
| `Dict[str, Tensor]` | Updated state dictionary with keys "x" and "p". |

Source code in `torchebm/integrators/leapfrog.py`

```python
def step(
    self,
    state: Dict[str, torch.Tensor],
    model: Optional[BaseModel] = None,
    step_size: torch.Tensor = None,
    mass: Optional[Union[float, torch.Tensor]] = None,
    *,
    potential_grad: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    drift: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    safe: bool = False,
) -> Dict[str, torch.Tensor]:
    r"""Advance one leapfrog step.

    Args:
        state: Current Hamiltonian state with keys `"x"` and `"p"`.
        model: Deprecated energy model. If provided and `drift` is `None`,
            uses `drift(x, t) = -model.gradient(x)`.
        step_size: Integration step size.
        mass: Optional mass term. Can be a scalar float or tensor.
        potential_grad: Deprecated callable for `\nabla_x U(x)`. If provided
            and `drift` is `None`, uses `drift(x, t) = -potential_grad(x)`.
        drift: Drift/force callable with signature `(x, t) -> force`.
        safe: If `True`, clamps force magnitudes and replaces NaNs by zeros.

    Returns:
        Updated state dictionary with keys `"x"` and `"p"`.
    """
    x = state["x"]
    p = state["p"]

    if not torch.is_tensor(step_size):
        step_size = torch.tensor(step_size, device=x.device, dtype=x.dtype)
    drift = self._resolve_deprecated_to_drift(model, potential_grad, drift)
    drift_fn = self._resolve_drift(drift)

    t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

    force = drift_fn(x, t)
    if safe:
        force = torch.clamp(force, min=-1e6, max=1e6)

    p_half = p + 0.5 * step_size * force

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

    force_new = drift_fn(x_new, t)
    if safe:
        force_new = torch.clamp(force_new, min=-1e6, max=1e6)
    p_new = p_half + 0.5 * step_size * force_new

    if safe and (torch.isnan(x_new).any() or torch.isnan(p_new).any()):
        x_new = torch.nan_to_num(x_new, nan=0.0)
        p_new = torch.nan_to_num(p_new, nan=0.0)
    return {"x": x_new, "p": p_new}
```

## `RK4Integrator`

Bases: `BaseRungeKuttaIntegrator`

Classical 4th-order Runge-Kutta integrator.

A 4-stage, 4th-order explicit method. Fixed-step only (no embedded error pair).

The update rule computes four stages and combines them:

[ k_1 = f(x_n,, t_n) ]

[ k_2 = f!\\bigl(x_n + \\tfrac{h}{2},k_1,; t_n + \\tfrac{h}{2}\\bigr) ]

[ k_3 = f!\\bigl(x_n + \\tfrac{h}{2},k_2,; t_n + \\tfrac{h}{2}\\bigr) ]

[ k_4 = f(x_n + h,k_3,; t_n + h) ]

[ x\_{n+1} = x_n + \\tfrac{h}{6}\\bigl(k_1 + 2k_2 + 2k_3 + k_4\\bigr) ]

The Butcher tableau:

[ \\begin{array}{c|cccc} 0 \\ \\tfrac{1}{2} & \\tfrac{1}{2} \\ \\tfrac{1}{2} & 0 & \\tfrac{1}{2} \\ 1 & 0 & 0 & 1 \\ \\hline & \\tfrac{1}{6} & \\tfrac{1}{3} & \\tfrac{1}{3} & \\tfrac{1}{6} \\end{array} ]

Reference

Kutta, W. (1901). Beitrag zur naherungsweisen Integration totaler Differentialgleichungen. Z. Math. Phys., 46, 435--453.

Parameters:

| Name     | Type               | Description                 | Default |
| -------- | ------------------ | --------------------------- | ------- |
| `device` | `Optional[device]` | Device for computations.    | `None`  |
| `dtype`  | `Optional[dtype]`  | Data type for computations. | `None`  |

Example

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

Source code in `torchebm/integrators/rk4.py`

````python
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
````
