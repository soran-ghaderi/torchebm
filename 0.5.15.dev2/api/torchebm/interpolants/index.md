# `torchebm.interpolants`

Stochastic interpolants for generative modeling.

Interpolants define conditional probability paths between source (noise) and target (data) distributions, parameterized by schedules α(t) and σ(t).

## `CosineInterpolant`

Bases: `BaseInterpolant`

Cosine (geodesic variance preserving) interpolant.

Also known as the GVP interpolant. Uses trigonometric functions to maintain unit variance throughout the interpolation path.

The interpolation is defined as:

[ x_t = \\sin\\left(\\frac{\\pi t}{2}\\right) x_1 + \\cos\\left(\\frac{\\pi t}{2}\\right) x_0 ]

This satisfies (\\alpha(t)^2 + \\sigma(t)^2 = 1).

Example

```python
from torchebm.interpolants import CosineInterpolant
import torch

interpolant = CosineInterpolant()
x0 = torch.randn(32, 2)  # noise
x1 = torch.randn(32, 2)  # data
t = torch.rand(32)
xt, ut = interpolant.interpolate(x0, x1, t)
```

Source code in `torchebm/interpolants/cosine.py`

````python
class CosineInterpolant(BaseInterpolant):
    r"""
    Cosine (geodesic variance preserving) interpolant.

    Also known as the GVP interpolant. Uses trigonometric functions to
    maintain unit variance throughout the interpolation path.

    The interpolation is defined as:

    \[
    x_t = \sin\left(\frac{\pi t}{2}\right) x_1 + \cos\left(\frac{\pi t}{2}\right) x_0
    \]

    This satisfies \(\alpha(t)^2 + \sigma(t)^2 = 1\).

    Example:
        ```python
        from torchebm.interpolants import CosineInterpolant
        import torch

        interpolant = CosineInterpolant()
        x0 = torch.randn(32, 2)  # noise
        x1 = torch.randn(32, 2)  # data
        t = torch.rand(32)
        xt, ut = interpolant.interpolate(x0, x1, t)
        ```
    """

    def compute_alpha_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute \(\alpha(t) = \sin(\pi t / 2)\) and its derivative.

        Args:
            t: Time tensor.

        Returns:
            Tuple of (α(t), α̇(t)).
        """
        alpha = torch.sin(t * math.pi / 2)
        d_alpha = (math.pi / 2) * torch.cos(t * math.pi / 2)
        return alpha, d_alpha

    def compute_sigma_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute \(\sigma(t) = \cos(\pi t / 2)\) and its derivative.

        Args:
            t: Time tensor.

        Returns:
            Tuple of (σ(t), σ̇(t)).
        """
        sigma = torch.cos(t * math.pi / 2)
        d_sigma = -(math.pi / 2) * torch.sin(t * math.pi / 2)
        return sigma, d_sigma

    def compute_d_alpha_alpha_ratio_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Compute \(\dot{\alpha}(t) / \alpha(t) = (\pi/2) \cot(\pi t / 2)\).

        Args:
            t: Time tensor.

        Returns:
            The ratio with clamping for stability.
        """
        return math.pi / (2 * torch.clamp(torch.tan(t * math.pi / 2), min=1e-8))
````

### `compute_alpha_t(t)`

Compute (\\alpha(t) = \\sin(\\pi t / 2)) and its derivative.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `t`  | `Tensor` | Time tensor. | *required* |

Returns:

| Type                    | Description            |
| ----------------------- | ---------------------- |
| `Tuple[Tensor, Tensor]` | Tuple of (α(t), α̇(t)). |

Source code in `torchebm/interpolants/cosine.py`

```python
def compute_alpha_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute \(\alpha(t) = \sin(\pi t / 2)\) and its derivative.

    Args:
        t: Time tensor.

    Returns:
        Tuple of (α(t), α̇(t)).
    """
    alpha = torch.sin(t * math.pi / 2)
    d_alpha = (math.pi / 2) * torch.cos(t * math.pi / 2)
    return alpha, d_alpha
```

### `compute_d_alpha_alpha_ratio_t(t)`

Compute (\\dot{\\alpha}(t) / \\alpha(t) = (\\pi/2) \\cot(\\pi t / 2)).

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `t`  | `Tensor` | Time tensor. | *required* |

Returns:

| Type     | Description                            |
| -------- | -------------------------------------- |
| `Tensor` | The ratio with clamping for stability. |

Source code in `torchebm/interpolants/cosine.py`

```python
def compute_d_alpha_alpha_ratio_t(self, t: torch.Tensor) -> torch.Tensor:
    r"""
    Compute \(\dot{\alpha}(t) / \alpha(t) = (\pi/2) \cot(\pi t / 2)\).

    Args:
        t: Time tensor.

    Returns:
        The ratio with clamping for stability.
    """
    return math.pi / (2 * torch.clamp(torch.tan(t * math.pi / 2), min=1e-8))
```

### `compute_sigma_t(t)`

Compute (\\sigma(t) = \\cos(\\pi t / 2)) and its derivative.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `t`  | `Tensor` | Time tensor. | *required* |

Returns:

| Type                    | Description            |
| ----------------------- | ---------------------- |
| `Tuple[Tensor, Tensor]` | Tuple of (σ(t), σ̇(t)). |

Source code in `torchebm/interpolants/cosine.py`

```python
def compute_sigma_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute \(\sigma(t) = \cos(\pi t / 2)\) and its derivative.

    Args:
        t: Time tensor.

    Returns:
        Tuple of (σ(t), σ̇(t)).
    """
    sigma = torch.cos(t * math.pi / 2)
    d_sigma = -(math.pi / 2) * torch.sin(t * math.pi / 2)
    return sigma, d_sigma
```

## `LinearInterpolant`

Bases: `BaseInterpolant`

Linear interpolant between noise and data distributions.

Also known as the optimal transport (OT) or rectified flow interpolant.

The interpolation is defined as:

[ x_t = t \\cdot x_1 + (1 - t) \\cdot x_0 ]

with (\\alpha(t) = t) and (\\sigma(t) = 1 - t).

Example

```python
from torchebm.interpolants import LinearInterpolant
import torch

interpolant = LinearInterpolant()
x0 = torch.randn(32, 2)  # noise
x1 = torch.randn(32, 2)  # data
t = torch.rand(32)
xt, ut = interpolant.interpolate(x0, x1, t)
```

Source code in `torchebm/interpolants/linear.py`

````python
class LinearInterpolant(BaseInterpolant):
    r"""
    Linear interpolant between noise and data distributions.

    Also known as the optimal transport (OT) or rectified flow interpolant.

    The interpolation is defined as:

    \[
    x_t = t \cdot x_1 + (1 - t) \cdot x_0
    \]

    with \(\alpha(t) = t\) and \(\sigma(t) = 1 - t\).

    Example:
        ```python
        from torchebm.interpolants import LinearInterpolant
        import torch

        interpolant = LinearInterpolant()
        x0 = torch.randn(32, 2)  # noise
        x1 = torch.randn(32, 2)  # data
        t = torch.rand(32)
        xt, ut = interpolant.interpolate(x0, x1, t)
        ```
    """

    def compute_alpha_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute \(\alpha(t) = t\) and \(\dot{\alpha}(t) = 1\).

        Args:
            t: Time tensor.

        Returns:
            Tuple of (α(t), α̇(t)).
        """
        alpha = t
        d_alpha = torch.ones_like(t)
        return alpha, d_alpha

    def compute_sigma_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute \(\sigma(t) = 1 - t\) and \(\dot{\sigma}(t) = -1\).

        Args:
            t: Time tensor.

        Returns:
            Tuple of (σ(t), σ̇(t)).
        """
        sigma = 1 - t
        d_sigma = -torch.ones_like(t)
        return sigma, d_sigma

    def compute_d_alpha_alpha_ratio_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Compute \(\dot{\alpha}(t) / \alpha(t) = 1/t\).

        Args:
            t: Time tensor.

        Returns:
            The ratio 1/t with clamping for stability.
        """
        return 1 / torch.clamp(t, min=1e-8)
````

### `compute_alpha_t(t)`

Compute (\\alpha(t) = t) and (\\dot{\\alpha}(t) = 1).

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `t`  | `Tensor` | Time tensor. | *required* |

Returns:

| Type                    | Description            |
| ----------------------- | ---------------------- |
| `Tuple[Tensor, Tensor]` | Tuple of (α(t), α̇(t)). |

Source code in `torchebm/interpolants/linear.py`

```python
def compute_alpha_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute \(\alpha(t) = t\) and \(\dot{\alpha}(t) = 1\).

    Args:
        t: Time tensor.

    Returns:
        Tuple of (α(t), α̇(t)).
    """
    alpha = t
    d_alpha = torch.ones_like(t)
    return alpha, d_alpha
```

### `compute_d_alpha_alpha_ratio_t(t)`

Compute (\\dot{\\alpha}(t) / \\alpha(t) = 1/t).

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `t`  | `Tensor` | Time tensor. | *required* |

Returns:

| Type     | Description                                |
| -------- | ------------------------------------------ |
| `Tensor` | The ratio 1/t with clamping for stability. |

Source code in `torchebm/interpolants/linear.py`

```python
def compute_d_alpha_alpha_ratio_t(self, t: torch.Tensor) -> torch.Tensor:
    r"""
    Compute \(\dot{\alpha}(t) / \alpha(t) = 1/t\).

    Args:
        t: Time tensor.

    Returns:
        The ratio 1/t with clamping for stability.
    """
    return 1 / torch.clamp(t, min=1e-8)
```

### `compute_sigma_t(t)`

Compute (\\sigma(t) = 1 - t) and (\\dot{\\sigma}(t) = -1).

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `t`  | `Tensor` | Time tensor. | *required* |

Returns:

| Type                    | Description            |
| ----------------------- | ---------------------- |
| `Tuple[Tensor, Tensor]` | Tuple of (σ(t), σ̇(t)). |

Source code in `torchebm/interpolants/linear.py`

```python
def compute_sigma_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute \(\sigma(t) = 1 - t\) and \(\dot{\sigma}(t) = -1\).

    Args:
        t: Time tensor.

    Returns:
        Tuple of (σ(t), σ̇(t)).
    """
    sigma = 1 - t
    d_sigma = -torch.ones_like(t)
    return sigma, d_sigma
```

## `VariancePreservingInterpolant`

Bases: `BaseInterpolant`

Variance preserving (VP) interpolant with linear beta schedule.

Corresponds to the noise schedule used in DDPM and score-based diffusion models.

The forward process is defined via:

[ \\alpha(t) = \\exp\\left(-\\frac{1}{4}(1-t)^2(\\sigma\_{\\max} - \\sigma\_{\\min}) - \\frac{1}{2}(1-t)\\sigma\_{\\min}\\right) ]

[ \\sigma(t) = \\sqrt{1 - \\alpha(t)^2} ]

Parameters:

| Name        | Type    | Description                          | Default |
| ----------- | ------- | ------------------------------------ | ------- |
| `sigma_min` | `float` | Minimum noise level (default: 0.1).  | `0.1`   |
| `sigma_max` | `float` | Maximum noise level (default: 20.0). | `20.0`  |

Example

```python
from torchebm.interpolants import VariancePreservingInterpolant
import torch

interpolant = VariancePreservingInterpolant(sigma_min=0.1, sigma_max=20.0)
x0 = torch.randn(32, 2)  # noise
x1 = torch.randn(32, 2)  # data
t = torch.rand(32)
xt, ut = interpolant.interpolate(x0, x1, t)
```

Source code in `torchebm/interpolants/variance_preserving.py`

````python
class VariancePreservingInterpolant(BaseInterpolant):
    r"""
    Variance preserving (VP) interpolant with linear beta schedule.

    Corresponds to the noise schedule used in DDPM and score-based diffusion models.

    The forward process is defined via:

    \[
    \alpha(t) = \exp\left(-\frac{1}{4}(1-t)^2(\sigma_{\max} - \sigma_{\min}) - \frac{1}{2}(1-t)\sigma_{\min}\right)
    \]

    \[
    \sigma(t) = \sqrt{1 - \alpha(t)^2}
    \]

    Args:
        sigma_min: Minimum noise level (default: 0.1).
        sigma_max: Maximum noise level (default: 20.0).

    Example:
        ```python
        from torchebm.interpolants import VariancePreservingInterpolant
        import torch

        interpolant = VariancePreservingInterpolant(sigma_min=0.1, sigma_max=20.0)
        x0 = torch.randn(32, 2)  # noise
        x1 = torch.randn(32, 2)  # data
        t = torch.rand(32)
        xt, ut = interpolant.interpolate(x0, x1, t)
        ```
    """

    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def _log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        r"""Compute log of mean coefficient."""
        return (
            -0.25 * ((1 - t) ** 2) * (self.sigma_max - self.sigma_min)
            - 0.5 * (1 - t) * self.sigma_min
        )

    def _d_log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        r"""Compute derivative of log mean coefficient."""
        return 0.5 * (1 - t) * (self.sigma_max - self.sigma_min) + 0.5 * self.sigma_min

    def compute_alpha_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute \(\alpha(t)\) and its derivative for VP schedule.

        Args:
            t: Time tensor.

        Returns:
            Tuple of (α(t), α̇(t)).
        """
        lmc = self._log_mean_coeff(t)
        alpha = torch.exp(lmc)
        d_alpha = alpha * self._d_log_mean_coeff(t)
        return alpha, d_alpha

    def compute_sigma_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute \(\sigma(t)\) and its derivative for VP schedule.

        Args:
            t: Time tensor.

        Returns:
            Tuple of (σ(t), σ̇(t)).
        """
        p_sigma = 2 * self._log_mean_coeff(t)
        exp_p = torch.exp(p_sigma)
        sigma = torch.sqrt(torch.clamp(1 - exp_p, min=1e-12))
        d_sigma = exp_p * (2 * self._d_log_mean_coeff(t)) / (-2 * sigma)
        return sigma, d_sigma

    def compute_d_alpha_alpha_ratio_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Compute \(\dot{\alpha}(t) / \alpha(t)\) directly from log mean coefficient.

        This is more numerically stable than dividing α̇ by α.

        Args:
            t: Time tensor.

        Returns:
            The ratio (which equals d_log_mean_coeff).
        """
        return self._d_log_mean_coeff(t)

    def compute_drift(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute drift for VP schedule using the beta parameterization.

        Args:
            x: Current state of shape (batch_size, ...).
            t: Time values of shape (batch_size,).

        Returns:
            Tuple of (drift_mean, drift_var).
        """
        t_expanded = expand_t_like_x(t, x)
        beta_t = self.sigma_min + (1 - t_expanded) * (self.sigma_max - self.sigma_min)
        return -0.5 * beta_t * x, beta_t / 2
````

### `compute_alpha_t(t)`

Compute (\\alpha(t)) and its derivative for VP schedule.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `t`  | `Tensor` | Time tensor. | *required* |

Returns:

| Type                    | Description            |
| ----------------------- | ---------------------- |
| `Tuple[Tensor, Tensor]` | Tuple of (α(t), α̇(t)). |

Source code in `torchebm/interpolants/variance_preserving.py`

```python
def compute_alpha_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute \(\alpha(t)\) and its derivative for VP schedule.

    Args:
        t: Time tensor.

    Returns:
        Tuple of (α(t), α̇(t)).
    """
    lmc = self._log_mean_coeff(t)
    alpha = torch.exp(lmc)
    d_alpha = alpha * self._d_log_mean_coeff(t)
    return alpha, d_alpha
```

### `compute_d_alpha_alpha_ratio_t(t)`

Compute (\\dot{\\alpha}(t) / \\alpha(t)) directly from log mean coefficient.

This is more numerically stable than dividing α̇ by α.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `t`  | `Tensor` | Time tensor. | *required* |

Returns:

| Type     | Description                                |
| -------- | ------------------------------------------ |
| `Tensor` | The ratio (which equals d_log_mean_coeff). |

Source code in `torchebm/interpolants/variance_preserving.py`

```python
def compute_d_alpha_alpha_ratio_t(self, t: torch.Tensor) -> torch.Tensor:
    r"""
    Compute \(\dot{\alpha}(t) / \alpha(t)\) directly from log mean coefficient.

    This is more numerically stable than dividing α̇ by α.

    Args:
        t: Time tensor.

    Returns:
        The ratio (which equals d_log_mean_coeff).
    """
    return self._d_log_mean_coeff(t)
```

### `compute_drift(x, t)`

Compute drift for VP schedule using the beta parameterization.

Parameters:

| Name | Type     | Description                               | Default    |
| ---- | -------- | ----------------------------------------- | ---------- |
| `x`  | `Tensor` | Current state of shape (batch_size, ...). | *required* |
| `t`  | `Tensor` | Time values of shape (batch_size,).       | *required* |

Returns:

| Type                    | Description                       |
| ----------------------- | --------------------------------- |
| `Tuple[Tensor, Tensor]` | Tuple of (drift_mean, drift_var). |

Source code in `torchebm/interpolants/variance_preserving.py`

```python
def compute_drift(
    self, x: torch.Tensor, t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute drift for VP schedule using the beta parameterization.

    Args:
        x: Current state of shape (batch_size, ...).
        t: Time values of shape (batch_size,).

    Returns:
        Tuple of (drift_mean, drift_var).
    """
    t_expanded = expand_t_like_x(t, x)
    beta_t = self.sigma_min + (1 - t_expanded) * (self.sigma_max - self.sigma_min)
    return -0.5 * beta_t * x, beta_t / 2
```

### `compute_sigma_t(t)`

Compute (\\sigma(t)) and its derivative for VP schedule.

Parameters:

| Name | Type     | Description  | Default    |
| ---- | -------- | ------------ | ---------- |
| `t`  | `Tensor` | Time tensor. | *required* |

Returns:

| Type                    | Description            |
| ----------------------- | ---------------------- |
| `Tuple[Tensor, Tensor]` | Tuple of (σ(t), σ̇(t)). |

Source code in `torchebm/interpolants/variance_preserving.py`

```python
def compute_sigma_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute \(\sigma(t)\) and its derivative for VP schedule.

    Args:
        t: Time tensor.

    Returns:
        Tuple of (σ(t), σ̇(t)).
    """
    p_sigma = 2 * self._log_mean_coeff(t)
    exp_p = torch.exp(p_sigma)
    sigma = torch.sqrt(torch.clamp(1 - exp_p, min=1e-12))
    d_sigma = exp_p * (2 * self._d_log_mean_coeff(t)) / (-2 * sigma)
    return sigma, d_sigma
```
