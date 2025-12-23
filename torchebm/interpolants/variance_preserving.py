r"""Variance preserving interpolant (DDPM-style schedule)."""

from typing import Tuple

import torch

from torchebm.core.base_interpolant import BaseInterpolant, expand_t_like_x


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


__all__ = ["VariancePreservingInterpolant"]
