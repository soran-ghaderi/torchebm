r"""Linear interpolant (optimal transport interpolant)."""

from typing import Tuple

import torch

from torchebm.core.base_interpolant import BaseInterpolant


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


__all__ = ["LinearInterpolant"]
