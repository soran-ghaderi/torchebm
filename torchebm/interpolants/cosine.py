r"""Cosine interpolant (geodesic variance preserving)."""

from typing import Tuple

import torch
import math

from torchebm.core.base_interpolant import BaseInterpolant


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


__all__ = ["CosineInterpolant"]
