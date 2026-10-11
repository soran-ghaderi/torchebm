r"""Variance exploding interpolant (NCSN / VE-SDE schedule)."""

import math
from typing import Tuple

import torch

from torchebm.core import BaseInterpolant, expand_t_like_x


class VarianceExplodingInterpolant(BaseInterpolant):
    r"""
    Variance exploding (VE) interpolant with a geometric noise schedule.

    The forward process of NCSN (Song and Ermon, 2019) and the VE-SDE of
    Song et al. (2021): data is never scaled and the noise level grows
    geometrically from \(\sigma_{\min}\) at the data end to \(\sigma_{\max}\)
    at the source end,

    \[
    \alpha(t) = 1, \qquad
    \sigma(t) = \sigma_{\min} \left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^{1 - t}.
    \]

    \(x_t = x_1 + \sigma(t)\, x_0\) is the perturbation kernel
    \(\mathcal{N}(x_1, \sigma(t)^2 I)\), so the ``"epsilon"`` target is
    denoising score matching at noise level \(\sigma(t)\). The source at
    \(t = 0\) is \(\mathcal{N}(0, \sigma_{\max}^2 I)\) once \(\sigma_{\max}\)
    dominates the data scale, as with the reference defaults.

    Args:
        sigma_min: Noise level at the data end \(t = 1\) (default: 0.01).
        sigma_max: Noise level at the source end \(t = 0\) (default: 50.0).
        prediction_type: Regression target returned by the call: ``"velocity"``
            (default), ``"epsilon"``, or ``"sample"``.

    Example:
        ```python
        from torchebm.interpolants import VarianceExplodingInterpolant
        import torch

        path = VarianceExplodingInterpolant(prediction_type="epsilon")
        x0 = torch.randn(32, 2)  # noise
        x1 = torch.randn(32, 2)  # data
        t = torch.rand(32)
        x_t, noise = path(x0, x1, t)
        ```
    """

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        *,
        prediction_type: str = "velocity",
    ):
        super().__init__(prediction_type=prediction_type)
        if not 0.0 < sigma_min < sigma_max:
            raise ValueError(
                f"need 0 < sigma_min < sigma_max, got {sigma_min} and {sigma_max}"
            )
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self._log_sigma_ratio = math.log(self.sigma_max / self.sigma_min)

    def compute_alpha_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Compute \(\alpha(t) = 1\) and \(\dot{\alpha}(t) = 0\)."""
        return torch.ones_like(t), torch.zeros_like(t)

    def compute_sigma_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Compute \(\sigma(t)\) and \(\dot{\sigma}(t) = -\log(\sigma_{\max}/\sigma_{\min})\, \sigma(t)\)."""
        sigma = self.sigma_min * torch.exp(self._log_sigma_ratio * (1 - t))
        return sigma, -self._log_sigma_ratio * sigma

    def velocity_to_score(
        self, velocity: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        r"""Score \(-x_0 / \sigma(t)\) of the perturbation kernel from \(u_t = \dot{\sigma}(t)\, x_0\)."""
        sigma, d_sigma = self.compute_sigma_t(expand_t_like_x(t, x))
        return -velocity / (sigma * d_sigma)

    def velocity_to_noise(
        self, velocity: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        r"""Noise \(x_0 = u_t / \dot{\sigma}(t)\)."""
        _, d_sigma = self.compute_sigma_t(expand_t_like_x(t, x))
        return velocity / d_sigma


__all__ = ["VarianceExplodingInterpolant"]
