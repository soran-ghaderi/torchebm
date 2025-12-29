r"""Base class for interpolant schedules."""

from abc import ABC, abstractmethod
from typing import Tuple

import torch


def expand_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""Expand time tensor to match spatial dimensions of x.

    Args:
        t: Time tensor of shape (batch_size,).
        x: Reference tensor of shape (batch_size, ...).

    Returns:
        Time tensor expanded to shape (batch_size, 1, 1, ...).
    """
    dims = [1] * (x.ndim - 1)
    return t.view(t.size(0), *dims)


class BaseInterpolant(ABC):
    r"""
    Abstract base class for stochastic interpolants.

    An interpolant defines a conditional probability path between a source
    distribution (typically Gaussian noise) and a target distribution (data).

    The interpolation is parameterized as:

    \[
    x_t = \alpha(t) x_1 + \sigma(t) x_0
    \]

    where \(x_0 \sim \mathcal{N}(0, I)\) and \(x_1 \sim p_{\text{data}}\).

    Subclasses must implement `compute_alpha_t` and `compute_sigma_t`.
    """

    @abstractmethod
    def compute_alpha_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute the data coefficient \(\alpha(t)\) and its time derivative.

        Args:
            t: Time tensor of shape (batch_size, ...).

        Returns:
            Tuple of (\(\alpha(t)\), \(\dot{\alpha}(t)\)).
        """
        raise NotImplementedError

    @abstractmethod
    def compute_sigma_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute the noise coefficient \(\sigma(t)\) and its time derivative.

        Args:
            t: Time tensor of shape (batch_size, ...).

        Returns:
            Tuple of (\(\sigma(t)\), \(\dot{\sigma}(t)\)).
        """
        raise NotImplementedError

    def compute_d_alpha_alpha_ratio_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the ratio \(\dot{\alpha}(t) / \alpha(t)\) for numerical stability.

        This method can be overridden for better numerical precision.

        Args:
            t: Time tensor.

        Returns:
            The ratio tensor.
        """
        alpha, d_alpha = self.compute_alpha_t(t)
        return d_alpha / torch.clamp(alpha, min=1e-8)

    def interpolate(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute the interpolated sample \(x_t\) and conditional velocity \(u_t\).

        Args:
            x0: Noise samples of shape (batch_size, ...).
            x1: Data samples of shape (batch_size, ...).
            t: Time values of shape (batch_size,).

        Returns:
            Tuple of (x_t, u_t) where:
                - x_t = α(t) x₁ + σ(t) x₀
                - u_t = α̇(t) x₁ + σ̇(t) x₀
        """
        t_expanded = expand_t_like_x(t, x0)
        alpha, d_alpha = self.compute_alpha_t(t_expanded)
        sigma, d_sigma = self.compute_sigma_t(t_expanded)

        xt = alpha * x1 + sigma * x0
        ut = d_alpha * x1 + d_sigma * x0

        return xt, ut

    def compute_drift(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute drift coefficients for score-based parameterization.

        For the probability flow ODE in score parameterization:
        dx = [-drift_mean + drift_var * score] dt

        Args:
            x: Current state of shape (batch_size, ...).
            t: Time values of shape (batch_size,).

        Returns:
            Tuple of (drift_mean, drift_var) for score parameterization.
        """
        t_expanded = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t_expanded)
        sigma, d_sigma = self.compute_sigma_t(t_expanded)

        drift_mean = alpha_ratio * x
        drift_var = alpha_ratio * (sigma**2) - sigma * d_sigma

        return -drift_mean, drift_var

    def compute_diffusion(
        self, x: torch.Tensor, t: torch.Tensor, form: str = "SBDM", norm: float = 1.0
    ) -> torch.Tensor:
        r"""
        Compute diffusion coefficient for SDE sampling.

        Args:
            x: Current state of shape (batch_size, ...).
            t: Time values of shape (batch_size,).
            form: Diffusion form. Choices:
                - 'constant': Constant diffusion
                - 'SBDM': Score-based diffusion matching
                - 'sigma': Proportional to noise schedule
                - 'linear': Linear decay
                - 'decreasing': Faster decay towards t=1
                - 'increasing-decreasing': Peak at midpoint
            norm: Scaling factor for diffusion.

        Returns:
            Diffusion coefficient tensor.
        """
        t_expanded = expand_t_like_x(t, x)
        sigma, _ = self.compute_sigma_t(t_expanded)
        _, drift_var = self.compute_drift(x, t)

        if form == "constant":
            return norm * torch.ones_like(drift_var)
        elif form == "SBDM":
            return norm * drift_var / (sigma + 1e-8)
        elif form == "sigma":
            return norm * sigma
        elif form == "linear":
            return norm * (1 - t_expanded)
        elif form == "decreasing":
            # Faster decay: (1-t)^2
            return norm * (1 - t_expanded) ** 2
        elif form == "increasing-decreasing":
            # Peak at t=0.5: 4*t*(1-t)
            return norm * 4 * t_expanded * (1 - t_expanded)
        else:
            raise ValueError(
                f"Unknown diffusion form '{form}'. "
                f"Choose from: constant, SBDM, sigma, linear, decreasing, increasing-decreasing"
            )

    def velocity_to_score(
        self, velocity: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Convert velocity prediction to score.

        Args:
            velocity: Predicted velocity of shape (batch_size, ...).
            x: Current state of shape (batch_size, ...).
            t: Time values of shape (batch_size,).

        Returns:
            Score tensor of shape (batch_size, ...).
        """
        t_expanded = expand_t_like_x(t, x)
        alpha, d_alpha = self.compute_alpha_t(t_expanded)
        sigma, d_sigma = self.compute_sigma_t(t_expanded)

        alpha = torch.clamp(alpha, min=1e-8)
        reverse_alpha_ratio = alpha / d_alpha
        var = sigma**2 - reverse_alpha_ratio * d_sigma * sigma
        score = (reverse_alpha_ratio * velocity - x) / torch.clamp(var, min=1e-12)

        return score

    def velocity_to_noise(
        self, velocity: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Convert velocity prediction to noise prediction.

        Args:
            velocity: Predicted velocity of shape (batch_size, ...).
            x: Current state of shape (batch_size, ...).
            t: Time values of shape (batch_size,).

        Returns:
            Noise tensor of shape (batch_size, ...).
        """
        t_expanded = expand_t_like_x(t, x)
        alpha, d_alpha = self.compute_alpha_t(t_expanded)
        sigma, d_sigma = self.compute_sigma_t(t_expanded)

        d_alpha = torch.where(d_alpha.abs() < 1e-8, torch.ones_like(d_alpha) * 1e-8, d_alpha)
        reverse_alpha_ratio = alpha / d_alpha
        var = sigma - reverse_alpha_ratio * d_sigma
        var = torch.where(var.abs() < 1e-12, torch.sign(var) * 1e-12 + (var == 0) * 1e-12, var)
        noise = (x - reverse_alpha_ratio * velocity) / var

        return noise

    def score_to_velocity(
        self, score: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Convert score prediction to velocity.

        Args:
            score: Predicted score of shape (batch_size, ...).
            x: Current state of shape (batch_size, ...).
            t: Time values of shape (batch_size,).

        Returns:
            Velocity tensor of shape (batch_size, ...).
        """
        drift_mean, drift_var = self.compute_drift(x, t)
        velocity = drift_var * score - drift_mean
        return velocity


__all__ = ["BaseInterpolant", "expand_t_like_x"]
