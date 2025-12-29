r"""Utility functions for loss computations."""

import torch

from torchebm.interpolants import (
    BaseInterpolant,
    LinearInterpolant,
    CosineInterpolant,
    VariancePreservingInterpolant,
)


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    r"""Take mean over all non-batch dimensions.

    Args:
        tensor: Input tensor of shape (batch_size, ...).

    Returns:
        Tensor of shape (batch_size,) with mean over spatial dims.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def get_interpolant(interpolant_type: str) -> BaseInterpolant:
    r"""Get interpolant instance by name.

    Args:
        interpolant_type: One of 'linear', 'cosine', or 'vp'.

    Returns:
        Interpolant instance.

    Raises:
        ValueError: If interpolant_type is not recognized.
    """
    interpolants = {
        "linear": LinearInterpolant,
        "cosine": CosineInterpolant,
        "vp": VariancePreservingInterpolant,
    }
    if interpolant_type not in interpolants:
        raise ValueError(
            f"Unknown interpolant: {interpolant_type}. "
            f"Choose from {list(interpolants.keys())}"
        )
    return interpolants[interpolant_type]()


def compute_eqm_ct(
    t: torch.Tensor,
    threshold: float = 0.8,
    multiplier: float = 4.0,
) -> torch.Tensor:
    r"""Energy-compatible target scaling c(t) used in EqM.

    The scaling function (truncated decay with gradient multiplier) is:

    \[
    c(t) = \lambda \cdot \min\left(1, \frac{1 - t}{1 - a}\right)
    \]

    where \(a\) is the threshold and \(\lambda\) is the multiplier.

    Args:
        t: Time tensor of shape (batch_size,).
        threshold: Decay threshold \(a\), decay starts after \(t > a\). Default: 0.8.
        multiplier: Gradient multiplier \(\lambda\). Default: 4.0.

    Returns:
        Scaling factor c(t) of same shape as t.
    """
    start = 1.0
    ct = (
        torch.minimum(
            start - (start - 1) / threshold * t,
            1 / (1 - threshold) - 1 / (1 - threshold) * t,
        )
        * multiplier
    )
    return ct


def dispersive_loss(z: torch.Tensor) -> torch.Tensor:
    r"""Dispersive loss (InfoNCE-L2 variant) for regularization.

    Encourages diversity in generated samples by penalizing samples
    that are too close to each other in feature space.

    Args:
        z: Feature tensor of shape (batch_size, ...).

    Returns:
        Scalar dispersive loss.
    """
    z = z.reshape((z.shape[0], -1))
    diff = torch.nn.functional.pdist(z).pow(2) / z.shape[1]
    diff = torch.concat((diff, diff, torch.zeros(z.shape[0], device=z.device)))
    return torch.log(torch.exp(-diff).mean())


__all__ = [
    "mean_flat",
    "get_interpolant",
    "compute_eqm_ct",
    "dispersive_loss",
]
