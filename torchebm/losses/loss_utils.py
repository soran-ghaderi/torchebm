r"""Utility functions for loss computations."""

import torch

from torchebm.core import BaseInterpolant
from torchebm.interpolants import (
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


def trimmed_mean(values: torch.Tensor, trim_fraction: float) -> torch.Tensor:
    r"""One-sided trimmed mean: drop the largest values, average the rest.

    Used to robustify the negative-energy statistic in Energy Matching:
    far-from-manifold negatives with outlier energies are discarded before
    averaging.

    Args:
        values: 1-D tensor of shape (n,).
        trim_fraction: Fraction of the highest values to drop, in [0, 1).
            `int(trim_fraction * n)` entries are removed.

    Returns:
        Scalar tensor: mean of the kept values.

    Raises:
        ValueError: If trim_fraction is outside [0, 1).
    """
    if not 0.0 <= trim_fraction < 1.0:
        raise ValueError(f"trim_fraction must be in [0, 1), got {trim_fraction}")
    n = values.shape[0]
    k = int(trim_fraction * n)
    if k == 0:
        return values.mean()
    return values.sort().values[: n - k].mean()


def compute_flow_weight(t: torch.Tensor, cutoff: float = 0.8) -> torch.Tensor:
    r"""Time gate for the flow-matching term in Energy Matching.

    Full weight in the transport window, linear decay to zero at \(t = 1\):

    \[
    w(t) = \min\left(1, \max\left(0, \frac{1 - t}{1 - a}\right)\right)
    \]

    where \(a\) is the cutoff. A cutoff >= 1 disables gating (all ones).

    Args:
        t: Time tensor of shape (batch_size,).
        cutoff: Decay onset \(a\). Default: 0.8.

    Returns:
        Weight tensor of the same shape as t.
    """
    if cutoff >= 1.0:
        return torch.ones_like(t)
    return ((1.0 - t) / (1.0 - cutoff)).clamp(min=0.0, max=1.0)


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
    diff = torch.nn.functional.pdist(z).square() / z.shape[1]
    diff = torch.concat((diff, diff, torch.zeros(z.shape[0], device=z.device)))
    return torch.log(torch.exp(-diff).mean())


__all__ = [
    "mean_flat",
    "get_interpolant",
    "trimmed_mean",
    "compute_flow_weight",
    "compute_eqm_ct",
    "dispersive_loss",
]
