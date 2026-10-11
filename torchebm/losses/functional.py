r"""Loss functions on tensors, the `torch.nn.functional` counterpart of `torchebm.losses`.

Each function takes model outputs and targets and returns a loss. None holds
state, calls a model, samples, or draws times; the loss classes compute those
inputs and call these. ``reduction`` follows `torch.nn.functional`: ``"mean"``
and ``"sum"`` reduce over the batch, ``"none"`` returns one value per sample.
"""

from typing import Optional

import torch

from torchebm.losses.loss_utils import compute_flow_weight, mean_flat, trimmed_mean


def _reduce(
    loss: torch.Tensor, reduction: str, weight: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if weight is not None:
        loss = loss * weight
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")


def contrastive_divergence_loss(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    energy_reg_weight: float = 0.0,
    trim_fraction: float = 0.0,
) -> torch.Tensor:
    r"""Contrastive divergence from the energies of data and model samples.

    \[
    L = \mathbb{E}[E(x^+)] - \widetilde{\mathbb{E}}[E(x^-)]
        + \lambda \left(\mathbb{E}[E(x^+)^2] + \mathbb{E}[E(x^-)^2]\right)
    \]

    where \(\widetilde{\mathbb{E}}\) drops the ``trim_fraction`` largest
    negative energies (the robust statistic of Energy Matching) and
    \(\lambda\) is the squared-energy regularization weight.

    Args:
        positive: Energies of data samples, shape (batch_size,).
        negative: Energies of model samples, shape (batch_size,).
        energy_reg_weight: Weight \(\lambda\) of the squared-energy penalty.
        trim_fraction: Fraction of the largest negative energies to drop, in [0, 1).

    Returns:
        Scalar loss.
    """
    loss = positive.mean() - trimmed_mean(negative, trim_fraction)
    if energy_reg_weight > 0:
        loss = loss + energy_reg_weight * (
            positive.square().mean() + negative.square().mean()
        )
    return loss


def flow_matching_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Conditional flow matching: squared error to the path velocity.

    \[
    L = w\, \|v_\theta(x_t, t) - u_t\|^2
    \]

    with the squared error averaged over the non-batch dimensions.

    Args:
        input: Predicted velocity, shape (batch_size, ...).
        target: Conditional velocity \(u_t\), same shape.
        weight: Per-sample weight \(w\), shape (batch_size,).
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss, or shape (batch_size,) for ``reduction="none"``.
    """
    return _reduce(mean_flat((input - target).square()), reduction, weight)


def equilibrium_matching_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    ct: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Equilibrium matching: squared error to the \(c(t)\)-scaled target.

    \[
    L = w\, \|f_\theta(x_t) - c(t)\, u\|^2
    \]

    with the squared error averaged over the non-batch dimensions.

    Args:
        input: Predicted equilibrium field, shape (batch_size, ...).
        target: Direction regressed onto before scaling, same shape.
        ct: Scale \(c(t)\), shape (batch_size,); see `compute_eqm_ct`.
        weight: Per-sample weight \(w\), shape (batch_size,).
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss, or shape (batch_size,) for ``reduction="none"``.
    """
    ct = ct.view(-1, *([1] * (input.ndim - 1)))
    return _reduce(mean_flat((input - target * ct).square()), reduction, weight)


def energy_matching_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    *,
    flow_weight_cutoff: float = 0.8,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Energy matching flow term: time-gated regression of \(-\nabla V\) onto the displacement.

    \[
    L_{OT} = w(t)\, \|-\nabla_x V_\theta(x_t) - u_t\|^2, \qquad
    w(t) = \min\left(1, \max\left(0, \frac{1 - t}{1 - a}\right)\right)
    \]

    (Balcerak et al., 2025). The contrastive term of the joint phase is
    `contrastive_divergence_loss` with its ``trim_fraction``.

    Args:
        input: \(-\nabla_x V_\theta(x_t)\), shape (batch_size, ...).
        target: Displacement \(u_t\), same shape.
        t: Times, shape (batch_size,).
        flow_weight_cutoff: Gate onset \(a\); a value >= 1 disables the gate.
        weight: Extra per-sample weight, shape (batch_size,).
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss, or shape (batch_size,) for ``reduction="none"``.
    """
    gate = compute_flow_weight(t, cutoff=flow_weight_cutoff)
    if weight is not None:
        gate = gate * weight
    return flow_matching_loss(input, target, weight=gate, reduction=reduction)


def score_matching_loss(
    input: torch.Tensor, divergence: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    r"""Score matching (Hyvärinen, 2005) from the score and its divergence.

    \[
    L = \tfrac{1}{2}\|s_\theta(x)\|^2 + \nabla_x \cdot s_\theta(x)
    \]

    Args:
        input: Score \(s_\theta(x) = \nabla_x \log p_\theta(x)\), shape (batch_size, ...).
        divergence: \(\nabla_x \cdot s_\theta(x)\), shape (batch_size,).
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss, or shape (batch_size,) for ``reduction="none"``.
    """
    squared_norm = input.square().sum(dim=list(range(1, input.ndim)))
    return _reduce(0.5 * squared_norm + divergence, reduction)


def denoising_score_matching_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Denoising score matching (Vincent, 2011): half squared error to the kernel score.

    \[
    L = \tfrac{1}{2}\, w\, \|s_\theta(\tilde x) - \nabla_{\tilde x} \log q(\tilde x \mid x)\|^2
    \]

    For Gaussian noise \(\tilde x = x + \sigma \epsilon\) the kernel score is
    \(-\epsilon / \sigma\) (\(+\epsilon / \sigma\) for an energy gradient), and
    \(w = \sigma^2\) is the noise-level weighting of Song and Ermon (2019).

    Args:
        input: Score at the perturbed input, shape (batch_size, ...).
        target: Score of the perturbation kernel, same shape.
        weight: Per-sample weight \(w\), shape (batch_size,).
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss, or shape (batch_size,) for ``reduction="none"``.
    """
    squared_error = (input - target).square().sum(dim=list(range(1, input.ndim)))
    return _reduce(0.5 * squared_error, reduction, weight)


def sliced_score_matching_loss(
    input: torch.Tensor,
    x: torch.Tensor,
    projections: torch.Tensor,
    *,
    n_projections: int,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Sliced score matching (Song et al., 2019) along random projections.

    \[
    L = \mathbb{E}_v\left[ v^\top \nabla_x \big(v^\top s_\theta(x)\big)
        + \tfrac{1}{2} \big(v^\top s_\theta(x)\big)^2 \right]
    \]

    The directional derivative is taken by autograd through ``input``, which
    must have been computed from ``x`` with ``create_graph=True``.

    Args:
        input: Score at the projection-tiled inputs, shape
            (n_projections * batch_size, ...).
        x: The tiled inputs ``input`` was computed from, same shape, requiring grad.
        projections: Projection directions \(v\), same shape.
        n_projections: Number of projections per sample.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss, or shape (batch_size,) for ``reduction="none"``.
    """
    dims = list(range(1, input.ndim))
    projected_score = torch.sum(input * projections, dim=dims)
    directional = torch.autograd.grad(projected_score.sum(), x, create_graph=True)[0]
    term1 = 0.5 * projected_score.square()
    term2 = torch.sum(projections * directional, dim=dims)
    loss = term2.view(n_projections, -1).mean(dim=0) + term1.view(
        n_projections, -1
    ).mean(dim=0)
    return _reduce(loss, reduction)


__all__ = [
    "contrastive_divergence_loss",
    "flow_matching_loss",
    "equilibrium_matching_loss",
    "energy_matching_loss",
    "score_matching_loss",
    "denoising_score_matching_loss",
    "sliced_score_matching_loss",
]
