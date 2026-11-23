r"""Loss functions on tensors, the `torch.nn.functional` counterpart of `torchebm.losses`.

Each function takes model outputs and targets and returns a loss. None holds
state, calls a model, samples, or draws times; the loss classes compute those
inputs and call these. ``reduction`` follows `torch.nn.functional`: ``"mean"``
and ``"sum"`` reduce over the batch, ``"none"`` returns one value per sample.
"""

import math
from typing import Optional, Sequence

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


def mixed_field_matching_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    ct: torch.Tensor,
    clean: Optional[torch.Tensor] = None,
    endpoint_penalty: Optional[str] = None,
    endpoint_weight: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Mixed-field matching loss on a predicted field.

    \[
    L = \|f_\theta(x_t, \min(t, a)) - c(t)\, u_t\|^2
    \]

    with the target scaling ``ct``, \(\lambda c_a(t)\) by default (see
    `compute_ct`), and the squared error averaged over the non-batch
    dimensions. ``endpoint_penalty`` adds \(w \|f_\theta(x_1, a)\|^2\) at
    clean examples: for ``"in_batch"`` the rows with \(t = 1\), whose target
    is set to zero, with each term averaged over its own rows; for
    ``"separate"`` the field ``clean``.

    Args:
        input: Predicted field \(f_\theta(x_t, \min(t, a))\), shape
            (batch_size, ...).
        target: Path velocity \(u_t\) (noise to data), same shape.
        t: Times, shape (batch_size,).
        ct: Target scaling \(c(t)\), shape (batch_size,).
        clean: Field at clean examples \(f_\theta(x_1, a)\), same shape as
            ``input``, for ``endpoint_penalty="separate"``.
        endpoint_penalty: ``None``, ``"in_batch"``, or ``"separate"``.
        endpoint_weight: Weight \(w\) of the endpoint penalty.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``; ``"none"`` returns
            per-sample losses whose mean is the loss.

    Returns:
        Scalar loss, or shape (batch_size,) for ``reduction="none"``.
    """
    if endpoint_penalty == "in_batch":
        is_clean = t >= 1.0
        ct = ct.masked_fill(is_clean, 0.0)
    ct = ct.view(-1, *([1] * (input.ndim - 1)))
    error = mean_flat((input - ct * target).square())
    if endpoint_penalty is None:
        return _reduce(error, reduction)
    if endpoint_penalty == "separate":
        return _reduce(error + endpoint_weight * mean_flat(clean.square()), reduction)
    n = t.shape[0]
    n_clean = is_clean.sum(dtype=torch.promote_types(t.dtype, torch.float32))
    weight = torch.where(
        is_clean,
        endpoint_weight * n / n_clean.clamp_min(1.0),
        n / (n - n_clean).clamp_min(1.0),
    )
    return _reduce(error, reduction, weight)


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


def shortcut_loss(
    input: torch.Tensor,
    first_step: torch.Tensor,
    second_step: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    target_clip: Optional[float] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Shortcut model self-consistency: one jump of size \(2d\) matches two of size \(d\).

    \[
    L = w\, \big\|u_\theta(x_t, t, t + 2d) - \mathrm{sg}\big(\tfrac{1}{2}(u_1 + u_2)\big)\big\|^2
    \]

    with \(u_1 = u(x_t, t, t + d)\), \(u_2 = u(x_t + d\, u_1, t + d, t + 2d)\)
    and the squared error averaged over the non-batch dimensions.

    Args:
        input: Average velocity of the jump of size \(2d\), shape (batch_size, ...).
        first_step: Average velocity \(u_1\) of the first jump of size \(d\), same shape.
        second_step: Average velocity \(u_2\) of the second jump of size \(d\), same shape.
        weight: Per-sample weight \(w\), shape (batch_size,).
        target_clip: If set, the target is clamped to ``[-target_clip, target_clip]``.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss, or shape (batch_size,) for ``reduction="none"``.
    """
    target = (0.5 * (first_step + second_step)).detach()
    if target_clip is not None:
        target = target.clamp(-target_clip, target_clip)
    return _reduce(mean_flat((input - target).square()), reduction, weight)


def adaptive_weight(
    squared_error: torch.Tensor, p: float = 1.0, c: float = 0.01
) -> torch.Tensor:
    r"""Adaptive loss weight of MeanFlow (Geng et al., 2025), detached.

    \[
    w = \mathrm{sg}\big(1 / (\|\Delta\|^2 + c)^p\big)
    \]

    Multiplying the per-sample squared error by \(w\) gives the MeanFlow
    loss (Eq. 22); ``p=0.5`` is similar to the pseudo-Huber metric and
    ``p=0`` is the squared error. The defaults are those of the MeanFlow
    code (``norm_p=1.0``, ``norm_eps=0.01``); the paper uses \(p = 1\) on
    ImageNet and \(p = 0.75\) on CIFAR-10, and gives \(c = 10^{-3}\) as an
    example.

    Args:
        squared_error: Per-sample \(\|\Delta\|^2\), summed over the non-batch
            dimensions, shape (batch_size,).
        p: Power \(p\).
        c: Constant \(c\).

    Returns:
        Weight of shape (batch_size,), without gradient.

    Example:
        ```python
        squared_error = (u - u_target).square().flatten(1).sum(1)
        loss = (adaptive_weight(squared_error) * squared_error).mean()
        ```
    """
    return (squared_error.detach() + c).pow(-p)


def meanflow_loss(
    input: torch.Tensor,
    du_ds: torch.Tensor,
    velocity: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    p: float = 1.0,
    c: float = 0.01,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    aux_input: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""MeanFlow loss (Geng et al., 2025, Eq. 9 and Eq. 22) in torchebm time.

    \[
    L = w\, \mathrm{sg}(\lambda)\, \big\|u_\theta(x_s, s, t)
    - \mathrm{sg}\big(v + (t - s)\, \tfrac{d}{ds} u_\theta\big)\big\|^2,
    \qquad \lambda = 1 / (\|\Delta\|^2 + c)^p,
    \]

    with \(\Delta\) the regression error, the squared norm summed over the
    non-batch dimensions as in the MeanFlow code, and
    \(\tfrac{d}{ds} u_\theta = v\, \partial_x u_\theta + \partial_s u_\theta\)
    the JVP along \((v, 1)\) in \((x_s, s)\) at fixed \(t\). With noise at
    \(t = 0\) the paper's identity \(u = v - (t - r)\, \tfrac{d}{dt} u\)
    becomes \(u = v + (t - s)\, \tfrac{d}{ds} u\). Improved MeanFlow
    (Geng et al., 2025b, Algorithm 1) computes the same loss with the JVP
    along the model velocity \(u_\theta(x_s, s, s)\) instead of \(v\).
    ``p=0`` gives the squared error.

    With ``aux_input``, the output \(v_\theta\) of an auxiliary velocity head
    adds the flow matching loss \(\mathrm{sg}(\lambda_v)\, \|v_\theta - v\|^2\)
    with its own adaptive weight, summed per sample as in the Improved
    MeanFlow code (App. A).

    Args:
        input: Average velocity \(u_\theta(x_s, s, t)\), shape (batch_size, ...).
        du_ds: Total derivative \(\tfrac{d}{ds} u_\theta\), same shape.
        velocity: Conditional velocity \(v\) of the path, or the guided
            velocity (Eq. 21) under classifier-free guidance, same shape.
        s: Times of the input, shape (batch_size,).
        t: Target times, shape (batch_size,).
        p: Power \(p\) of the adaptive weight.
        c: Constant \(c\) of the adaptive weight.
        weight: Per-sample weight \(w\), shape (batch_size,).
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
        aux_input: Output \(v_\theta\) of an auxiliary velocity head, same
            shape as ``input``.

    Returns:
        Scalar loss, or shape (batch_size,) for ``reduction="none"``.
    """
    interval = (t - s).to(input.dtype).reshape(-1, *(1,) * (input.ndim - 1))
    target = (velocity + interval * du_ds).detach()
    squared_error = (input - target).square().flatten(1).sum(1)
    loss = adaptive_weight(squared_error, p, c) * squared_error
    if aux_input is not None:
        aux_error = (aux_input - velocity.detach()).square().flatten(1).sum(1)
        loss = loss + adaptive_weight(aux_error, p, c) * aux_error
    return _reduce(loss, reduction, weight)


def pseudo_huber_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    c: Optional[float] = None,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Pseudo-Huber metric of improved consistency training (Song and Dhariwal, 2023, Eq. 8).

    \[
    L = w \left(\sqrt{\|x - y\|^2 + c^2} - c\right)
    \]

    with the squared norm summed over the non-batch dimensions. It is close
    to \(\|x - y\|^2 / (2c)\) for small errors and to \(\|x - y\|\) for large
    ones.

    Args:
        input: Model output \(x\), shape (batch_size, ...).
        target: Target \(y\), same shape.
        c: Constant \(c\); ``None`` uses the iCT choice
            \(0.00054 \sqrt{d}\) with \(d\) the number of non-batch elements.
        weight: Per-sample weight \(w\), shape (batch_size,).
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss, or shape (batch_size,) for ``reduction="none"``.
    """
    if c is None:
        c = 0.00054 * math.sqrt(input[0].numel())
    squared_norm = (input - target).square().flatten(1).sum(1)
    return _reduce(torch.sqrt(squared_norm + c**2) - c, reduction, weight)


def drifting_loss(
    input: torch.Tensor,
    positive: torch.Tensor,
    negative: Optional[torch.Tensor] = None,
    negative_weight: Optional[torch.Tensor] = None,
    temperatures: Sequence[float] = (0.02, 0.05, 0.2),
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Drifting loss of one feature (Deng et al., 2026, Eq. 6 and App. A.6).

    \[
    L = \big\|\tilde\phi(x) - \mathrm{sg}\big(\tilde\phi(x)
        + \textstyle\sum_\tau \tilde V_\tau(\tilde\phi(x))\big)\big\|^2
    \]

    averaged over the feature dimensions. The drifting field
    \(V = V^+_p - V^-_q\) (Eqs. 8 to 11) attracts each generated sample to the
    positives and repels it from the negatives: the generated samples of its
    group, its own one masked, and the unconditional negatives. The kernel
    \(\exp(-\|x - y\| / \tau)\) (Eq. 12) is normalized by a softmax over the
    positives and negatives and a softmax over the generated samples, and the
    weights take their geometric mean with a floor of \(10^{-3}\). The floor
    weights every pair, so groups of thousands of samples flatten the kernel;
    the paper uses 64 generated and up to 128 positive samples per group
    (Table 8). Features
    are divided by the scalar \(S\) that sets the mean distance between
    generated samples and all others to \(\sqrt{D}\), with the negative
    weights in the mean, and the field of each temperature by its root mean
    square \(\lambda_\tau\) over all groups and locations. \(S\), the field
    and the target are computed without gradients and in float32 or wider, as in the
    reference code (lambertae/drifting, ``drift_loss.py``); its loss value is
    the mean of \(\|\sum_\tau \tilde V_\tau\|^2 / D\). \(S\) and
    \(\lambda_\tau\) are statistics of the local batch; the reference code
    takes them over the global batch.

    Args:
        input: Features \(\phi(x)\) of the generated samples, shape
            (groups, n, ..., D): groups of ``n`` samples with one field each,
            such as one group per class label, then any location dimensions.
        positive: Features of the positive samples (data of each group's
            class), shape (groups, n_pos, ..., D).
        negative: Features of the unconditional negatives (data of random
            classes) for classifier-free guidance, shape
            (groups, n_neg, ..., D).
        negative_weight: Kernel weight \(w\) of the unconditional negatives
            of each group, shape (groups,); the guidance scale
            \(\alpha\) sets \(w = (\alpha - 1)(n - 1) / n_{neg}\) (App. A.7).
            Weight 1 when ``None``.
        temperatures: Temperatures \(\tau\) of the kernel; the fields of all
            temperatures are summed.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss, or shape (groups, n) for ``reduction="none"``.
    """
    n = input.shape[1]
    n_pos = positive.shape[1]
    dtype = torch.promote_types(input.dtype, torch.float32)
    with torch.no_grad(), torch.autocast(input.device.type, enabled=False):
        x = input.detach().movedim(1, -2).to(dtype)
        y_pos = positive.movedim(1, -2).to(dtype)
        y_neg = x if negative is None else torch.cat([x, negative.movedim(1, -2).to(dtype)], dim=-2)
        targets = torch.cat([y_neg, y_pos], dim=-2)
        squared_dist = (
            x.square().sum(-1, keepdim=True)
            + targets.square().sum(-1).unsqueeze(-2)
            - 2 * x @ targets.mT
        )
        dist = squared_dist.clamp_min_(1e-8).sqrt_()
        if negative is None or negative_weight is None:
            weight = None
            scale = dist.mean()
        else:
            weight = dist.new_ones(dist.shape[0], *(1,) * (dist.ndim - 2), dist.shape[-1])
            weight[..., n : y_neg.shape[-2]] = negative_weight.to(dtype).reshape(-1, *(1,) * (dist.ndim - 1))
            scale = (dist.mean(-2, keepdim=True) * weight).mean() / weight.mean()
        input_scale = (scale / math.sqrt(x.shape[-1])).clamp_min(1e-3)
        dist.div_(scale.clamp_min(1e-3))
        dist[..., :n].diagonal(dim1=-2, dim2=-1).add_(100.0)
        drift = torch.zeros_like(x)
        for tau in temperatures:
            logits = dist / -tau
            row_lse = logits.logsumexp(-1, keepdim=True)
            col_lse = logits.logsumexp(-2, keepdim=True)
            affinity = logits.sub_(0.5 * row_lse).sub_(0.5 * col_lse).clamp_min_(math.log(1e-3)).exp_()
            if weight is not None:
                affinity.mul_(weight)
            a_neg, a_pos = affinity.split([affinity.shape[-1] - n_pos, n_pos], dim=-1)
            field = (
                a_neg.sum(-1, keepdim=True) * (a_pos @ y_pos)
                - a_pos.sum(-1, keepdim=True) * (a_neg @ y_neg)
            ) / input_scale
            drift += field / field.square().mean().clamp_min(1e-8).sqrt()
        target = x / input_scale + drift
    error = (input.movedim(1, -2).to(dtype) / input_scale - target).square().mean(-1)
    return _reduce(error.reshape(error.shape[0], -1, n).mean(1), reduction)


__all__ = [
    "contrastive_divergence_loss",
    "flow_matching_loss",
    "equilibrium_matching_loss",
    "mixed_field_matching_loss",
    "energy_matching_loss",
    "score_matching_loss",
    "denoising_score_matching_loss",
    "sliced_score_matching_loss",
    "shortcut_loss",
    "adaptive_weight",
    "meanflow_loss",
    "pseudo_huber_loss",
    "drifting_loss",
]
