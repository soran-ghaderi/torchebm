r"""Flow-map losses for one-step and few-step generation."""

from typing import Optional

import torch

from torchebm.core import BaseLoss
from torchebm.losses.functional import meanflow_loss, shortcut_loss


class ShortcutLoss(BaseLoss):
    r"""Self-consistency loss of shortcut models (Frans et al., 2024).

    A shortcut model is a two-time model that outputs the average velocity
    \(u(x_t, t, t + d)\) of a jump of size \(d\) (``prediction_type=
    "average_velocity"``). One jump of size \(2d\) must match two jumps of
    size \(d\):

    \[
    L = \big\|u_\theta(x_t, t, t + 2d)
    - \mathrm{sg}\big(\tfrac{1}{2}(u_1 + u_2)\big)\big\|^2,
    \]

    with \(u_1 = u(x_t, t, t + d)\) and
    \(u_2 = u(x_t + d\, u_1, t + d, t + 2d)\). The caller computes the two
    small jumps, without gradients and in the reference code with the EMA
    model; the loss takes the three outputs. The term at \(d = 0\) is
    `FlowMatchingLoss` on the instantaneous velocity \(u(x_t, t, t)\); the
    reference code conditions it on the smallest jump \(1/M\) instead. No
    JVP is needed. Sample with `FlowMapSampler`. See `shortcut_loss`.

    Args:
        target_clip: If set, the target is clamped to
            ``[-target_clip, target_clip]``; the reference code uses 4.
        **kwargs: Keyword arguments of `BaseLoss`: ``reduction`` (``"mean"``,
            ``"sum"``, or ``"none"``), ``dtype`` and ``device``.

    Example:
        One training step of the reference code (kvfrans/shortcut-models,
        ``targets_shortcut.py``) on a class-conditional `DiT` with
        ``prediction_type="average_velocity"`` over ImageNet 256 latents:
        \(M = 128\), one eighth of the batch on the self-consistency loss
        (``bootstrap_every=8``; the paper states \(k = 1/4\)) with big jumps \(2d \in \{1, 1/2, \dots, 1/64\}\) and \(t\) on
        the grid of \(2d\), targets from the EMA model (decay 0.999), the
        intermediate state and the target clamped to \([-4, 4]\), and the
        flow-matching term on the rest at \(t \in \{0, 1/128, \dots\}\):

        ```python
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        from torchebm.interpolants import LinearInterpolant
        from torchebm.losses import FlowMatchingLoss, ShortcutLoss

        path = LinearInterpolant()
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999)).eval()
        loss_fn = ShortcutLoss(target_clip=4.0, reduction="none")
        fm_loss_fn = FlowMatchingLoss(reduction="none")

        n = len(x1) // 8
        k = torch.arange(6, -1, -1).repeat_interleave(n // 7)
        k = torch.cat([k, k.new_zeros(n - len(k))])
        d = 0.5 ** (k + 1)
        t = torch.floor(torch.rand(n) * 2.0**k) / 2.0**k
        x0 = torch.randn_like(x1)
        x_t, _ = path(x0[:n], x1[:n], t)
        with torch.no_grad():
            u1 = ema(x_t, t + d, y[:n], s=t)
            x_mid = (x_t + d.view(-1, 1, 1, 1) * u1).clamp(-4, 4)
            u2 = ema(x_mid, t + 2 * d, y[:n], s=t + d)
        t_fm = torch.randint(128, (len(x1) - n,)) / 128
        x_fm, v = path(x0[n:], x1[n:], t_fm)
        loss = torch.cat([
            loss_fn(model(x_t, t + 2 * d, y[:n], s=t), u1, u2),
            fm_loss_fn(model(x_fm, t_fm + 1 / 128, y[n:], s=t_fm), v),
        ]).mean()
        ```
    """

    def __init__(self, target_clip: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.target_clip = target_clip

    def forward(
        self,
        input: torch.Tensor,
        first_step: torch.Tensor,
        second_step: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Compute the loss from the big jump and the two small jumps.

        Args:
            input: Average velocity of the jump of size \(2d\),
                \(u_\theta(x_t, t, t + 2d)\), shape (batch_size, ...).
            first_step: Average velocity \(u_1\) of the first jump of size
                \(d\), same shape.
            second_step: Average velocity \(u_2\) of the second jump of size
                \(d\), same shape.
            weight: Per-sample weight, shape (batch_size,).

        Returns:
            Scalar loss, or shape (batch_size,) for ``reduction="none"``.
        """
        return shortcut_loss(
            input,
            first_step,
            second_step,
            weight=weight,
            target_clip=self.target_clip,
            reduction=self.reduction,
        )


class MeanFlowLoss(BaseLoss):
    r"""MeanFlow loss for one-step generation (Geng et al., 2025).

    A MeanFlow model is a two-time model that outputs the average velocity
    \(u(x_s, s, t) = \frac{1}{t - s} \int_s^t v(x_\tau, \tau)\, d\tau\)
    (``prediction_type="average_velocity"``). Differentiating
    \((t - s)\, u\) in \(s\) gives the MeanFlow identity in torchebm time,
    \(u = v + (t - s)\, \tfrac{d}{ds} u\), and the loss regresses the model
    on its detached right-hand side with the adaptive weight of Eq. 22:

    \[
    L = \mathrm{sg}(\lambda)\, \big\|u_\theta(x_s, s, t)
    - \mathrm{sg}\big(v + (t - s)\, \tfrac{d}{ds} u_\theta\big)\big\|^2,
    \qquad \lambda = 1 / (\|\Delta\|^2 + c)^p.
    \]

    The caller computes \(u_\theta\) and \(\tfrac{d}{ds} u_\theta\) with
    `average_velocity_jvp`, the JVP along the tangent \((v, 1)\) in
    \((x_s, s)\), run only on the pairs with \(s < t\); `DiT` attention then
    uses its JVP kernel. Pairs with \(s = t\) reduce to flow
    matching. Improved MeanFlow (Geng et al., 2025b, Algorithm 1) takes the
    tangent \((u_\theta(x_s, s, s), 1)\), computed without gradient, and the
    loss is unchanged. Sample with `FlowMapSampler`. See `meanflow_loss`.

    Args:
        p: Power \(p\) of the adaptive weight; ``0`` gives the squared error.
        c: Constant \(c\) of the adaptive weight.
        **kwargs: Keyword arguments of `BaseLoss`: ``reduction`` (``"mean"``,
            ``"sum"``, or ``"none"``), ``dtype`` and ``device``.

    Example:
        One training step of MeanFlow-XL/2 (Gsunshine/meanflow, Table 4) on
        a class-conditional `DiT` with ``prediction_type="average_velocity"``
        over ImageNet 256 latents: 25% of the pairs with \(s \neq t\) from
        a logit-normal (the defaults of `sample_time_pairs`), \(p = 1\),
        classifier-free guidance in the target (Eq. 21, \(\omega = 0.2\),
        \(\kappa = 0.92\)) for \(s \geq 0.25\), and 10% of the labels
        dropped at the start of the batch, as the code does:

        ```python
        from torchebm.interpolants import LinearInterpolant
        from torchebm.losses import MeanFlowLoss
        from torchebm.utils import average_velocity_jvp, sample_time_pairs

        path = LinearInterpolant()
        loss_fn = MeanFlowLoss()

        s, t = sample_time_pairs(len(x1))
        x_s, v = path(torch.randn_like(x1), x1, s)
        null = torch.full_like(y, 1000)
        with torch.no_grad():
            v_cond = model(x_s, s, y, s=s)
            v_uncond = model(x_s, s, null, s=s)
        guided = (s >= 0.25).view(-1, 1, 1, 1)
        v_guided = torch.where(guided, 0.2 * v + 0.92 * v_cond - 0.12 * v_uncond, v)
        drop = torch.arange(len(y)) < (torch.rand(len(y)) < 0.1).sum()
        y = torch.where(drop, null, y)
        v_guided = torch.where(drop.view(-1, 1, 1, 1), v, v_guided)
        u, du_ds = average_velocity_jvp(model, x_s, s, t, v_guided, y)
        loss = loss_fn(u, du_ds, v_guided, s, t)
        ```
    """

    def __init__(self, p: float = 1.0, c: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.c = c

    def forward(
        self,
        input: torch.Tensor,
        du_ds: torch.Tensor,
        velocity: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Compute the loss from the model output and its time derivative.

        Args:
            input: Average velocity \(u_\theta(x_s, s, t)\), shape
                (batch_size, ...).
            du_ds: Total derivative \(\tfrac{d}{ds} u_\theta\) from
                `average_velocity_jvp`, same shape.
            velocity: Conditional velocity \(v\) of the path, or the guided
                velocity under classifier-free guidance, same shape.
            s: Times of the input, shape (batch_size,).
            t: Target times, shape (batch_size,).
            weight: Per-sample weight, shape (batch_size,).

        Returns:
            Scalar loss, or shape (batch_size,) for ``reduction="none"``.
        """
        return meanflow_loss(
            input,
            du_ds,
            velocity,
            s,
            t,
            p=self.p,
            c=self.c,
            weight=weight,
            reduction=self.reduction,
        )
