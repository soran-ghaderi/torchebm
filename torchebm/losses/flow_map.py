r"""Flow-map losses for one-step and few-step generation."""

from typing import Optional

import torch

from torchebm.core import BaseLoss
from torchebm.losses.functional import shortcut_loss


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
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
        dtype: Data type for computations.
        device: Device for computations.

    Example:
        One training step of the reference code on a class-conditional `DiT`
        with ``prediction_type="average_velocity"`` over ImageNet 256
        latents: \(M = 128\), one eighth of the batch on the self-consistency
        loss with big jumps \(2d \in \{1, 1/2, \dots, 1/64\}\) and \(t\) on
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
