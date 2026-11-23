r"""Mixed-field matching loss."""

from typing import Callable, Literal, Optional, Union

import torch

from torchebm.core import BaseLoss
from torchebm.losses.functional import mixed_field_matching_loss
from torchebm.losses.loss_utils import compute_ct

_CT = ("truncated", "linear", "constant")
_ENDPOINT_PENALTIES = ("in_batch", "separate", None)


class MixedFieldMatchingLoss(BaseLoss):
    r"""Mixed-field matching loss (Ghaderi et al., 2026).

    One field \(f_\theta(x, \tau)\) is a time-conditioned transport field for
    \(t \le a\) and a single time-independent refinement field for \(t > a\):
    the model receives the time input \(\tau_a(t) = \min(t, a)\), and the
    target uses the true \(t\),

    \[
    \mathcal{L} = \|f_\theta(x_t, \min(t, a)) - \lambda c_a(t)\, u_t\|^2,
    \qquad c_a(t) = \min\left(1, \frac{1 - t}{1 - a}\right),
    \]

    with \(u_t = x_1 - \epsilon\) on the linear path
    \(x_t = (1 - t)\epsilon + t x_1\). Below \(a\) this is flow matching with
    the target scaled by \(\lambda\); above \(a\) the target is
    \(\lambda (x_1 - x_t) / (1 - a)\). Sample with `MixedFieldSampler`.

    ``endpoint_penalty`` optionally adds \(w \|f_\theta(x_1, a)\|^2\), which
    sets the field to zero at clean examples so long refinement stays at the
    data: ``"in_batch"`` uses the last ``n_clean`` rows of `sample_times`
    (\(t = 1\)), and ``"separate"`` takes the field at clean examples as
    ``clean``.

    Args:
        ct: Target scaling \(c_a(t)\), multiplied by ``ct_multiplier``, as in
            `EquilibriumMatchingLoss`: ``"truncated"`` (default, above),
            ``"linear"`` (\(1 - t\)), ``"constant"`` (1), or a callable
            ``t -> c(t)`` on a (batch_size,) time tensor.
        ct_threshold: Threshold \(a\) between transport and refinement,
            strictly inside (0, 1).
        ct_multiplier: Target multiplier \(\lambda\).
        endpoint_penalty: ``None`` (default), ``"in_batch"``, or
            ``"separate"``.
        endpoint_weight: Weight \(w\) of the endpoint penalty.
        n_clean: Number of clean rows per batch for ``"in_batch"``; the paper
            uses 4 of 64 per device on ImageNet.
        **kwargs: Keyword arguments of `BaseLoss`: ``reduction`` (``"mean"``,
            ``"sum"``, or ``"none"``), ``dtype`` and ``device``.

    Example:
        MFM-XL/2 on ImageNet 256 (App. C): ``x1`` are sd-vae-ft-ema latents
        scaled by 0.18215, 64 per device on four devices, and ``y`` their
        labels. Sampling takes 230 Euler transport and 20 refinement steps
        with guidance 1.5 on the first three latent channels.

        ```python
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        from torchebm.interpolants import LinearInterpolant
        from torchebm.losses import MixedFieldMatchingLoss
        from torchebm.models import ClassifierFreeGuidance, dit_xl_2
        from torchebm.samplers import MixedFieldSampler

        model = dit_xl_2(input_size=32, in_channels=4, num_classes=1000, class_dropout_prob=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9999))
        path = LinearInterpolant()
        loss_fn = MixedFieldMatchingLoss(endpoint_penalty="in_batch", n_clean=4)

        t = loss_fn.sample_times(len(x1))
        x_t, u_t = path(torch.randn_like(x1), x1, t)
        loss = loss_fn(model(x_t, t.clamp(max=0.8), y=y), u_t, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update_parameters(model)

        guided = ClassifierFreeGuidance(
            ema.module.eval(), guidance_scale=1.5, null_condition=1000, guide_channels=3
        )
        sampler = MixedFieldSampler()
        samples = sampler(
            guided, torch.randn(64, 4, 32, 32), n_steps=230, n_refinement_steps=20,
            model_kwargs={"y": y},
        )
        ```

        With ``endpoint_penalty="separate"`` the loss call is
        ``loss_fn(model(x_t, t.clamp(max=0.8), y=y), u_t, t, clean=model(x1, torch.full_like(t, 0.8), y=y))``.
    """

    def __init__(
        self,
        ct: Union[
            Literal["truncated", "linear", "constant"],
            Callable[[torch.Tensor], torch.Tensor],
        ] = "truncated",
        ct_threshold: float = 0.8,
        ct_multiplier: float = 4.0,
        endpoint_penalty: Optional[Literal["in_batch", "separate"]] = None,
        endpoint_weight: float = 0.1,
        n_clean: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not callable(ct) and ct not in _CT:
            raise ValueError(
                f"ct must be one of {_CT} or a callable t -> c(t), got {ct!r}"
            )
        if not 0.0 < ct_threshold < 1.0:
            raise ValueError(f"ct_threshold must be in (0, 1), got {ct_threshold}")
        if endpoint_penalty not in _ENDPOINT_PENALTIES:
            raise ValueError(
                f"endpoint_penalty must be one of {_ENDPOINT_PENALTIES}, "
                f"got {endpoint_penalty!r}"
            )
        self.ct = ct
        self.ct_threshold = float(ct_threshold)
        self.ct_multiplier = float(ct_multiplier)
        self.endpoint_penalty = endpoint_penalty
        self.endpoint_weight = float(endpoint_weight)
        self.n_clean = int(n_clean)

    def sample_times(
        self,
        batch_size: int,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        r"""Training times, uniform on \([0, 1)\); with
        ``endpoint_penalty="in_batch"`` the last ``n_clean`` are 1, the clean
        examples.

        Args:
            batch_size: Number of times.
            device: Device of the result.
            dtype: Data type of the result.
            generator: RNG for the draws; the global RNG when ``None``.

        Returns:
            Times of shape (batch_size,).
        """
        t = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        if self.endpoint_penalty == "in_batch":
            t[batch_size - self.n_clean :] = 1.0
        return t

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        clean: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Mixed-field matching loss on a predicted field.

        Args:
            input: Predicted field \(f_\theta(x_t, \min(t, a))\), shape
                (batch_size, ...).
            target: Path velocity \(u_t\) (noise to data), same shape.
            t: Times, shape (batch_size,), from `sample_times`.
            clean: Field at clean examples \(f_\theta(x_1, a)\), same shape as
                ``input``; required by ``endpoint_penalty="separate"`` only.

        Returns:
            Scalar loss, or shape (batch_size,) for ``reduction="none"``.

        Raises:
            TypeError: If ``clean`` is missing for ``"separate"`` or given
                otherwise.
        """
        if (clean is None) == (self.endpoint_penalty == "separate"):
            raise TypeError(
                "clean is required by endpoint_penalty='separate' and not "
                "taken otherwise"
            )
        return mixed_field_matching_loss(
            input,
            target,
            t,
            compute_ct(t, self.ct, self.ct_threshold, self.ct_multiplier),
            clean,
            self.endpoint_penalty,
            self.endpoint_weight,
            self.reduction,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(ct={self.ct!r}, "
            f"ct_threshold={self.ct_threshold}, "
            f"ct_multiplier={self.ct_multiplier}, "
            f"endpoint_penalty={self.endpoint_penalty!r}, "
            f"endpoint_weight={self.endpoint_weight}, n_clean={self.n_clean})"
        )


__all__ = ["MixedFieldMatchingLoss"]
