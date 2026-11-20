r"""Drifting loss for one-step generators."""

from typing import List, Optional, Sequence, Union

import torch

from torchebm.core import BaseLoss
from torchebm.losses.functional import drifting_loss

Features = Union[torch.Tensor, Sequence[torch.Tensor]]


def _std(x: torch.Tensor, dim) -> torch.Tensor:
    x = x.to(torch.promote_types(x.dtype, torch.float32))
    return x.var(dim, correction=0).add(1e-6).sqrt()


def drifting_features(
    feature_maps: Sequence[torch.Tensor], patch_sizes: Sequence[int] = (2, 4)
) -> List[torch.Tensor]:
    r"""Multi-scale features of Drifting models (Deng et al., 2026, App. A.5).

    For each feature map of shape (batch_size, C, H, W) of a pretrained
    encoder: the \(HW\) vectors of the locations, the mean and standard
    deviation over all locations, and the means and standard deviations over
    the \(k \times k\) patches for each ``k`` in ``patch_sizes`` that divides
    \(H\) and \(W\), as in the reference code (lambertae/drifting,
    ``get_activations``). Each feature is one input of `DriftingLoss`.

    Args:
        feature_maps: Feature maps of shape (batch_size, C, H, W), such as the
            stage outputs of a ResNet.
        patch_sizes: Patch sizes of the local means and standard deviations.

    Returns:
        Features of shape (batch_size, locations, C).
    """
    features = []
    for f in feature_maps:
        h, w = f.shape[-2:]
        patches = [
            f.unflatten(3, (w // k, k)).unflatten(2, (h // k, k))
            for k in patch_sizes
            if h % k == 0 and w % k == 0
        ]
        features += [f.flatten(2).mT, f.mean((2, 3)).unsqueeze(1), _std(f, (2, 3)).unsqueeze(1)]
        features += [p.mean((3, 5)).flatten(2).mT for p in patches]
        features += [_std(p, (3, 5)).flatten(2).mT for p in patches]
    return features


class DriftingLoss(BaseLoss):
    r"""Drifting loss for one-step generators (Deng et al., 2026).

    A generator \(x = f_\theta(\epsilon, c, \alpha)\) is trained so that its
    samples stop drifting: the loss regresses the features of each generated
    sample onto their detached values moved by the drifting field (Eq. 6),
    summed over features (Eq. 14):

    \[
    L = \sum_j \big\|\tilde\phi_j(x) - \mathrm{sg}\big(\tilde\phi_j(x)
        + \tilde V_j(\tilde\phi_j(x))\big)\big\|^2,
    \qquad V = V^+_p - V^-_q,
    \]

    where the positives are data of the sample's class and the negatives
    are the generated samples of the same class and, for classifier-free
    guidance, unconditional data with kernel weight \(w\) (Eq. 15, App. A.7).
    Each feature tensor holds groups of samples, one drifting field per
    group and location; see `drifting_loss`. The features come from a
    pretrained encoder that stays outside the loss; gradients flow through it
    into the generator. Generation is one model call.

    Args:
        temperatures: Temperatures \(\tau\) of the kernel; the paper uses
            0.02, 0.05 and 0.2 for every configuration.
        **kwargs: Keyword arguments of `BaseLoss`: ``reduction`` (``"mean"``,
            ``"sum"``, or ``"none"``), ``dtype`` and ``device``.

    Example:
        One training step of the latent B/2 model (Table 8): 128 class
        labels with 64 generated samples, 128 positives and 32 unconditional
        negatives each, \(\alpha\) on \([1, 4]\) with
        \(p(\alpha) \propto \alpha^{-5}\), AdamW with learning rate 4e-4,
        betas (0.9, 0.95), no weight decay and 10k warmup steps, gradients
        clipped at 2.0, 200k steps and an EMA of the weights; the sizes are
        those of the global batch, split over the data-parallel ranks by
        label. ``x_pos`` holds the latents of 128 samples of each label's
        class and ``x_neg`` 32 samples of random classes per label, from
        per-class queues of 128 and a global queue of 1000 latents
        (App. A.8). The features are the generated latents and the
        multi-scale features of the four stages of a ResNet-50 from the
        Hugging Face Hub on the decoded images; the paper's best encoder is a
        ResNet MAE trained on the latents (App. A.3).

        ```python
        import timm
        import torch
        from diffusers.models import AutoencoderKL

        from torchebm.losses import DriftingLoss, drifting_features
        from torchebm.models import DiT
        from torchebm.utils import sample_guidance

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").requires_grad_(False)
        encoder = timm.create_model(
            "resnet50", pretrained=True, features_only=True, out_indices=(1, 2, 3, 4)
        ).requires_grad_(False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = DiT(
            input_size=32,
            in_channels=4,
            patch_size=2,
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_classes=1000,
            guidance_conditioning="scale",
            qk_norm=True,
            swiglu=True,
            rope="1d",
            rms_norm=True,
            time_invariant=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, betas=(0.9, 0.95), weight_decay=0.0)
        loss_fn = DriftingLoss()


        def features(latents, groups):
            images = (vae.decode(latents / 0.18215).sample + 1) / 2
            maps = encoder((images - mean) / std)
            return [f.unflatten(0, (groups, -1)) for f in [latents.flatten(1)[:, None], *drifting_features(maps)]]


        n_c, n_gen, n_pos, n_neg = 128, 64, 128, 32
        y = torch.randint(0, 1000, (n_c,))
        alpha, _ = sample_guidance(n_c, 1.0, 4.0, power=5.0)
        x = model(
            torch.randn(n_c * n_gen, 4, 32, 32),
            y=y.repeat_interleave(n_gen),
            guidance_scale=alpha.repeat_interleave(n_gen),
        )
        with torch.no_grad():
            positive, negative = features(x_pos, n_c), features(x_neg, n_c)
        weight = (alpha - 1) * (n_gen - 1) / n_neg
        loss = loss_fn(features(x, n_c), positive, negative, negative_weight=weight)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        ```
    """

    def __init__(self, temperatures: Sequence[float] = (0.02, 0.05, 0.2), **kwargs):
        super().__init__(**kwargs)
        self.temperatures = tuple(float(tau) for tau in temperatures)

    def forward(
        self,
        input: Features,
        positive: Features,
        negative: Optional[Features] = None,
        negative_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Sum the drifting losses of one feature or a list of features.

        Args:
            input: Features of the generated samples, shape
                (groups, n, ..., D), or a list of such features.
            positive: Features of the positive samples, shape
                (groups, n_pos, ..., D), or a list matching ``input``.
            negative: Features of the unconditional negatives, shape
                (groups, n_neg, ..., D), or a list matching ``input``.
            negative_weight: Kernel weight of the unconditional negatives of
                each group, shape (groups,).

        Returns:
            Scalar loss, or shape (groups, n) for ``reduction="none"``.
        """
        if isinstance(input, torch.Tensor):
            return drifting_loss(
                input, positive, negative, negative_weight, self.temperatures, self.reduction
            )
        negative = [None] * len(input) if negative is None else negative
        losses = [
            drifting_loss(i, p, n, negative_weight, self.temperatures, self.reduction)
            for i, p, n in zip(input, positive, negative)
        ]
        return torch.stack(losses).sum(0)
