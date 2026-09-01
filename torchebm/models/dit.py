from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from torchebm.models.backbone import _ConditionalBackbone, _xavier_linear_init
from torchebm.models.components import (
    AdaLNZeroBlock,
    AdaLNZeroPatchHead,
    ConvPatchEmbed2d,
    build_2d_sincos_pos_embed,
)

_DIT_CONFIGS = {
    "S": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "B": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "L": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    "XL": {"embed_dim": 1152, "depth": 28, "num_heads": 16},
}

_POS_EMBED_KINDS = ("sincos", "learnable", None)


class DiT(_ConditionalBackbone):
    r"""Diffusion Transformer (DiT) backbone with adaLN-Zero conditioning.

    The architecture of Peebles & Xie (2023, arXiv:2212.09748): patchify,
    transformer blocks whose LayerNorms are modulated by a per-sample
    conditioning vector (adaLN-Zero, residual branches gated from zero), and a
    linear patch head, also zero-initialized, so the model is the identity-plus-
    zero map at initialization.

    The module is loss-agnostic: it maps ``(B, C, H, W)`` to
    ``(B, out_channels, H, W)`` and can serve as the velocity field of
    `FlowMatchingLoss`, the gradient field of `EquilibriumMatchingLoss`, a
    noise/score predictor, or any other image-shaped regressor.

    Standard sizes ship as factory functions (`dit_s_2` ... `dit_xl_8`):

    | Size | depth | embed_dim | num_heads |
    |------|-------|-----------|-----------|
    | S    | 12    | 384       | 6         |
    | B    | 12    | 768       | 12        |
    | L    | 24    | 1024      | 16        |
    | XL   | 28    | 1152      | 16        |

    Conditioning is composable. `forward` accepts any combination of a
    timestep ``t``, integer class labels ``y``, and a pre-built vector
    ``cond``; the embeddings of whichever are present are summed into the
    single adaLN vector. Both embedders are pluggable modules.

    When ``num_classes`` is set, the label table always allocates one extra
    null row with id ``num_classes``, so classifier-free guidance works
    regardless of where the label dropout happens: model-side
    (``class_dropout_prob > 0``), loss-side (``cfg_dropout`` with
    ``null_condition=num_classes``), or at sampling time
    (`ClassifierFreeGuidance` with ``null_condition=num_classes``).

    ``head_dim``/``head_depth`` build a wide, shallow head (the DiT-DH variant
    used for high-dimensional latent spaces): tokens are projected to
    ``head_dim``, run through ``head_depth`` additional adaLN-Zero blocks at
    that width, and decoded by the patch head there.

    Parity with the reference implementation
    (github.com/facebookresearch/DiT): block math, modulation order,
    unpatchify layout, sin/cos tables (computed in float64 and cast, so the
    values are bit-identical), and the weight-initialization scheme all
    match. Two constructor defaults differ deliberately:
    ``out_channels=None`` means ``in_channels`` (the reference defaults to
    ``learn_sigma=True``, i.e. ``2 * in_channels``) and
    ``class_dropout_prob`` defaults to 0.0 (the reference trains with 0.1).
    The timestep frequency table is built on the input's device instead of
    on CPU; the math is unchanged.

    Args:
        input_size: Spatial size of the input, an int for square inputs or an
            ``(H, W)`` tuple. Each side must be divisible by ``patch_size``.
        in_channels: Number of input channels.
        patch_size: Side length of square patches.
        out_channels: Output channels. ``None`` (default) matches
            ``in_channels``; pass ``2 * in_channels`` for a learned-sigma
            head.
        embed_dim: Token width of the transformer trunk.
        depth: Number of trunk blocks.
        num_heads: Attention heads in the trunk (must divide ``embed_dim``).
        mlp_ratio: Feed-forward expansion ratio.
        cond_dim: Width of the conditioning vector. ``None`` (default) uses
            ``embed_dim``. Custom embedders and ``cond=`` inputs must produce
            this width.
        num_classes: Enables the built-in label embedder with this many
            classes (plus the null row). ``None`` (default) builds no label
            table; ``y=`` then requires a custom ``y_embedder``.
        class_dropout_prob: Probability of replacing labels with the null
            token during training (model-side classifier-free dropout).
            Requires ``num_classes``.
        t_embedder: Module mapping ``(B,)`` timesteps to ``(B, cond_dim)``.
            ``None`` (default) builds an `MLPTimestepEmbedder`. The default is
            always constructed; a model driven purely through ``cond=``
            carries it unused.
        y_embedder: Module mapping labels to ``(B, cond_dim)``. Mutually
            exclusive with ``num_classes``/``class_dropout_prob``.
        head_dim: Token width of the head. ``None`` (default) keeps
            ``embed_dim`` and adds no projection.
        head_depth: Number of extra adaLN-Zero blocks in the head.
        head_num_heads: Attention heads of the head blocks. ``None`` (default)
            reuses ``num_heads``.
        pos_embed: Positional embedding kind: ``"sincos"`` (fixed 2D
            sin/cos, default), ``"learnable"``, or ``None`` for none.

    Example:
        ```python
        from torchebm.models import DiT, dit_b_2

        model = DiT(
            input_size=32, in_channels=4, patch_size=2,
            embed_dim=384, depth=12, num_heads=6, num_classes=1000,
        )
        preset = dit_b_2(input_size=32, in_channels=4, num_classes=1000)
        v = preset(x, t, y=labels)  # (B, 4, 32, 32)
        ```
    """

    def __init__(
        self,
        *,
        input_size: Union[int, Tuple[int, int]],
        in_channels: int,
        patch_size: int = 2,
        out_channels: Optional[int] = None,
        embed_dim: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        cond_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        class_dropout_prob: float = 0.0,
        t_embedder: Optional[nn.Module] = None,
        y_embedder: Optional[nn.Module] = None,
        head_dim: Optional[int] = None,
        head_depth: int = 0,
        head_num_heads: Optional[int] = None,
        pos_embed: Optional[str] = "sincos",
    ):
        if isinstance(input_size, int):
            size = (int(input_size), int(input_size))
        else:
            size = tuple(int(s) for s in input_size)
            if len(size) != 2:
                raise ValueError(f"input_size must be an int or (H, W), got {input_size!r}")
        p = int(patch_size)
        if p <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if size[0] % p or size[1] % p:
            raise ValueError(f"input_size {size} must be divisible by patch_size {p}")
        if pos_embed not in _POS_EMBED_KINDS:
            raise ValueError(f"pos_embed must be one of {_POS_EMBED_KINDS}, got {pos_embed!r}")
        if head_depth < 0:
            raise ValueError(f"head_depth must be non-negative, got {head_depth}")
        resolved_cond = int(cond_dim) if cond_dim is not None else int(embed_dim)
        if resolved_cond <= 0:
            raise ValueError(f"cond_dim must be positive, got {resolved_cond}")

        super().__init__(
            cond_dim=resolved_cond,
            num_classes=num_classes,
            class_dropout_prob=class_dropout_prob,
            t_embedder=t_embedder,
            y_embedder=y_embedder,
        )

        self.input_size = size
        self.patch_size = p
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels) if out_channels is not None else self.in_channels
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.grid_size = (size[0] // p, size[1] // p)
        self.head_dim = int(head_dim) if head_dim is not None else self.embed_dim
        self.head_depth = int(head_depth)
        self.pos_embed_type = pos_embed

        self.patch_embed = ConvPatchEmbed2d(
            in_channels=self.in_channels, embed_dim=self.embed_dim, patch_size=p
        )

        num_patches = self.grid_size[0] * self.grid_size[1]
        if pos_embed == "sincos":
            pe = build_2d_sincos_pos_embed(self.embed_dim, self.grid_size)
            self.register_buffer("pos_embed", pe.unsqueeze(0), persistent=False)
        elif pos_embed == "learnable":
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        self.blocks = nn.ModuleList(
            AdaLNZeroBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                cond_dim=self.cond_dim,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(self.depth)
        )

        self.head_proj = (
            nn.Linear(self.embed_dim, self.head_dim)
            if self.head_dim != self.embed_dim
            else None
        )
        self.head_blocks = nn.ModuleList(
            AdaLNZeroBlock(
                embed_dim=self.head_dim,
                num_heads=int(head_num_heads) if head_num_heads is not None else self.num_heads,
                cond_dim=self.cond_dim,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(self.head_depth)
        )
        self.head = AdaLNZeroPatchHead(
            embed_dim=self.head_dim,
            cond_dim=self.cond_dim,
            patch_size=p,
            out_channels=self.out_channels,
            grid_size=self.grid_size,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        r"""Apply the reference DiT initialization scheme.

        Xavier-uniform on every trunk Linear with zeroed biases, the patch
        conv initialized as a flattened Linear, std-0.02 normal weights for
        the default timestep and label embedders, and re-zeroed adaLN
        modulations and final head. User-supplied embedder modules are left
        untouched; they own their initialization.
        """
        for module in (self.blocks, self.head_blocks, self.head):
            module.apply(_xavier_linear_init)
        if self.head_proj is not None:
            _xavier_linear_init(self.head_proj)

        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.proj.bias)

        self._init_default_embedder_weights()

        for block in (*self.blocks, *self.head_blocks):
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.head.modulation[-1].weight)
        nn.init.zeros_(self.head.modulation[-1].bias)
        nn.init.zeros_(self.head.proj.weight)
        nn.init.zeros_(self.head.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        *,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Map ``(B, C, H, W)`` to ``(B, out_channels, H, W)``.

        Follows the library's ``model(x, t, y=)`` convention. The adaLN
        conditioning vector is the sum of the embeddings of whichever of
        ``t``, ``y``, and ``cond`` are given; passing none raises.

        Args:
            x: Input of shape ``(B, in_channels, H, W)`` matching
                ``input_size``.
            t: Timesteps of shape ``(B,)``, embedded by ``t_embedder``.
            y: Integer class labels of shape ``(B,)``, embedded by
                ``y_embedder`` (id ``num_classes`` is the null token).
            cond: Pre-built conditioning vector of shape ``(B, cond_dim)``,
                added as-is.

        Returns:
            torch.Tensor: Output of shape ``(B, out_channels, H, W)``.
        """
        c = self._condition(t, y, cond)
        if tuple(x.shape[-2:]) != self.input_size:
            raise ValueError(
                f"input has spatial size {tuple(x.shape[-2:])}, "
                f"expected input_size={self.input_size}"
            )
        tokens = self.patch_embed(x)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens, c)
        if self.head_proj is not None:
            tokens = self.head_proj(tokens)
        for block in self.head_blocks:
            tokens = block(tokens, c)
        return self.head(tokens, c)


def _preset(size: str, patch_size: int, kwargs: dict) -> DiT:
    merged = {**_DIT_CONFIGS[size], "patch_size": patch_size, **kwargs}
    return DiT(**merged)


def dit_s_2(**kwargs) -> DiT:
    r"""DiT-S/2: depth 12, width 384, 6 heads, patch 2. Overrides pass through."""
    return _preset("S", 2, kwargs)


def dit_s_4(**kwargs) -> DiT:
    r"""DiT-S/4: depth 12, width 384, 6 heads, patch 4. Overrides pass through."""
    return _preset("S", 4, kwargs)


def dit_s_8(**kwargs) -> DiT:
    r"""DiT-S/8: depth 12, width 384, 6 heads, patch 8. Overrides pass through."""
    return _preset("S", 8, kwargs)


def dit_b_2(**kwargs) -> DiT:
    r"""DiT-B/2: depth 12, width 768, 12 heads, patch 2. Overrides pass through."""
    return _preset("B", 2, kwargs)


def dit_b_4(**kwargs) -> DiT:
    r"""DiT-B/4: depth 12, width 768, 12 heads, patch 4. Overrides pass through."""
    return _preset("B", 4, kwargs)


def dit_b_8(**kwargs) -> DiT:
    r"""DiT-B/8: depth 12, width 768, 12 heads, patch 8. Overrides pass through."""
    return _preset("B", 8, kwargs)


def dit_l_2(**kwargs) -> DiT:
    r"""DiT-L/2: depth 24, width 1024, 16 heads, patch 2. Overrides pass through."""
    return _preset("L", 2, kwargs)


def dit_l_4(**kwargs) -> DiT:
    r"""DiT-L/4: depth 24, width 1024, 16 heads, patch 4. Overrides pass through."""
    return _preset("L", 4, kwargs)


def dit_l_8(**kwargs) -> DiT:
    r"""DiT-L/8: depth 24, width 1024, 16 heads, patch 8. Overrides pass through."""
    return _preset("L", 8, kwargs)


def dit_xl_2(**kwargs) -> DiT:
    r"""DiT-XL/2: depth 28, width 1152, 16 heads, patch 2. Overrides pass through."""
    return _preset("XL", 2, kwargs)


def dit_xl_4(**kwargs) -> DiT:
    r"""DiT-XL/4: depth 28, width 1152, 16 heads, patch 4. Overrides pass through."""
    return _preset("XL", 4, kwargs)


def dit_xl_8(**kwargs) -> DiT:
    r"""DiT-XL/8: depth 28, width 1152, 16 heads, patch 8. Overrides pass through."""
    return _preset("XL", 8, kwargs)
