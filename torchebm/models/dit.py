from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from torchebm.models.backbone import _ConditionalBackbone, _xavier_linear_init
from torchebm.models.components import (
    AdaLNZeroBlock,
    AdaLNZeroPatchHead,
    ConvPatchEmbed2d,
    InContextBlock,
    PatchHead,
    RotaryEmbedding1d,
    RotaryEmbedding2d,
    build_2d_sincos_pos_embed,
)

_DIT_CONFIGS = {
    "S": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "B": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "L": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    "XL": {"embed_dim": 1152, "depth": 28, "num_heads": 16},
}

_POS_EMBED_KINDS = ("sincos", "learnable", None)
_CONDITIONING_KINDS = ("adaln_zero", "in_context")
_ROPE_KINDS = (None, "1d", "2d")
_IN_CONTEXT_TOKENS = (
    ("y", 8),
    ("guidance", 4),
    ("interval_start", 2),
    ("interval_end", 2),
    ("t", 4),
    ("dt", 4),
)
_IMF_BLOCK_INIT_GAIN = 0.32


def _scaled_normal_init(module: nn.Module, gain: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=gain / module.in_features**0.5)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _rotary_embedding(kind, head_dim, grid_size, seq_len) -> Optional[nn.Module]:
    if kind == "2d":
        return RotaryEmbedding2d(head_dim, grid_size)
    if kind == "1d":
        return RotaryEmbedding1d(head_dim, seq_len)
    return None


class DiT(_ConditionalBackbone):
    r"""Diffusion Transformer (DiT) backbone with adaLN-Zero conditioning.

    The architecture of Peebles & Xie (2023, arXiv:2212.09748): patchify,
    transformer blocks whose LayerNorms are modulated by a per-sample
    conditioning vector (adaLN-Zero, residual branches gated from zero), and a
    linear patch head, also zero-initialized, so the model is the identity-plus-
    zero map at initialization.

    The module is loss-agnostic: it maps ``(B, C, H, W)`` to
    ``(B, out_channels, H, W)`` and can serve as the velocity field of
    `FlowMatchingLoss`, the time-invariant field of `EquilibriumMatchingLoss`, a
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

    ``guidance_conditioning`` makes the classifier-free guidance scale, and
    with ``"scale_and_interval"`` the guidance interval, inputs of
    ``forward``, as in Improved MeanFlow (Geng et al., 2025b, Sec. 4.2).
    Training draws them with `sample_guidance` and regresses on the guided
    velocity, so sampling at a fixed scale takes one forward pass per step
    in place of a conditional and an unconditional pass.

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

    ``qk_norm``, ``swiglu``, ``rope="2d"`` and ``rms_norm`` switch on the
    blocks of LightningDiT (Yao et al., 2025, arXiv:2501.01423): QK-norm in
    the attention, a `SwiGLUFeedForward`, a `RotaryEmbedding2d` on queries
    and keys on top of ``pos_embed``, and RMSNorm with a learnable scale in
    place of the LayerNorms of the blocks and the patch head. They match the
    reference code (github.com/hustvl/LightningDiT) and are off by default,
    which leaves the model above unchanged.

    ``conditioning="in_context"`` is the in-context conditioning of Improved
    MeanFlow (Geng et al., 2025b, Sec. 4.3): each conditioning embedding is
    added to its own learnable tokens, 8 for the class, 4 for the guidance
    scale, 2 for each end of the guidance interval and 4 for each time, and
    these tokens are prepended to the patch tokens. The blocks are
    `InContextBlock`, without modulation and with residual gates initialized
    at zero, the head is a `PatchHead` on the patch tokens, and ``rope="1d"``
    numbers the whole sequence. The model then matches ``imfDiT`` of the
    reference code (Lyy-iiis/imeanflow) and its initialization.

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
        conditioning: ``"adaln_zero"`` (default) sums the conditioning
            embeddings into one adaLN vector; ``"in_context"`` turns them
            into tokens of the sequence and takes no ``cond``.
        cond_dim: Width of the conditioning vector. ``None`` (default) uses
            ``embed_dim``, which ``"in_context"`` requires. Custom embedders
            and ``cond=`` inputs must produce this width.
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
        dt_embedder: Module mapping ``(B,)`` intervals \(t - s\) to
            ``(B, cond_dim)`` for the two-time prediction types. ``None``
            (default) builds an `MLPTimestepEmbedder` when ``prediction_type``
            is ``"average_velocity"`` or ``"flow_map"``, and nothing otherwise.
        guidance_conditioning: Guidance inputs of ``forward``: ``None``
            (default) for none, ``"scale"`` for ``guidance_scale``, or
            ``"scale_and_interval"`` for ``guidance_scale`` and
            ``guidance_interval``. Each end of the interval is embedded by
            its own `MLPTimestepEmbedder`.
        guidance_embedder: Module mapping ``(B,)`` guidance scales to
            ``(B, cond_dim)``. ``None`` (default) embeds \(1 - 1/\omega\)
            with an `MLPTimestepEmbedder`, as Improved MeanFlow does.
            Requires ``guidance_conditioning``.
        head_dim: Token width of the head. ``None`` (default) keeps
            ``embed_dim`` and adds no projection.
        head_depth: Number of extra adaLN-Zero blocks in the head.
        head_num_heads: Attention heads of the head blocks. ``None`` (default)
            reuses ``num_heads``.
        pos_embed: Positional embedding kind: ``"sincos"`` (fixed 2D
            sin/cos, default), ``"learnable"``, or ``None`` for none.
        qk_norm: Normalize queries and keys over the head dimension in every
            attention, with an affine LayerNorm, or RMSNorm with
            ``rms_norm``.
        swiglu: Use `SwiGLUFeedForward` in place of the GELU feed-forward
            network.
        rope: Rotary embedding of queries and keys in every attention, in
            addition to ``pos_embed``: ``None`` (default) for none, ``"2d"``
            for `RotaryEmbedding2d` on the patch grid (LightningDiT; the head
            dimension must be a multiple of 4), or ``"1d"`` for
            `RotaryEmbedding1d` on the sequence, conditioning tokens first
            (Improved MeanFlow). ``"2d"`` requires ``"adaln_zero"``.
        rms_norm: Use RMSNorm with a learnable scale in place of the
            LayerNorms of the blocks and the patch head.
        block_bias: Biases of the attention and feed-forward linear layers of
            the blocks; Improved MeanFlow uses none.
        time_invariant: Pin ``t`` to zero on every call, the Equilibrium
            Matching model convention; labels and ``cond`` still apply. A
            two-time model embeds \(t - s\) only, as Improved MeanFlow does.
        prediction_type: What ``forward`` returns: ``"velocity"`` (default),
            ``"score"``, or ``"epsilon"``, a declaration for consumers that
            leaves the computation unchanged; or a two-time output,
            ``"average_velocity"`` \(u(x_s, s, t)\) or ``"flow_map"``
            \(F(x_s, s, t) = x_s + (t - s)\, u(x_s, s, t)\), which adds
            ``dt_embedder`` and requires ``s=`` and ``t`` on every call.
            ``"flow_map"`` requires ``out_channels == in_channels``.

    Example:
        DiT-XL/2 of the paper on the 32x32x4 SD-VAE latents of ImageNet 256,
        with the reference label dropout; ``out_channels=8`` adds the
        learned-sigma head of the diffusion setup:

        ```python
        from torchebm.models import dit_xl_2

        model = dit_xl_2(
            input_size=32, in_channels=4, num_classes=1000, class_dropout_prob=0.1
        )
        v = model(x_t, t, y=y)
        ```

        LightningDiT-XL/1 on the 16x16x32 VA-VAE latents of ImageNet 256 with
        the blocks of its final configuration
        (``configs/lightningdit_xl_vavae_f16d32.yaml``: SwiGLU, RoPE and
        RMSNorm, no QK-norm):

        ```python
        from torchebm.models import DiT

        model = DiT(
            input_size=16,
            in_channels=32,
            patch_size=1,
            num_classes=1000,
            class_dropout_prob=0.1,
            swiglu=True,
            rope="2d",
            rms_norm=True,
        )
        v = model(x_t, t, y=y)
        ```

        The \(u\) branch of iMF-B/2 (Lyy-iiis/imeanflow, ``imfDiT_B_2``:
        12 blocks of width 768 with 12 heads on the 32x32x4 SD-VAE latents,
        SwiGLU of hidden width 2048, conditioning on \(t - s\), the guidance
        scale and the guidance interval):

        ```python
        from torchebm.models import DiT

        model = DiT(
            input_size=32,
            in_channels=4,
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_classes=1000,
            conditioning="in_context",
            pos_embed=None,
            qk_norm=True,
            swiglu=True,
            rope="1d",
            rms_norm=True,
            block_bias=False,
            guidance_conditioning="scale_and_interval",
            time_invariant=True,
            prediction_type="average_velocity",
        )
        u = model(x_s, t, y, s=s, guidance_scale=w, guidance_interval=interval)
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
        conditioning: str = "adaln_zero",
        cond_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        class_dropout_prob: float = 0.0,
        t_embedder: Optional[nn.Module] = None,
        y_embedder: Optional[nn.Module] = None,
        dt_embedder: Optional[nn.Module] = None,
        guidance_conditioning: Optional[str] = None,
        guidance_embedder: Optional[nn.Module] = None,
        head_dim: Optional[int] = None,
        head_depth: int = 0,
        head_num_heads: Optional[int] = None,
        pos_embed: Optional[str] = "sincos",
        qk_norm: bool = False,
        swiglu: bool = False,
        rope: Optional[str] = None,
        rms_norm: bool = False,
        block_bias: bool = True,
        time_invariant: bool = False,
        prediction_type: str = "velocity",
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
        if conditioning not in _CONDITIONING_KINDS:
            raise ValueError(
                f"conditioning must be one of {_CONDITIONING_KINDS}, got {conditioning!r}"
            )
        if rope not in _ROPE_KINDS:
            raise ValueError(f"rope must be one of {_ROPE_KINDS}, got {rope!r}")
        in_context = conditioning == "in_context"
        if in_context and rope == "2d":
            raise ValueError(
                "rope='2d' positions the patch grid only; in-context conditioning "
                "takes rope='1d' or None"
            )
        resolved_cond = int(cond_dim) if cond_dim is not None else int(embed_dim)
        if resolved_cond <= 0:
            raise ValueError(f"cond_dim must be positive, got {resolved_cond}")
        if in_context and resolved_cond != embed_dim:
            raise ValueError(
                f"in-context conditioning requires cond_dim == embed_dim, got "
                f"{resolved_cond} and {embed_dim}"
            )
        if (
            prediction_type == "flow_map"
            and out_channels is not None
            and out_channels != in_channels
        ):
            raise ValueError(
                "prediction_type='flow_map' requires out_channels == in_channels, "
                f"got {out_channels} and {in_channels}"
            )

        super().__init__(
            cond_dim=resolved_cond,
            num_classes=num_classes,
            class_dropout_prob=class_dropout_prob,
            t_embedder=t_embedder,
            y_embedder=y_embedder,
            dt_embedder=dt_embedder,
            guidance_conditioning=guidance_conditioning,
            guidance_embedder=guidance_embedder,
            time_invariant=time_invariant,
            prediction_type=prediction_type,
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
        self.conditioning = conditioning

        self.patch_embed = ConvPatchEmbed2d(
            in_channels=self.in_channels, embed_dim=self.embed_dim, patch_size=p
        )

        if in_context:
            embedders = {
                "y": self.y_embedder,
                "guidance": self.guidance_embedder,
                "interval_start": self.interval_start_embedder,
                "interval_end": self.interval_end_embedder,
                "t": self.t_embedder,
                "dt": self.dt_embedder,
            }
            groups = [(name, n) for name, n in _IN_CONTEXT_TOKENS if embedders[name] is not None]
            self.token_groups = tuple(name for name, _ in groups)
            counts = torch.tensor([n for _, n in groups])
            self.condition_tokens = nn.Parameter(torch.empty(int(counts.sum()), self.embed_dim))
            self.register_buffer(
                "token_group_index",
                torch.repeat_interleave(torch.arange(len(groups)), counts),
                persistent=False,
            )
        else:
            self.token_groups = ()
            self.condition_tokens = None
        num_prefix = 0 if self.condition_tokens is None else self.condition_tokens.shape[0]

        num_patches = self.grid_size[0] * self.grid_size[1]
        if pos_embed == "sincos":
            pe = build_2d_sincos_pos_embed(self.embed_dim, self.grid_size)
            self.register_buffer("pos_embed", pe.unsqueeze(0), persistent=False)
        elif pos_embed == "learnable":
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        def block(width, heads):
            options = dict(
                embed_dim=width,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qk_norm=qk_norm,
                swiglu=swiglu,
                rms_norm=rms_norm,
                bias=block_bias,
            )
            if in_context:
                return InContextBlock(**options)
            return AdaLNZeroBlock(cond_dim=self.cond_dim, **options)

        self.blocks = nn.ModuleList(block(self.embed_dim, self.num_heads) for _ in range(self.depth))

        self.head_proj = (
            nn.Linear(self.embed_dim, self.head_dim)
            if self.head_dim != self.embed_dim
            else None
        )
        head_heads = int(head_num_heads) if head_num_heads is not None else self.num_heads
        self.head_blocks = nn.ModuleList(block(self.head_dim, head_heads) for _ in range(self.head_depth))
        head_options = dict(
            embed_dim=self.head_dim,
            patch_size=p,
            out_channels=self.out_channels,
            grid_size=self.grid_size,
            rms_norm=rms_norm,
        )
        if in_context:
            self.head = PatchHead(**head_options)
        else:
            self.head = AdaLNZeroPatchHead(cond_dim=self.cond_dim, **head_options)
        seq_len = num_prefix + num_patches
        self.rope = _rotary_embedding(rope, self.embed_dim // self.num_heads, self.grid_size, seq_len)
        self.head_rope = (
            _rotary_embedding(rope, self.head_dim // head_heads, self.grid_size, seq_len)
            if self.head_depth
            else None
        )

        if in_context:
            self._initialize_in_context_weights()
        else:
            self._initialize_weights()

    def _initialize_in_context_weights(self) -> None:
        r"""Apply the initialization of the Improved MeanFlow code (``imfDiT``).

        Normal weights of standard deviation \(0.32/\sqrt{\text{fan-in}}\) in
        the blocks and \(1/\sqrt{\text{fan-in}}\) in the default embedders,
        zero biases, conditioning tokens and default label rows of standard
        deviation \(1/\sqrt{D}\), the patch conv as a flattened Linear, and
        zero residual gates and head projection. User-supplied embedder
        modules are left untouched.
        """
        for module in (self.blocks, self.head_blocks):
            module.apply(lambda m: _scaled_normal_init(m, _IMF_BLOCK_INIT_GAIN))
        if self.head_proj is not None:
            _scaled_normal_init(self.head_proj, _IMF_BLOCK_INIT_GAIN)

        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.proj.bias)

        for embedder in self._default_timestep_embedders():
            embedder.apply(lambda m: _scaled_normal_init(m, 1.0))
        if self._default_y_embedder:
            nn.init.normal_(self.y_embedder.embedding.weight, std=self.embed_dim**-0.5)
        nn.init.normal_(self.condition_tokens, std=self.embed_dim**-0.5)

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
        s: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        guidance_scale: Optional[torch.Tensor] = None,
        guidance_interval: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Map ``(B, C, H, W)`` to ``(B, out_channels, H, W)``.

        Follows the library's ``model(x, t, y=)`` convention. The adaLN
        conditioning vector is the sum of the embeddings of whichever of
        ``t``, ``y``, and ``cond`` are given; passing none raises.
        Two-time models are called as ``model(x_s, t, s=s)`` and embed
        \(s\) and \(t - s\) in place of \(t\). In-context conditioning needs
        every conditioning input of the model on every call, the null label
        for unconditional outputs.

        Args:
            x: Input of shape ``(B, in_channels, H, W)`` matching
                ``input_size``; \(x_s\) for the two-time prediction types.
            t: Timesteps of shape ``(B,)``, embedded by ``t_embedder``; the
                target time for the two-time prediction types.
            y: Integer class labels of shape ``(B,)``, embedded by
                ``y_embedder`` (id ``num_classes`` is the null token).
            s: Times of ``x`` of shape ``(B,)`` for the two-time prediction
                types: ``t_embedder`` embeds \(s\) and ``dt_embedder``
                embeds \(t - s\). Rejected by the one-time types.
            cond: Pre-built conditioning vector of shape ``(B, cond_dim)``,
                added as-is.
            guidance_scale: Guidance scales \(\omega \geq 1\) of shape
                ``(B,)`` for models with ``guidance_conditioning``;
                \(\omega = 1\) is no guidance.
            guidance_interval: Start and end times of the guidance interval,
                in torchebm time, of shape ``(B, 2)``, for
                ``guidance_conditioning="scale_and_interval"``.

        Returns:
            torch.Tensor: Output of shape ``(B, out_channels, H, W)``.
        """
        if tuple(x.shape[-2:]) != self.input_size:
            raise ValueError(
                f"input has spatial size {tuple(x.shape[-2:])}, "
                f"expected input_size={self.input_size}"
            )
        if self.condition_tokens is not None:
            return self._forward_in_context(x, t, y, cond, s, guidance_scale, guidance_interval)
        c = self._condition(x, t, y, cond, s, guidance_scale, guidance_interval)
        tokens = self.patch_embed(x)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens, c, self.rope)
        if self.head_proj is not None:
            tokens = self.head_proj(tokens)
        for block in self.head_blocks:
            tokens = block(tokens, c, self.head_rope)
        return self._output(x, self.head(tokens, c), s, t)

    def _forward_in_context(self, x, t, y, cond, s, guidance_scale, guidance_interval):
        if cond is not None:
            raise ValueError("in-context conditioning takes no cond")
        embeddings = dict(self._embeddings(x, t, y, cond, s, guidance_scale, guidance_interval))
        missing = [name for name in self.token_groups if name not in embeddings]
        if missing:
            raise ValueError(f"in-context conditioning needs {missing} on every call")
        groups = torch.stack([embeddings[name] for name in self.token_groups], dim=1)
        prefix = self.condition_tokens + groups[:, self.token_group_index]
        tokens = self.patch_embed(x)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        tokens = torch.cat([prefix.to(tokens.dtype), tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens, self.rope)
        if self.head_proj is not None:
            tokens = self.head_proj(tokens)
        for block in self.head_blocks:
            tokens = block(tokens, self.head_rope)
        return self._output(x, self.head(tokens[:, prefix.shape[1] :]), s, t)


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
