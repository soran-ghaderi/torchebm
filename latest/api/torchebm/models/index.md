# `torchebm.models`

Model namespace.

TorchEBM is designed for plug-and-play experimentation:

- try different losses with the same backbone
- try different backbones with the same loss
- use samplers as long as the model signature matches

This package therefore exposes *reusable building blocks* under `torchebm.models.components` and a small set of generic backbones/wrappers.

## `AdaLNZeroBlock`

Bases: `Module`

Transformer block with adaLN-Zero conditioning.

Takes a per-sample conditioning vector `cond` (B, cond_dim) and applies it to modulate norms + gate residuals.

This is a reusable block; it does not assume anything about what `cond` represents (time, labels, text, etc.).

Source code in `torchebm/models/components/transformer.py`

```python
class AdaLNZeroBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning.

    Takes a per-sample conditioning vector `cond` (B, cond_dim) and applies it
    to modulate norms + gate residuals.

    This is a reusable block; it does not assume anything about what `cond`
    represents (time, labels, text, etc.).
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        cond_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        attn: Optional[nn.Module] = None,
        mlp: Optional[nn.Module] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.cond_dim = int(cond_dim) if cond_dim is not None else int(embed_dim)

        self.norm1 = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=eps)
        self.attn = attn if attn is not None else MultiheadSelfAttention(self.embed_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=eps)
        self.mlp = mlp if mlp is not None else FeedForward(self.embed_dim, mlp_ratio=mlp_ratio)

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.cond_dim, 6 * self.embed_dim, bias=True),
        )

        # Zero-init to start near identity.
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B,N,D), cond: (B,cond_dim)
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(cond).chunk(6, dim=1)

        x = x + gate1.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift1, scale1))
        x = x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x
```

## `AdaLNZeroPatchHead`

Bases: `Module`

Final layer that maps token features to patch pixels with adaLN-Zero.

Source code in `torchebm/models/components/heads.py`

```python
class AdaLNZeroPatchHead(nn.Module):
    """Final layer that maps token features to patch pixels with adaLN-Zero."""

    def __init__(
        self,
        *,
        embed_dim: int,
        cond_dim: Optional[int] = None,
        patch_size: int,
        out_channels: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.cond_dim = int(cond_dim) if cond_dim is not None else int(embed_dim)
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)

        self.norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=eps)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.cond_dim, 2 * self.embed_dim, bias=True),
        )
        self.proj = nn.Linear(self.embed_dim, self.patch_size * self.patch_size * self.out_channels, bias=True)

        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(cond).chunk(2, dim=1)
        tokens = modulate(self.norm(tokens), shift, scale)
        patches = self.proj(tokens)
        return unpatchify2d(patches, self.patch_size, out_channels=self.out_channels)
```

## `ConditionalTransformer2D`

Bases: `Module`

Generic conditional 2D Transformer backbone.

This module is intentionally *loss-agnostic*.

Inputs

- `x`: image-like tensor (B,C,H,W)
- `cond`: conditioning vector (B, cond_dim)

Output

- image-like tensor (B, out_channels, H, W)

You can plug this into EqM, diffusion, score matching, etc. by choosing:

- how `cond` is produced (time, labels, text, ...)
- `out_channels` and head behavior

Source code in `torchebm/models/conditional_transformer_2d.py`

```python
class ConditionalTransformer2D(nn.Module):
    """Generic conditional 2D Transformer backbone.

    This module is intentionally *loss-agnostic*.

    Inputs:
      - `x`: image-like tensor (B,C,H,W)
      - `cond`: conditioning vector (B, cond_dim)

    Output:
      - image-like tensor (B, out_channels, H, W)

    You can plug this into EqM, diffusion, score matching, etc. by choosing:
      - how `cond` is produced (time, labels, text, ...)
      - `out_channels` and head behavior
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        input_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        cond_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        use_sincos_pos_embed: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.cond_dim = int(cond_dim) if cond_dim is not None else int(embed_dim)

        self.patch_embed = ConvPatchEmbed2d(
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
        )

        num_patches_per_side = self.input_size // self.patch_size
        if num_patches_per_side * self.patch_size != self.input_size:
            raise ValueError("input_size must be divisible by patch_size")

        self.use_sincos_pos_embed = bool(use_sincos_pos_embed)
        if self.use_sincos_pos_embed:
            pe = build_2d_sincos_pos_embed(self.embed_dim, num_patches_per_side)
            self.register_buffer("pos_embed", pe.unsqueeze(0), persistent=False)  # (1,N,D)
        else:
            self.pos_embed = None

        self.blocks = nn.ModuleList(
            [
                AdaLNZeroBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    cond_dim=self.cond_dim,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(self.depth)
            ]
        )

        self.head = AdaLNZeroPatchHead(
            embed_dim=self.embed_dim,
            cond_dim=self.cond_dim,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)  # (B,N,D)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)

        for block in self.blocks:
            tokens = block(tokens, cond)

        return self.head(tokens, cond)
```

## `ConvPatchEmbed2d`

Bases: `Module`

Patch embedding via strided conv.

This is a lightweight replacement for timm's PatchEmbed.

Source code in `torchebm/models/components/patch.py`

```python
class ConvPatchEmbed2d(nn.Module):
    """Patch embedding via strided conv.

    This is a lightweight replacement for timm's PatchEmbed.
    """

    def __init__(self, *, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        p = int(patch_size)
        self.patch_size = p
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=p, stride=p, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W) -> (B, N, D)
        y = self.proj(x)
        b, d, gh, gw = y.shape
        return y.flatten(2).transpose(1, 2).contiguous()
```

## `LabelClassifierFreeGuidance`

Bases: `Module`

Classifier-free guidance wrapper for label-conditioned models.

This wrapper is intentionally small and generic:

- assumes the base model accepts `y` (labels) and supports a *null label id*
- performs two forward passes (cond and uncond)
- applies guidance to the first `guide_channels` channels by default

It does **not** assume a specific loss (EqM/diffusion/etc).

Expected base signature

`base(x, t, y=..., **kwargs) -> Tensor[B,C,H,W]`

You can use it with `FlowSampler` by wrapping your model instance.

Source code in `torchebm/models/wrappers.py`

```python
class LabelClassifierFreeGuidance(nn.Module):
    """Classifier-free guidance wrapper for label-conditioned models.

    This wrapper is intentionally small and generic:
    - assumes the base model accepts `y` (labels) and supports a *null label id*
    - performs two forward passes (cond and uncond)
    - applies guidance to the first `guide_channels` channels by default

    It does **not** assume a specific loss (EqM/diffusion/etc).

    Expected base signature:
      `base(x, t, y=..., **kwargs) -> Tensor[B,C,H,W]`

    You can use it with `FlowSampler` by wrapping your model instance.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        null_label_id: int,
        cfg_scale: float = 1.0,
        guide_channels: int = 3,
    ):
        super().__init__()
        self.base = base
        self.null_label_id = int(null_label_id)
        self.cfg_scale = float(cfg_scale)
        self.guide_channels = int(guide_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor, *, y: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.cfg_scale <= 1.0:
            return self.base(x, t, y=y, **kwargs)

        y_null = torch.full_like(y, fill_value=self.null_label_id)

        cond = self.base(x, t, y=y, **kwargs)
        uncond = self.base(x, t, y=y_null, **kwargs)

        c = min(self.guide_channels, cond.shape[1])
        guided = uncond[:, :c] + self.cfg_scale * (cond[:, :c] - uncond[:, :c])

        if c == cond.shape[1]:
            return guided
        return torch.cat([guided, uncond[:, c:]], dim=1)
```

## `LabelEmbedder`

Bases: `Module`

Label embedding with optional classifier-free guidance token dropping.

If `dropout_prob>0`, one extra embedding row is allocated to represent the *null/unconditional* label.

Note: this module does *not* assume any specific loss/sampler; it only produces vectors.

Source code in `torchebm/models/components/embeddings.py`

```python
class LabelEmbedder(nn.Module):
    """Label embedding with optional classifier-free guidance token dropping.

    If `dropout_prob>0`, one extra embedding row is allocated to represent the
    *null/unconditional* label.

    Note: this module does *not* assume any specific loss/sampler; it only
    produces vectors.
    """

    def __init__(self, num_classes: int, out_dim: int, dropout_prob: float = 0.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.dropout_prob = float(dropout_prob)
        use_null = self.dropout_prob > 0
        self.null_label_id = self.num_classes if use_null else None
        self.embedding = nn.Embedding(self.num_classes + (1 if use_null else 0), out_dim)

    def maybe_drop_labels(
        self,
        labels: torch.Tensor,
        *,
        force_drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.dropout_prob <= 0:
            return labels
        if self.null_label_id is None:
            raise RuntimeError("LabelEmbedder configured without null label.")

        if force_drop_mask is None:
            drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_mask = force_drop_mask.to(device=labels.device, dtype=torch.bool)

        return torch.where(drop_mask, torch.full_like(labels, self.null_label_id), labels)

    def forward(
        self,
        labels: torch.Tensor,
        *,
        training: bool,
        force_drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if training or (force_drop_mask is not None):
            labels = self.maybe_drop_labels(labels, force_drop_mask=force_drop_mask)
        return self.embedding(labels)
```

## `MLPTimestepEmbedder`

Bases: `Module`

Embed a scalar timestep into a vector via sinusoid + MLP.

This is a generic block (useful for EqM, diffusion, flows, etc.).

Source code in `torchebm/models/components/embeddings.py`

```python
class MLPTimestepEmbedder(nn.Module):
    """Embed a scalar timestep into a vector via sinusoid + MLP.

    This is a generic block (useful for EqM, diffusion, flows, etc.).
    """

    def __init__(self, out_dim: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, out_dim, bias=True),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim, bias=True),
        )

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        # t: (B,)
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            t = t.reshape(t.shape[0])
        freq = self.sinusoidal_embedding(t, self.frequency_embedding_size)
        return self.mlp(freq)
```

## `MultiheadSelfAttention`

Bases: `Module`

Self-attention wrapper with batch-first API.

Source code in `torchebm/models/components/transformer.py`

```python
class MultiheadSelfAttention(nn.Module):
    """Self-attention wrapper with batch-first API."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.mha(x, x, x, need_weights=False)
        return y
```

## `build_2d_sincos_pos_embed(embed_dim, grid_size, *, device=None, dtype=torch.float32)`

Create 2D sin/cos positional embeddings.

Returns tensor with shape (grid_size\*grid_size, embed_dim).

Source code in `torchebm/models/components/positional.py`

```python
def build_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create 2D sin/cos positional embeddings.

    Returns tensor with shape (grid_size*grid_size, embed_dim).
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    dev = device if device is not None else torch.device("cpu")
    grid_h = torch.arange(grid_size, device=dev, dtype=torch.float32)
    grid_w = torch.arange(grid_size, device=dev, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0).reshape(2, -1)  # (2, M)

    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = torch.cat([emb_h, emb_w], dim=1).to(dtype=dtype)
    return emb
```

## `patchify2d(x, patch_size)`

Convert (B,C,H,W) into patch tokens (B, N, C\*P\*P).

Source code in `torchebm/models/components/patch.py`

```python
def patchify2d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert (B,C,H,W) into patch tokens (B, N, C*P*P)."""
    b, c, h, w = x.shape
    p = int(patch_size)
    if h % p != 0 or w % p != 0:
        raise ValueError(f"H,W must be divisible by patch_size={p}, got {(h, w)}")

    gh, gw = h // p, w // p
    x = x.reshape(b, c, gh, p, gw, p)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, gh, gw, p, p, C)
    return x.view(b, gh * gw, p * p * c)
```

## `unpatchify2d(tokens, patch_size, *, out_channels)`

Convert patch tokens (B,N,P\*P\*C) back to (B,C,H,W).

Source code in `torchebm/models/components/patch.py`

```python
def unpatchify2d(tokens: torch.Tensor, patch_size: int, *, out_channels: int) -> torch.Tensor:
    """Convert patch tokens (B,N,P*P*C) back to (B,C,H,W)."""
    b, n, d = tokens.shape
    p = int(patch_size)

    c = int(out_channels)
    if d != p * p * c:
        raise ValueError(f"Token dim {d} != patch_size^2*out_channels ({p*p*c})")

    grid = int(n ** 0.5)
    if grid * grid != n:
        raise ValueError("Number of tokens must be a perfect square for 2D unpatchify.")

    x = tokens.view(b, grid, grid, p, p, c)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B,C,gh,p,gw,p)
    return x.view(b, c, grid * p, grid * p)
```
