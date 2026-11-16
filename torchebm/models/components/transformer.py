from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import scaled_dot_product_attention


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: (B,N,D), shift/scale: (B,D)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _RMSNorm(nn.RMSNorm):
    r"""`nn.RMSNorm`, written out under `torch.compile`.

    PyTorch 2.14 fails to trace the forward-mode derivative of the fused
    CUDA kernel under `torch.compile`, so compiled graphs use the RMSNorm of
    the LightningDiT reference code: statistics in float32, cast back, then
    the learnable scale.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.compiler.is_compiling():
            return super().forward(x)
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


def _norm(dim: int, eps: float, rms_norm: bool) -> nn.Module:
    if rms_norm:
        return _RMSNorm(dim, eps=eps)
    return nn.LayerNorm(dim, elementwise_affine=False, eps=eps)


class MultiheadSelfAttention(nn.Module):
    """Self-attention with fused QKV projection, dispatching to PyTorch SDPA.

    Uses ``F.scaled_dot_product_attention`` so the flash-attention and
    memory-efficient kernels are selected automatically on supported hardware
    (Ampere+ GPUs in fp16/bf16, any CUDA device via mem-efficient fallback).
    Under ``torch.func.jvp`` the output tangent comes from the JVP of
    attention in `scaled_dot_product_attention`.

    ``qk_norm`` normalizes queries and keys over the head dimension before
    attention (QK-norm), with an affine LayerNorm, or RMSNorm when
    ``rms_norm`` is set, as in LightningDiT. ``forward`` takes an optional
    `RotaryEmbedding2d` applied to queries and keys after QK-norm.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qk_norm: bool = False,
        rms_norm: bool = False,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        if qk_norm:
            self.q_norm = _RMSNorm(self.head_dim, eps=1e-6) if rms_norm else nn.LayerNorm(self.head_dim)
            self.k_norm = _RMSNorm(self.head_dim, eps=1e-6) if rms_norm else nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x: torch.Tensor, rope: Optional[nn.Module] = None) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = rope(q), rope(k)
        dropout_p = self.dropout if self.training else 0.0
        y = scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        y = y.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        act_layer: Optional[Callable[[], nn.Module]] = None,
    ):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        act = act_layer() if act_layer is not None else nn.GELU(approximate="tanh")
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden, bias=True),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwiGLUFeedForward(nn.Module):
    r"""SwiGLU feed-forward network (Shazeer, 2020).

    \(\mathrm{fc2}\big(\mathrm{SiLU}(x W_1) \odot x W_2\big)\), with \(W_1\)
    and \(W_2\) fused in ``fc1``. The hidden width is
    \(\lfloor \tfrac{2}{3} \lfloor r D \rfloor \rfloor\) for ratio \(r\) and
    width \(D\), so it has as many weights as `FeedForward` of the same
    ratio, as in LightningDiT (github.com/hustvl/LightningDiT,
    ``SwiGLUFFN``).

    Args:
        embed_dim: Input and output width \(D\).
        mlp_ratio: Expansion ratio \(r\) of the equivalent `FeedForward`.
    """

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(2 / 3 * int(embed_dim * mlp_ratio))
        self.fc1 = nn.Linear(embed_dim, 2 * hidden, bias=True)
        self.fc2 = nn.Linear(hidden, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(x1) * x2)


class AdaLNZeroBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning.

    Takes a per-sample conditioning vector `cond` (B, cond_dim) and applies it
    to modulate norms + gate residuals.

    This is a reusable block; it does not assume anything about what `cond`
    represents (time, labels, text, etc.).

    ``qk_norm``, ``swiglu`` and ``rms_norm`` switch on the LightningDiT block:
    QK-norm in the attention, a `SwiGLUFeedForward`, and RMSNorm with a
    learnable scale in place of the LayerNorms. ``forward`` passes an
    optional `RotaryEmbedding2d` to the attention.
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
        qk_norm: bool = False,
        swiglu: bool = False,
        rms_norm: bool = False,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.cond_dim = int(cond_dim) if cond_dim is not None else int(embed_dim)

        self.norm1 = _norm(self.embed_dim, eps, rms_norm)
        if attn is None:
            attn = MultiheadSelfAttention(
                self.embed_dim, num_heads=num_heads, qk_norm=qk_norm, rms_norm=rms_norm
            )
        self.attn = attn
        self.norm2 = _norm(self.embed_dim, eps, rms_norm)
        if mlp is None:
            ffn = SwiGLUFeedForward if swiglu else FeedForward
            mlp = ffn(self.embed_dim, mlp_ratio=mlp_ratio)
        self.mlp = mlp

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.cond_dim, 6 * self.embed_dim, bias=True),
        )

        # Zero-init to start near identity.
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, rope: Optional[nn.Module] = None
    ) -> torch.Tensor:
        # x: (B,N,D), cond: (B,cond_dim)
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(cond).chunk(6, dim=1)

        h = modulate(self.norm1(x), shift1, scale1)
        x = x + gate1.unsqueeze(1) * (self.attn(h) if rope is None else self.attn(h, rope))
        x = x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x


class AdaLNZeroMLPBlock(nn.Module):
    """Residual MLP block with adaLN-Zero conditioning on a vector stream.

    The MLP branch of `AdaLNZeroBlock` for (B, D) inputs: LayerNorm without
    affine parameters, conditioning-derived shift/scale/gate, and a
    zero-initialized modulation so the block starts as the identity.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        cond_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        act_layer: Optional[Callable[[], nn.Module]] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.cond_dim = int(cond_dim) if cond_dim is not None else int(embed_dim)

        self.norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=eps)
        self.mlp = FeedForward(self.embed_dim, mlp_ratio=mlp_ratio, act_layer=act_layer)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.cond_dim, 3 * self.embed_dim, bias=True),
        )

        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B,D), cond: (B,cond_dim)
        shift, scale, gate = self.modulation(cond).chunk(3, dim=1)
        return x + gate * self.mlp(self.norm(x) * (1 + scale) + shift)
