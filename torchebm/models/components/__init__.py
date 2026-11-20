"""Reusable neural network building blocks for TorchEBM models.

These are intentionally *model-agnostic* components that can be composed into
backbones compatible with different losses and samplers.

Paper-preset factories (e.g. the DiT sizes) live next to their backbone in
`torchebm.models`, not here.
"""

from .embeddings import MLPTimestepEmbedder, LabelEmbedder
from .positional import RotaryEmbedding1d, RotaryEmbedding2d, build_2d_sincos_pos_embed
from .patch import ConvPatchEmbed2d, patchify2d, unpatchify2d
from .attention import scaled_dot_product_attention
from .transformer import (
    FeedForward,
    SwiGLUFeedForward,
    MultiheadSelfAttention,
    AdaLNZeroBlock,
    AdaLNZeroMLPBlock,
    InContextBlock,
)
from .heads import AdaLNZeroLinearHead, AdaLNZeroPatchHead, PatchHead

__all__ = [
    "MLPTimestepEmbedder",
    "LabelEmbedder",
    "build_2d_sincos_pos_embed",
    "RotaryEmbedding1d",
    "RotaryEmbedding2d",
    "ConvPatchEmbed2d",
    "patchify2d",
    "unpatchify2d",
    "scaled_dot_product_attention",
    "FeedForward",
    "SwiGLUFeedForward",
    "MultiheadSelfAttention",
    "AdaLNZeroBlock",
    "AdaLNZeroMLPBlock",
    "InContextBlock",
    "AdaLNZeroLinearHead",
    "AdaLNZeroPatchHead",
    "PatchHead",
]
