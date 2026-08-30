"""Reusable neural network building blocks for TorchEBM models.

These are intentionally *model-agnostic* components that can be composed into
backbones compatible with different losses and samplers.

Paper-preset factories (e.g. the DiT sizes) live next to their backbone in
`torchebm.models`, not here.
"""

from .embeddings import MLPTimestepEmbedder, LabelEmbedder
from .positional import build_2d_sincos_pos_embed
from .patch import ConvPatchEmbed2d, patchify2d, unpatchify2d
from .transformer import FeedForward, MultiheadSelfAttention, AdaLNZeroBlock
from .heads import AdaLNZeroPatchHead

__all__ = [
    "MLPTimestepEmbedder",
    "LabelEmbedder",
    "build_2d_sincos_pos_embed",
    "ConvPatchEmbed2d",
    "patchify2d",
    "unpatchify2d",
    "FeedForward",
    "MultiheadSelfAttention",
    "AdaLNZeroBlock",
    "AdaLNZeroPatchHead",
]
