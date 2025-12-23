"""Reusable neural network building blocks for TorchEBM models.

These are intentionally *model-agnostic* components that can be composed into
backbones compatible with different losses and samplers.

The library avoids exposing paper-specific preset/config objects here; keep
those at the example/use-case layer.
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
