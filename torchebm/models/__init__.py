"""Model namespace.

TorchEBM is designed for plug-and-play experimentation:
- try different losses with the same backbone
- try different backbones with the same loss
- use samplers as long as the model signature matches

This package therefore exposes *reusable building blocks* under
`torchebm.models.components` and a small set of generic backbones/wrappers.
"""

from .conditional_transformer_2d import ConditionalTransformer2D
from .components import (
    AdaLNZeroBlock,
    AdaLNZeroPatchHead,
    ConvPatchEmbed2d,
    FeedForward,
    LabelEmbedder,
    MLPTimestepEmbedder,
    MultiheadSelfAttention,
    build_2d_sincos_pos_embed,
    patchify2d,
    unpatchify2d,
)
from .wrappers import LabelClassifierFreeGuidance

__all__ = [
    "ConditionalTransformer2D",
    "LabelClassifierFreeGuidance",
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
