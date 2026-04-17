"""Model namespace.

TorchEBM is designed for plug-and-play experimentation:
- try different losses with the same backbone
- try different backbones with the same loss
- use samplers as long as the model signature matches

This package therefore exposes *reusable building blocks* under
`torchebm.models.components` and a small set of generic backbones/wrappers.
"""

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

_LAZY_IMPORTS = {
    "ConditionalTransformer2D": ".conditional_transformer_2d",
    "LabelClassifierFreeGuidance": ".wrappers",
    "AdaLNZeroBlock": ".components",
    "AdaLNZeroPatchHead": ".components",
    "ConvPatchEmbed2d": ".components",
    "FeedForward": ".components",
    "LabelEmbedder": ".components",
    "MLPTimestepEmbedder": ".components",
    "MultiheadSelfAttention": ".components",
    "build_2d_sincos_pos_embed": ".components",
    "patchify2d": ".components",
    "unpatchify2d": ".components",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
