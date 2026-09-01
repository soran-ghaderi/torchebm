"""Model namespace.

TorchEBM is designed for plug-and-play experimentation:
- try different losses with the same backbone
- try different backbones with the same loss
- use samplers as long as the model signature matches

This package therefore exposes *reusable building blocks* under
`torchebm.models.components` and a small set of generic backbones/wrappers.
"""

__all__ = [
    "DiT",
    "TimeConditionedMLP",
    "dit_s_2",
    "dit_s_4",
    "dit_s_8",
    "dit_b_2",
    "dit_b_4",
    "dit_b_8",
    "dit_l_2",
    "dit_l_4",
    "dit_l_8",
    "dit_xl_2",
    "dit_xl_4",
    "dit_xl_8",
    "ConditionalTransformer2D",
    "ClassifierFreeGuidance",
    "InteractionModel",
    "EqMEnergy",
    "MLPTimestepEmbedder",
    "LabelEmbedder",
    "build_2d_sincos_pos_embed",
    "ConvPatchEmbed2d",
    "patchify2d",
    "unpatchify2d",
    "FeedForward",
    "MultiheadSelfAttention",
    "AdaLNZeroBlock",
    "AdaLNZeroMLPBlock",
    "AdaLNZeroLinearHead",
    "AdaLNZeroPatchHead",
]

_LAZY_IMPORTS = {
    "DiT": ".dit",
    "TimeConditionedMLP": ".mlp",
    "dit_s_2": ".dit",
    "dit_s_4": ".dit",
    "dit_s_8": ".dit",
    "dit_b_2": ".dit",
    "dit_b_4": ".dit",
    "dit_b_8": ".dit",
    "dit_l_2": ".dit",
    "dit_l_4": ".dit",
    "dit_l_8": ".dit",
    "dit_xl_2": ".dit",
    "dit_xl_4": ".dit",
    "dit_xl_8": ".dit",
    "ConditionalTransformer2D": ".conditional_transformer_2d",
    "ClassifierFreeGuidance": ".wrappers",
    "InteractionModel": ".wrappers",
    "EqMEnergy": ".wrappers",
    "AdaLNZeroBlock": ".components",
    "AdaLNZeroMLPBlock": ".components",
    "AdaLNZeroLinearHead": ".components",
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
