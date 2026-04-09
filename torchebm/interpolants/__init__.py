r"""
Stochastic interpolants for generative modeling.

Interpolants define conditional probability paths between source (noise)
and target (data) distributions, parameterized by schedules α(t) and σ(t).
"""

__all__ = [
    "LinearInterpolant",
    "CosineInterpolant",
    "VariancePreservingInterpolant",
]

_LAZY_IMPORTS = {
    "LinearInterpolant": ".linear",
    "CosineInterpolant": ".cosine",
    "VariancePreservingInterpolant": ".variance_preserving",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
