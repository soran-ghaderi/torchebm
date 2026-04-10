"""
TorchEBM: Energy-Based Modeling library for PyTorch, offering tools for sampling, inference, and learning in complex distributions.

All subpackages are lazy-loaded so that ``import torchebm`` is near-instant.
Access any subpackage normally (``torchebm.samplers``, ``torchebm.core``, etc.)
and it will be imported on first use.
"""

# Version information
from ._version import __version__

__all__ = [
    "core",
    "samplers",
    "losses",
    "utils",
    "cuda",
    "datasets",
    "models",
    "integrators",
    "interpolants",
    "__version__",
]

_SUBPACKAGES = {
    "core",
    "samplers",
    "losses",
    "utils",
    "cuda",
    "datasets",
    "models",
    "integrators",
    "interpolants",
}


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        import importlib

        module = importlib.import_module(f".{name}", __package__)
        globals()[name] = module  # cache so __getattr__ isn't called again
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
