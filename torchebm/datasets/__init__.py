__all__ = [
    "GaussianMixtureDataset",
    "EightGaussiansDataset",
    "TwoMoonsDataset",
    "SwissRollDataset",
    "CircleDataset",
    "CheckerboardDataset",
    "PinwheelDataset",
    "GridDataset",
]

_LAZY_IMPORTS = {name: ".generators" for name in __all__}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
