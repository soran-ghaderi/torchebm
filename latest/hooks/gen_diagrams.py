r"""MkDocs hook that renders live component views from the installed torchebm.

Markers in any markdown page are replaced at build time:

    <!-- torchebm:diagram components -->    mermaid: the composition map
    <!-- torchebm:diagram samplers -->      mermaid: class tree of a subpackage
    <!-- torchebm:diagram energies -->      mermaid: analytic energies in core
    <!-- torchebm:cards components -->      Material grid cards, one per subpackage
    <!-- torchebm:tree packages -->         the package layout tree
    <!-- torchebm:table contracts -->       root base classes + docstring contracts

Tree keys: ``samplers``, ``losses``, ``integrators``, ``interpolants``,
``couplings``, ``datasets``, ``energies``. Everything is introspected from the
package's public exports and class hierarchies, so new components appear in
the docs automatically; nothing is hardcoded here or in the pages. New
subpackages get a card automatically (default icon, module docstring as the
blurb) until they receive an entry in ``CARD_META``.
"""

import enum
import importlib
import inspect
import logging
import posixpath
import re

logger = logging.getLogger("mkdocs")

MARKER = re.compile(r"<!--\s*torchebm:diagram\s+([a-z_]+)\s*-->")
CARDS_MARKER = re.compile(r"<!--\s*torchebm:cards\s+components\s*-->")
TREE_MARKER = re.compile(r"<!--\s*torchebm:tree\s+packages\s*-->")
CONTRACTS_MARKER = re.compile(r"<!--\s*torchebm:table\s+contracts\s*-->")

TREE_PACKAGES = {
    "samplers": "torchebm.samplers",
    "losses": "torchebm.losses",
    "integrators": "torchebm.integrators",
    "interpolants": "torchebm.interpolants",
    "couplings": "torchebm.couplings",
    "datasets": "torchebm.datasets",
}

# Curated icon per subpackage; new subpackages fall back to the default icon.
CARD_ICONS = {
    "core": "material/function-variant",
    "samplers": "material/chart-scatter-plot",
    "losses": "material/scale-balance",
    "interpolants": "material/sine-wave",
    "couplings": "material/vector-link",
    "integrators": "material/math-integral",
    "models": "material/brain",
    "datasets": "material/database-search",
    "utils": "material/tools",
    "cuda": "material/rocket-launch",
}
DEFAULT_CARD_ICON = "material/cube-outline"

# One-line comment per subpackage in the generated layout tree; new
# subpackages fall back to their module docstring's first line.
PACKAGE_BLURBS = {
    "core": "Base classes, analytic energies, schedulers, TorchEBMModule",
    "samplers": "MCMC, optimization, and flow/diffusion samplers",
    "losses": "Training objectives (CD, score matching, equilibrium/energy matching)",
    "interpolants": "Noise-to-data probability paths",
    "couplings": "Pairings between noise and data batches",
    "integrators": "Numerical integrators for SDE/ODE/Hamiltonian dynamics",
    "models": "Neural architectures used as energies or fields",
    "datasets": "Synthetic data generators",
    "utils": "Shared helpers",
    "cuda": "Custom CUDA kernels",
}


def _exported_classes(module_name: str) -> list[type]:
    mod = importlib.import_module(module_name)
    classes = []
    for name in sorted(getattr(mod, "__all__", dir(mod))):
        try:
            obj = getattr(mod, name)
        except (AttributeError, ImportError):
            continue
        if inspect.isclass(obj) and not issubclass(obj, enum.Enum):
            classes.append(obj)
    return classes


def _torchebm_chain(cls: type) -> list[type]:
    r"""The class and its torchebm ``Base*`` ancestors, subclass first.

    Infrastructure mixins (``TorchEBMModule``, ``Schedulable``, ...) are not
    part of a family's conceptual tree and are excluded.
    """
    chain = [cls]
    chain += [
        c
        for c in cls.__mro__[1:]
        if (c.__module__ or "").startswith("torchebm")
        and c.__name__.startswith("Base")
    ]
    return chain


def _concrete(classes: list[type]) -> list[type]:
    return [c for c in classes if not c.__name__.startswith("Base")]


def _fence(body: str) -> str:
    return f"```mermaid\n{body}\n```"


def _tree_diagram(module_name: str) -> str:
    r"""``graph TD`` of Base classes (stadium nodes) to concrete classes."""
    edges: list[tuple[str, str]] = []
    bases: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for cls in _exported_classes(module_name):
        chain = _torchebm_chain(cls)
        if len(chain) < 2:
            continue  # standalone helper types (e.g. result dataclasses)
        for child, parent in zip(chain[:-1], chain[1:]):
            edge = (parent.__name__, child.__name__)
            if edge in seen:
                continue
            seen.add(edge)
            edges.append(edge)
            if parent.__name__.startswith("Base"):
                bases.add(parent.__name__)
            if child.__name__.startswith("Base"):
                bases.add(child.__name__)
    lines = ["graph TD"]
    for name in sorted(bases):
        lines.append(f'    {name}(["{name}"])')
    for parent, child in edges:
        lines.append(f"    {parent} --> {child}")
    return _fence("\n".join(lines))


def _components_diagram() -> str:
    r"""``graph LR`` composition map with live per-package export counts."""
    counts = {
        key: len(_concrete(_exported_classes(mod))) for key, mod in TREE_PACKAGES.items()
    }
    energies = len(
        [
            c
            for c in _concrete(_exported_classes("torchebm.core"))
            if any(b.__name__ == "BaseModel" for b in c.__mro__)
        ]
    )
    body = "\n".join(
        [
            "graph LR",
            f'    field["energy / field<br/>core: {energies} analytic energies · models"]',
            f'    interp["interpolants ({counts["interpolants"]})"]',
            f'    coup["couplings ({counts["couplings"]})"]',
            f'    integ["integrators ({counts["integrators"]})"]',
            f'    samp["samplers ({counts["samplers"]})"]',
            f'    loss["objectives ({counts["losses"]})"]',
            f'    data[("datasets ({counts["datasets"]})")]',
            '    out(("samples"))',
            "    field --> samp",
            "    field --> loss",
            "    interp --> loss",
            "    interp --> samp",
            "    coup --> loss",
            "    integ --> samp",
            "    samp -- negatives --> loss",
            "    data --> loss",
            "    samp --> out",
        ]
    )
    return _fence(body)


def _energies_diagram() -> str:
    r"""Analytic energies in ``torchebm.core`` under ``BaseModel``."""
    lines = ["graph TD", '    BaseModel(["BaseModel"])']
    for cls in _concrete(_exported_classes("torchebm.core")):
        chain = [c.__name__ for c in _torchebm_chain(cls)]
        if "BaseModel" in chain[1:]:
            lines.append(f"    BaseModel --> {cls.__name__}")
    return _fence("\n".join(lines))


def _subpackages() -> list[str]:
    r"""torchebm subpackage names, curated order first, new ones appended."""
    mod = importlib.import_module("torchebm")
    exported = [n for n in getattr(mod, "__all__", []) if not n.startswith("__")]
    ordered = [n for n in CARD_ICONS if n in exported]
    ordered += [n for n in exported if n not in CARD_ICONS]
    return ordered


def _components_cards(page_src_uri: str) -> str:
    r"""Compact Material ``grid cards`` block: one icon + name card per subpackage.

    The package list is introspected at build time, so cards can never drift;
    each card links to the subpackage's API page.
    """
    page_dir = posixpath.dirname(page_src_uri)
    lines = ['<div class="grid cards" markdown>', ""]
    for name in _subpackages():
        icon = CARD_ICONS.get(name, DEFAULT_CARD_ICON)
        api_page = f"api/torchebm/{name}/index.md"
        link = posixpath.relpath(api_page, page_dir) if page_dir else api_page
        title = name.upper() if name == "cuda" else name.capitalize()
        lines.append(
            f"-   [:{icon.replace('/', '-')}:{{ .lg .middle }} __{title}__]({link})"
        )
        lines.append("")
    lines.append("</div>")
    return "\n".join(lines)


def _package_blurb(name: str) -> str:
    if name in PACKAGE_BLURBS:
        return PACKAGE_BLURBS[name]
    try:
        doc = importlib.import_module(f"torchebm.{name}").__doc__ or ""
        return doc.strip().splitlines()[0] if doc.strip() else ""
    except Exception:  # noqa: BLE001
        return ""


def _packages_tree() -> str:
    r"""The ``torchebm/`` layout tree, one line per live subpackage."""
    subs = _subpackages()
    lines = ["```", "torchebm/"]
    for i, name in enumerate(subs):
        branch = "└──" if i == len(subs) - 1 else "├──"
        lines.append(f"{branch} {name + '/':<14} # {_package_blurb(name)}")
    lines.append("```")
    return "\n".join(lines)


def _contracts_table() -> str:
    r"""Root base classes in ``torchebm.core`` with their docstring contracts.

    Root means no ``Base*`` ancestor of its own; family-level bases
    (e.g. ``BaseContrastiveDivergence``) belong to their subpackage trees.
    """
    core = importlib.import_module("torchebm.core")
    rows: list[tuple[str, str]] = []
    for name in sorted(getattr(core, "__all__", [])):
        obj = getattr(core, name, None)
        if not inspect.isclass(obj):
            continue
        is_root_base = name.startswith("Base") and len(_torchebm_chain(obj)) == 1
        if not (is_root_base or name == "TorchEBMModule"):
            continue
        doc = (inspect.getdoc(obj) or "").strip().splitlines()
        rows.append((name, doc[0] if doc else ""))
    lines = ["| Base class | Contract |", "| --- | --- |"]
    for name, desc in rows:
        lines.append(f"| `{name}` | {desc} |")
    return "\n".join(lines)


def _render(key: str) -> str | None:
    try:
        if key == "components":
            return _components_diagram()
        if key == "energies":
            return _energies_diagram()
        if key in TREE_PACKAGES:
            return _tree_diagram(TREE_PACKAGES[key])
    except Exception as exc:  # noqa: BLE001 - never fail the build over a diagram
        logger.warning("gen_diagrams: could not render %r (%s)", key, exc)
        return None
    logger.warning("gen_diagrams: unknown diagram key %r", key)
    return None


def on_page_markdown(markdown: str, page, config, files) -> str:
    def replace(match: re.Match) -> str:
        rendered = _render(match.group(1))
        return rendered if rendered is not None else ""

    markdown = MARKER.sub(replace, markdown)
    if CARDS_MARKER.search(markdown):
        try:
            cards = _components_cards(page.file.src_uri)
        except Exception as exc:  # noqa: BLE001 - never fail the build
            logger.warning("gen_diagrams: could not render cards (%s)", exc)
            cards = ""
        markdown = CARDS_MARKER.sub(lambda _: cards, markdown)
    for marker, renderer in ((TREE_MARKER, _packages_tree), (CONTRACTS_MARKER, _contracts_table)):
        if marker.search(markdown):
            try:
                block = renderer()
            except Exception as exc:  # noqa: BLE001 - never fail the build
                logger.warning("gen_diagrams: could not render %s (%s)", renderer.__name__, exc)
                block = ""
            markdown = marker.sub(lambda _: block, markdown)
    return markdown
