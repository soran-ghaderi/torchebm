"""Deprecation ledger, derived entirely from the code.

Deprecations self-register in `torchebm._deprecation.DEPRECATIONS` when
their declaring module is imported. This file imports every package module
at collection time, then enforces:

- every module imports (import-health; optional-dependency modules may skip
  with the missing dependency named, and may not declare deprecations);
- every declaration site found by AST scan actually registered;
- no deprecation warning is emitted outside the `torchebm._deprecation` API
  (AST lint over warn calls, categories, and warn_once imports);
- no registered removal window has closed for the next release (latest git
  tag + 1 patch, falling back to the installed version on sdist builds;
  skipped with a reason when neither yields a release, e.g. shallow tagless
  CI checkouts).

There is intentionally no per-deprecation content here: adding a deprecation
means declaring it at its site, nothing else.

Accepted lint residuals (code review's job, not static analysis): a
deprecation message hidden behind a variable or built dynamically, and
categories reached via getattr tricks.
"""

import ast
import importlib
import subprocess
from pathlib import Path

import pytest

import torchebm
from torchebm._deprecation import (
    DEPRECATIONS,
    TorchEBMDeprecationWarning,
    _parse_release,
    declare_deprecation,
    deprecated,
)

_PACKAGE_ROOT = Path(torchebm.__file__).parent
_REPO_ROOT = _PACKAGE_ROOT.parent

#: Modules whose top-level import may fail when the named optional
#: dependency is absent. Deprecations must not be declared in these.
_OPTIONAL_DEPENDENCY_MODULES = {"torchebm.cuda.fused_langevin": "triton"}

#: Files that may import warn_once: the internal and public compat
#: re-exports. Everything else must go through the deprecation API.
_WARN_ONCE_REEXPORTS = {
    "torchebm/core/base_module.py",
    "torchebm/core/__init__.py",
}

#: Emitting any of these outside torchebm/_deprecation.py is forbidden.
_FORBIDDEN_CATEGORIES = {
    "DeprecationWarning",
    "PendingDeprecationWarning",
    "FutureWarning",
    "TorchEBMDeprecationWarning",
}


def _module_name(path: Path) -> str:
    rel = path.relative_to(_REPO_ROOT).with_suffix("")
    parts = rel.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


_ALL_MODULE_FILES = sorted(_PACKAGE_ROOT.rglob("*.py"))
_IMPORT_FAILURES = []
_SKIPPED_OPTIONAL = set()
for _py in _ALL_MODULE_FILES:
    _mod = _module_name(_py)
    try:
        importlib.import_module(_mod)
    except ImportError as _e:
        _dep = _OPTIONAL_DEPENDENCY_MODULES.get(_mod)
        if _dep is not None and _dep in str(_e):
            _SKIPPED_OPTIONAL.add(_mod)
        else:
            _IMPORT_FAILURES.append((_mod, repr(_e)))
    except Exception as _e:
        _IMPORT_FAILURES.append((_mod, repr(_e)))

_REGISTRY = dict(DEPRECATIONS)


def test_every_package_module_imports():
    assert _ALL_MODULE_FILES, "package walk found no modules; discovery is broken"
    assert not _IMPORT_FAILURES, (
        f"package modules failed to import: {_IMPORT_FAILURES}"
    )


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _declaration_site_counts() -> dict:
    r"""Per-module count of declare_deprecation calls and @deprecated uses."""
    counts: dict = {}
    for py in _ALL_MODULE_FILES:
        if py.name == "_deprecation.py":
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node) in (
                "declare_deprecation",
                "deprecated",
            ):
                mod = _module_name(py)
                counts[mod] = counts.get(mod, 0) + 1
    return counts


def test_every_declaration_site_registered():
    sites = _declaration_site_counts()
    in_skipped = set(sites) & _SKIPPED_OPTIONAL
    assert not in_skipped, (
        f"deprecations declared in optional-dependency modules cannot be "
        f"gated and are not allowed: {sorted(in_skipped)}"
    )
    registered: dict = {}
    for module, _ in _REGISTRY:
        registered[module] = registered.get(module, 0) + 1
    assert sites == registered, (
        f"declaration sites and registry disagree (a site inside a function "
        f"body registers late or never; declare at module level): "
        f"sites={sites}, registered={registered}"
    )


def _gate_version() -> str:
    r"""The next release: latest git tag + 1 patch, so the gate fires before
    the closing release is tagged. Falls back to the installed version
    (sdist/tag builds, where that version is the release itself)."""
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        tag = out.stdout.strip().lstrip("v")
        if out.returncode == 0 and tag:
            major, minor, patch = _parse_release(tag)
            return f"{major}.{minor}.{patch + 1}"
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return torchebm.__version__


_GATE_VERSION = _gate_version()
try:
    _parse_release(_GATE_VERSION)
    _GATE_SKIP = ""
except ValueError:
    _GATE_SKIP = (
        f"no release context to gate against (version {_GATE_VERSION!r}; "
        "tagless checkout with an unstamped install)"
    )


@pytest.mark.parametrize(
    "info", list(_REGISTRY.values()), ids=lambda i: f"{i.module}:{i.name}"
)
def test_removal_window_open(info):
    if _GATE_SKIP:
        pytest.skip(_GATE_SKIP)
    assert not info.removal_due(_GATE_VERSION), (
        f"The removal window for '{info.name}' (deprecated in {info.since}, "
        f"open for {info.grace} releases) closes at {_GATE_VERSION}: delete "
        f"{info.removal or 'the deprecated path'} together with its "
        f"declaration, and point users at {info.replacement}."
    )


def test_window_rule():
    entry = declare_deprecation(
        module=__name__, name="rule-probe", since="0.8.3", replacement="x"
    )
    try:
        assert not entry.removal_due("0.8.3")
        assert not entry.removal_due("0.8.4")
        assert entry.removal_due("0.8.5")
        assert entry.removal_due("0.9.0")
        assert entry.removal_due("1.0.0")
        assert not entry.removal_due("0.8.2")
        assert not entry.removal_due("0.7.9")
        assert not entry.removal_due("0.8.4.dev2+g0142c5d43.d20260713")
        assert entry.removal_due("0.9.0rc1")
        assert entry.removal_due("1!0.9.0")
        with pytest.raises(ValueError, match="release"):
            entry.removal_due("garbage")
    finally:
        DEPRECATIONS.pop(entry.key)


def _lint_file(py: Path, errors: list) -> None:
    tree = ast.parse(py.read_text(encoding="utf-8"))
    rel = py.relative_to(_REPO_ROOT).as_posix()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "warn_once" and rel not in _WARN_ONCE_REEXPORTS:
                    errors.append(
                        f"{rel}:{node.lineno}: import of warn_once; emit "
                        "deprecations via torchebm._deprecation"
                    )
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name == "warn_once":
            errors.append(
                f"{rel}:{node.lineno}: warn_once call; use "
                "declare_deprecation(...).warn() or @deprecated"
            )
        if name != "warn":
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            names = {
                n.id for n in ast.walk(arg) if isinstance(n, ast.Name)
            } | {
                n.attr for n in ast.walk(arg) if isinstance(n, ast.Attribute)
            }
            if names & _FORBIDDEN_CATEGORIES:
                errors.append(
                    f"{rel}:{node.lineno}: raw deprecation-category warning; "
                    "use the torchebm._deprecation API"
                )
            for const in ast.walk(arg):
                if (
                    isinstance(const, ast.Constant)
                    and isinstance(const.value, str)
                    and "deprecat" in const.value.lower()
                ):
                    errors.append(
                        f"{rel}:{node.lineno}: warnings.warn with a "
                        "deprecation message; use the torchebm._deprecation "
                        "API"
                    )


def test_no_deprecation_warnings_outside_the_api():
    errors = []
    for py in _ALL_MODULE_FILES:
        if py.name == "_deprecation.py":
            continue
        _lint_file(py, errors)
    assert not errors, "\n".join(errors)


def test_declare_is_idempotent_and_conflicts_raise():
    a = declare_deprecation(
        module=__name__, name="idem-probe", since="0.8.3", replacement="x"
    )
    try:
        b = declare_deprecation(
            module=__name__, name="idem-probe", since="0.8.3", replacement="x"
        )
        assert a == b
        with pytest.raises(ValueError, match="conflicting"):
            declare_deprecation(
                module=__name__, name="idem-probe", since="0.8.4", replacement="x"
            )
    finally:
        DEPRECATIONS.pop(a.key)


def test_declare_validates_since():
    with pytest.raises(ValueError, match="release"):
        declare_deprecation(
            module=__name__, name="bad-since", since="soon", replacement="x"
        )


def test_warn_requires_registration():
    entry = declare_deprecation(
        module=__name__, name="warn-probe", since="0.8.3", replacement="x"
    )
    DEPRECATIONS.pop(entry.key)
    with pytest.raises(RuntimeError, match="not registered"):
        entry.warn()


def test_warn_dedups_per_qualifier_and_attributes_to_caller():
    import warnings as w

    entry = declare_deprecation(
        module=__name__, name="dedup-probe", since="0.8.3", replacement="the new thing"
    )
    try:
        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            entry.warn(qualifier="a")
            entry.warn(qualifier="a")
            entry.warn(qualifier="b")
        assert len(caught) == 2
        assert all(
            issubclass(c.category, TorchEBMDeprecationWarning) for c in caught
        )
        assert "the new thing" in str(caught[0].message)
        assert caught[0].filename == __file__
    finally:
        DEPRECATIONS.pop(entry.key)


def test_decorator_registers_warns_and_preserves_identity():
    @deprecated(since="0.8.3", replacement="NewThing")
    class OldThing:
        def __init__(self, value):
            self.value = value

    key = (__name__, OldThing.__qualname__)
    try:
        assert key in DEPRECATIONS
        assert "NewThing" in OldThing.__deprecated__
        with pytest.warns(DeprecationWarning, match="NewThing") as caught:
            obj = OldThing(3)
        assert obj.value == 3
        assert isinstance(obj, OldThing)
        assert caught[0].filename == __file__

        class Child(OldThing):
            pass

        with pytest.warns(TorchEBMDeprecationWarning):
            assert Child(5).value == 5
    finally:
        DEPRECATIONS.pop(key, None)


def test_decorator_on_function():
    @deprecated(since="0.8.3", replacement="new_fn")
    def old_fn(x):
        return x + 1

    key = (__name__, old_fn.__qualname__)
    try:
        assert key in DEPRECATIONS
        with pytest.warns(TorchEBMDeprecationWarning, match="new_fn"):
            assert old_fn(1) == 2
    finally:
        DEPRECATIONS.pop(key, None)
