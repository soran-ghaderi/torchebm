r"""MkDocs hook that auto-generates example pages from ``examples/`` in memory.

Discovers runnable demos under the repo ``examples/`` directory and renders one
documentation page per example, entirely in memory via ``File.generated`` (the
same mechanism as ``docs/hooks/benchmarks.py``). Nothing is written to disk.

Granularity is README-aware: a sub-folder that contains a ``README.md`` becomes
one cohesive page; a sub-folder without one is recursed into a page per script.

Curated ``docs/examples/*/index.md`` pages are preserved: they stay as each
category's "Overview" and generated pages are layered under them in the nav. A
generated page is never emitted for a ``src_uri`` that already exists on disk.

Registered in ``mkdocs.yml`` under ``hooks:``. Uses two events:

- ``on_config``  : discover, render, stash pages, rebuild the Examples nav.
- ``on_files``   : inject the stashed pages as in-memory ``File`` objects.
"""

import logging
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from mkdocs.structure.files import File

logger = logging.getLogger("mkdocs")

ROOT_DIR = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = ROOT_DIR / "examples"
DOCS_DIR = ROOT_DIR / "docs"

DEFAULT_BRANCH = "master"
CATEGORY_ALIASES = {"training_models": "training"}
EXCLUDE_NAMES = {"__init__.py", "utils.py", "main.py"}

SUPPORTED_PATTERNS = (
    "*.py",
    "*.md",
    "*.sh",
    "*.json",
    "*.yaml",
    "*.yml",
    "*.toml",
    "*.txt",
)
LANGUAGE_ALIASES = {"yml": "yaml", "md": "markdown", "py": "python", "sh": "bash"}

# Populated in on_config, consumed in on_files. src_uri -> markdown content.
_generated: dict[str, str] = {}


def _title(text: str) -> str:
    text = text.replace("_", " ").replace("/", " - ").title()
    subs = {
        "api": "API",
        "cli": "CLI",
        "cpu": "CPU",
        "gpu": "GPU",
        "ode": "ODE",
        "sde": "SDE",
        "hmc": "HMC",
        "mcmc": "MCMC",
        "ebm": "EBM",
        "mlp": "MLP",
        "json": "JSON",
        "yaml": "YAML",
        "toml": "TOML",
    }
    for pattern, repl in subs.items():
        text = re.sub(rf"\b{pattern}\b", repl, text, flags=re.IGNORECASE)
    return text


def _is_supported(file: Path) -> bool:
    return any(file.match(pattern) for pattern in SUPPORTED_PATTERNS)


def _included(file: Path) -> bool:
    return file.name.lower() != "readme.md" and file.name not in EXCLUDE_NAMES


def _has_readme(directory: Path) -> bool:
    return any((directory / name).exists() for name in ("README.md", "readme.md"))


def _supported_files_in(directory: Path) -> list[Path]:
    return sorted(f for f in directory.iterdir() if f.is_file() and _is_supported(f))


def _fence_for(text: str) -> str:
    r"""Return a backtick fence longer than any run of backticks in *text*."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _github_url(base: str, path: Path) -> str:
    slug = "tree" if path.is_dir() else "blob"
    rel = path.relative_to(ROOT_DIR).as_posix()
    return f"{base}/{slug}/{DEFAULT_BRANCH}/{rel}"


def _fix_relative_links(content: str, src_file: Path, base: str) -> str:
    r"""Rewrite relative links that resolve outside ``docs/`` to GitHub URLs."""
    link_pattern = r"\[([^\]]*)\]\((?!(?:https?|ftp)://|#)([^)]+)\)"

    def replace_link(match: re.Match) -> str:
        text = match.group(1)
        resolved = (src_file.parent / match.group(2)).resolve()
        if not resolved.exists():
            return match.group(0)
        try:
            resolved.relative_to(ROOT_DIR)
        except ValueError:
            return match.group(0)
        return f"[{text}]({_github_url(base, resolved)})"

    return re.sub(link_pattern, replace_link, content)


@dataclass
class Example:
    path: Path
    category: str

    @cached_property
    def main_file(self) -> Path | None:
        if self.path.is_file():
            return self.path
        for name in ("README.md", "readme.md"):
            readme = self.path / name
            if readme.exists():
                return readme
        md_files = sorted(self.path.glob("*.md"))
        if md_files:
            return md_files[0]
        for pattern in SUPPORTED_PATTERNS:
            files = sorted(f for f in self.path.glob(pattern) if f.is_file())
            if files:
                return files[0]
        return None

    @cached_property
    def other_files(self) -> list[Path]:
        if self.path.is_file():
            return []
        return sorted(
            f
            for f in self.path.rglob("*")
            if f.is_file() and f != self.main_file and _is_supported(f)
        )

    @cached_property
    def is_code(self) -> bool:
        return self.main_file is not None and self.main_file.suffix.lower() != ".md"

    @cached_property
    def title(self) -> str:
        if self.main_file is None or self.is_code:
            return _title(self.path.stem)
        with open(self.main_file, encoding="utf-8") as f:
            first_line = f.readline().strip()
        match = re.match(r"^#\s+(?P<title>.+)$", first_line)
        return match.group("title") if match else _title(self.path.stem)

    def _render_code_block(self, file: Path, indent: str = "") -> str:
        ext = file.suffix[1:].lower()
        lang = LANGUAGE_ALIASES.get(ext, ext)
        text = file.read_text(encoding="utf-8").rstrip()
        fence = _fence_for(text)
        block = f"{fence}{lang}\n{text}\n{fence}"
        if indent:
            block = "\n".join(indent + line if line else line for line in block.splitlines())
        return block + "\n"

    def generate(self, base: str) -> str:
        url = _github_url(base, self.path)
        parts: list[str] = [f"# {self.title}", "", f"Source <{url}>.", ""]

        if self.main_file is not None:
            if self.is_code:
                parts.append(self._render_code_block(self.main_file))
            else:
                lines = self.main_file.read_text(encoding="utf-8").splitlines(keepends=True)
                if lines and lines[0].lstrip().startswith("#"):
                    lines = lines[1:]
                parts.append(_fix_relative_links("".join(lines), self.main_file, base))
                parts.append("")
        elif self.path.is_dir() and self.other_files:
            for file in self.other_files:
                file_title = _title(str(file.relative_to(self.path).with_suffix("")))
                parts.extend([f"## {file_title}", "", self._render_code_block(file), ""])
            return "\n".join(parts)

        if self.other_files:
            parts.extend(["## Example materials", ""])
            for file in self.other_files:
                rel = file.relative_to(self.path)
                parts.append(f'??? abstract "{rel}"')
                if file.suffix.lower() != ".md":
                    parts.append(self._render_code_block(file, indent="    ").rstrip())
                else:
                    body = file.read_text(encoding="utf-8").rstrip().splitlines()
                    parts.extend(f"    {line}" if line else line for line in body)
                parts.append("")

        return "\n".join(parts)


def _walk(directory: Path, category: str) -> list[Example]:
    r"""README-aware recursion: README'd dir -> one page; else page per file."""
    if _has_readme(directory):
        return [Example(path=directory, category=category)]
    examples = [Example(path=f, category=category) for f in _supported_files_in(directory) if _included(f)]
    for sub in sorted(p for p in directory.iterdir() if p.is_dir()):
        examples.extend(_walk(sub, category))
    return examples


def _discover_examples() -> list[Example]:
    if not EXAMPLES_DIR.exists():
        return []

    examples: list[Example] = []
    for category_dir in sorted(p for p in EXAMPLES_DIR.iterdir() if p.is_dir()):
        category = category_dir.name
        logger.info("gen_examples: scanning category %s", category)
        for file in _supported_files_in(category_dir):
            if _included(file):
                examples.append(Example(path=file, category=category))
        for sub in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            examples.extend(_walk(sub, category))

    for file in _supported_files_in(EXAMPLES_DIR):
        if _included(file):
            examples.append(Example(path=file, category="general"))

    seen: set[Path] = set()
    unique: list[Example] = []
    for example in examples:
        key = example.path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(example)
    return unique


def _doc_src_path(example: Example) -> str:
    r"""Virtual ``src_uri`` under ``docs/`` mirroring ``examples/`` (aliased)."""
    if example.category == "general":
        rel = example.path.relative_to(EXAMPLES_DIR)
        return f"examples/{rel.with_suffix('.md').as_posix()}"
    alias = CATEGORY_ALIASES.get(example.category, example.category)
    rel = example.path.relative_to(EXAMPLES_DIR / example.category)
    return f"examples/{alias}/{rel.with_suffix('.md').as_posix()}"


def _tree_to_nav(tree: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for name in sorted(tree):
        value = tree[name]
        if isinstance(value, str):
            items.append({_title(Path(name).stem): value})
        else:
            items.append({_title(name): _tree_to_nav(value)})
    return items


def _build_entries(category: str, src_uris: list[str]) -> list[dict[str, Any]]:
    prefix = f"examples/{category}/"
    tree: dict[str, Any] = {}
    for src in sorted(src_uris):
        rel = src[len(prefix):] if src.startswith(prefix) else Path(src).name
        parts = rel.split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = src
    return _tree_to_nav(tree)


def _build_examples_section(
    existing_children: list[Any], by_category: dict[str, list[str]]
) -> list[Any]:
    result: list[Any] = []
    used: set[str] = set()

    for child in existing_children:
        if isinstance(child, dict) and len(child) == 1:
            (label, target), = child.items()
            parts = str(target).split("/")
            if len(parts) == 3 and parts[0] == "examples" and parts[2] == "index.md":
                category = parts[1]
                used.add(category)
                group = [{"Overview": target}, *_build_entries(category, by_category.get(category, []))]
                result.append({label: group})
                continue
        result.append(child)

    for category in sorted(c for c in by_category if c not in used and c != "general"):
        result.append({_title(category): _build_entries(category, by_category[category])})

    for src in by_category.get("general", []):
        result.append({_title(Path(src).stem): src})

    return result


def _inject_nav(config: Any, by_category: dict[str, list[str]]) -> None:
    nav = config.get("nav") or []
    new_nav: list[Any] = []
    for item in nav:
        if isinstance(item, dict) and "Examples" in item:
            new_nav.append({"Examples": _build_examples_section(item["Examples"], by_category)})
        else:
            new_nav.append(item)
    config["nav"] = new_nav


def on_config(config: Any) -> Any:
    r"""Discover examples, render virtual pages, and rebuild the Examples nav."""
    base = str(config.get("repo_url") or "").rstrip("/")
    _generated.clear()

    by_category: dict[str, list[str]] = {}
    for example in _discover_examples():
        src_uri = _doc_src_path(example)
        if (DOCS_DIR / src_uri).exists():
            continue  # curated on-disk page wins
        _generated[src_uri] = example.generate(base)
        parts = src_uri.split("/")
        category = parts[1] if len(parts) > 2 else "general"
        by_category.setdefault(category, []).append(src_uri)

    _inject_nav(config, by_category)
    logger.info("gen_examples: prepared %d virtual example pages", len(_generated))
    return config


def on_files(files: Any, config: Any) -> Any:
    r"""Inject the stashed pages as in-memory files (never clobber real ones)."""
    existing = {f.src_uri for f in files}
    added = 0
    for src_uri, content in _generated.items():
        if src_uri not in existing:
            files.append(File.generated(config, src_uri, content=content))
            added += 1
    if added:
        logger.info("gen_examples: injected %d virtual files", added)
    return files
