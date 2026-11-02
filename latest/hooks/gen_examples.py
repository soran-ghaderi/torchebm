r"""MkDocs hook that auto-generates example pages from ``examples/`` in memory.

The examples tree is a uniform **folder-per-example gallery**: every example is a
self-contained directory that carries a ``meta.yaml`` (the discovery signal and the
single source of metadata). A directory with ``meta.yaml`` is one example; the hook
never recurses into it. Anything without ``meta.yaml`` (helpers, the old loose
examples, ``_shared``/``vendor``/``__pycache__``) is ignored, so the legacy files can
stay on disk untouched while only the new tree renders.

Each example renders to one in-memory docs page (``File.generated`` - nothing is
written to disk):

    front-matter tags -> # title -> summary + level line
    -> run command (from meta.entrypoint)
    -> folder prose from index.md (optional, links rewritten to GitHub)
    -> code, one section per file (entrypoint first, plot.py excluded)

Pages are text-only by design: no thumbnails, no embedded images, no committed
figure assets. The learning narrative is driven by a global ``order`` integer in
each ``meta.yaml`` (numeric folder prefixes like ``00-`` are readability hints
only). The nav and the per-tier index tables are sorted by ``order``
(filename-sort fallback).

Registered in ``mkdocs.yml`` under ``hooks:``. Uses ``on_config`` (discover, render,
rebuild nav) and ``on_files`` (inject the pages).
"""

import logging
import re
from dataclasses import dataclass
from functools import cached_property
from math import inf
from pathlib import Path
from typing import Any

import yaml
from mkdocs.plugins import event_priority
from mkdocs.structure.files import File

logger = logging.getLogger("mkdocs")

ROOT_DIR = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = ROOT_DIR / "examples"
DOCS_DIR = ROOT_DIR / "docs"

DEFAULT_BRANCH = "master"
META_FILE = "meta.yaml"
INDEX_FILE = "index.md"  # a folder's prose, mirroring the docs/ convention
EXCLUDE_DIRS = {"__pycache__", "vendor", "assets", "configs"}
EXCLUDE_CODE = {"plot.py"}  # viz companions: kept in repo, not shown in the lesson page
CODE_SUFFIXES = (".py", ".sh")
LANGUAGE_ALIASES = {"yml": "yaml", "md": "markdown", "py": "python", "sh": "bash"}

# Populated in on_config, consumed in on_files. src_uri -> markdown content.
_generated: dict[str, str] = {}


def _title(text: str) -> str:
    text = text.replace("_", " ").replace("-", " ").replace("/", " - ").title()
    subs = {
        "api": "API", "cli": "CLI", "cpu": "CPU", "gpu": "GPU",
        "ode": "ODE", "sde": "SDE", "hmc": "HMC", "mcmc": "MCMC", "mc": "MC",
        "ebm": "EBM", "mlp": "MLP", "eqm": "EqM", "cfg": "CFG", "dsm": "DSM",
        "ssm": "SSM", "pcd": "PCD", "cd": "CD", "sm": "SM", "ema": "EMA",
        "nfe": "NFE", "ood": "OOD", "2d": "2D", "3d": "3D",
        "json": "JSON", "yaml": "YAML", "toml": "TOML",
    }
    for pattern, repl in subs.items():
        text = re.sub(rf"\b{pattern}\b", repl, text, flags=re.IGNORECASE)
    return text


def _nav_label(segment: str) -> str:
    r"""Pretty label for a folder segment: drop the numeric ``NN-`` prefix."""
    return _title(re.sub(r"^\d+[-_]", "", segment))


def _fence_for(text: str) -> str:
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


def _render_code_block(file: Path, indent: str = "") -> str:
    ext = file.suffix[1:].lower()
    lang = LANGUAGE_ALIASES.get(ext, ext)
    text = file.read_text(encoding="utf-8").rstrip()
    fence = _fence_for(text)
    block = f"{fence}{lang}\n{text}\n{fence}"
    if indent:
        block = "\n".join(indent + line if line else line for line in block.splitlines())
    return block + "\n"


@dataclass
class Example:
    path: Path  # the example folder (contains meta.yaml)

    @cached_property
    def meta(self) -> dict[str, Any]:
        f = self.path / META_FILE
        try:
            return yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:  # noqa: BLE001
            logger.warning("gen_examples: bad %s (%s)", f, exc)
            return {}

    @cached_property
    def order(self) -> float:
        value = self.meta.get("order")
        return float(value) if isinstance(value, (int, float)) else inf

    @cached_property
    def summary(self) -> str:
        return str(self.meta.get("summary", "")).strip()

    @cached_property
    def difficulty(self) -> str:
        return str(self.meta.get("difficulty", "")).strip().lower()

    @cached_property
    def status(self) -> str:
        return str(self.meta.get("status", "")).strip().lower()

    @cached_property
    def tags(self) -> list[str]:
        tags = self.meta.get("tags") or []
        return [str(t) for t in tags] if isinstance(tags, list) else []

    @cached_property
    def entrypoint(self) -> str:
        return str(self.meta.get("entrypoint", "main.py"))

    @cached_property
    def readme(self) -> Path | None:
        for name in (INDEX_FILE,):
            if (self.path / name).exists():
                return self.path / name
        return None

    @cached_property
    def title(self) -> str:
        if self.meta.get("title"):
            return str(self.meta["title"])
        if self.readme is not None:
            first = self.readme.read_text(encoding="utf-8").splitlines()[:1]
            match = re.match(r"^#\s+(?P<t>.+)$", first[0].strip()) if first else None
            if match:
                return match.group("t")
        return _nav_label(self.path.name)

    @cached_property
    def code_files(self) -> list[Path]:
        files = [
            f for f in self.path.iterdir()
            if f.is_file()
            and f.suffix.lower() in CODE_SUFFIXES
            and f.name != META_FILE
            and f.name not in EXCLUDE_CODE
        ]
        ep = self.entrypoint
        return sorted(files, key=lambda f: (f.name != ep, f.name != "main.py", f.name))

    def _badge_line(self) -> str:
        bits: list[str] = []
        if self.difficulty:
            bits.append(self.difficulty)
        if self.status and self.status != "stable":
            bits.append(self.status)
        if self.summary:
            bits.append(self.summary)
        return " · ".join(bits)

    def _run_block(self) -> str:
        rel = self.path.relative_to(ROOT_DIR).as_posix()
        return f"Run it:\n\n```bash\npython {rel}/{self.entrypoint}\n```\n"

    def _render_code_sections(self) -> str:
        parts: list[str] = []
        heading = len(self.code_files) > 1
        for file in self.code_files:
            if heading:
                parts.append(f"## {file.name}")
                parts.append("")
            parts.append(_render_code_block(file).rstrip())
            parts.append("")
        return "\n".join(parts)

    def generate(self, base: str) -> str:
        parts: list[str] = []
        if self.tags:
            parts += ["---", yaml.safe_dump({"tags": self.tags}).strip(), "---", ""]
        parts += [f"# {self.title}", ""]
        badge = self._badge_line()
        if badge:
            parts += [badge, ""]
        parts += [self._run_block(), ""]
        if self.readme is not None:
            lines = self.readme.read_text(encoding="utf-8").splitlines(keepends=True)
            if lines and lines[0].lstrip().startswith("#"):
                lines = lines[1:]
            body = _fix_relative_links("".join(lines).strip(), self.readme, base)
            if body:
                parts += [body, ""]
        code = self._render_code_sections()
        if code:
            parts += [code]
        return "\n".join(parts)


def _collect(directory: Path, out: list[Example]) -> None:
    for sub in sorted(p for p in directory.iterdir() if p.is_dir()):
        if sub.name in EXCLUDE_DIRS or sub.name.startswith((".", "_")):
            continue
        if (sub / META_FILE).exists():
            out.append(Example(path=sub))
        else:
            _collect(sub, out)


def _discover_examples() -> list[Example]:
    if not EXAMPLES_DIR.exists():
        return []
    examples: list[Example] = []
    _collect(EXAMPLES_DIR, examples)
    return examples


def _doc_src_path(example: Example) -> str:
    rel = example.path.relative_to(EXAMPLES_DIR)
    return f"examples/{rel.as_posix()}.md"


def _render_table(items: list[tuple[str, Example]], category: str) -> str:
    if not items:
        return ""
    prefix = f"examples/{category}/"
    lines = ["| Example | Summary | Level |", "| --- | --- | --- |"]
    for src, ex in sorted(items, key=lambda it: (it[1].order, it[0])):
        rel = src[len(prefix):] if src.startswith(prefix) else Path(src).name
        level = ex.difficulty
        if ex.status and ex.status != "stable":
            level = f"{level} ({ex.status})" if level else ex.status
        lines.append(f"| [{ex.title}]({rel}) | {ex.summary} | {level} |")
    return "\n".join(lines)


def _render_tier_index(category: str, items: list[tuple[str, Example]], base: str) -> str:
    readme = EXAMPLES_DIR / category / INDEX_FILE
    if readme.exists():
        head = _fix_relative_links(readme.read_text(encoding="utf-8").rstrip(), readme, base)
    else:
        head = f"# {_nav_label(category)}"
    table = _render_table(items, category)
    return head + ("\n\n" + table if table else "") + "\n"


def _render_overview(base: str) -> str:
    r"""The Examples section landing page, rendered from ``examples/index.md``.

    The curriculum spec in the repository is the single source: it is what a
    contributor reads on GitHub and what a user reads on the site.
    """
    readme = EXAMPLES_DIR / INDEX_FILE
    if not readme.exists():
        return "# Examples\n"
    body = _fix_relative_links(readme.read_text(encoding="utf-8").rstrip(), readme, base)
    front = "---\ntitle: Examples\nicon: fontawesome/regular/image\n---\n\n"
    return front + body + "\n"


def _eff_order(value: Any, order_map: dict[str, float]) -> float:
    if isinstance(value, str):
        return order_map.get(value, inf)
    return min((_eff_order(v, order_map) for v in value.values()), default=inf)


def _tree_to_nav(
    tree: dict[str, Any], order_map: dict[str, float], title_map: dict[str, str]
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for name, value in sorted(tree.items(), key=lambda kv: (_eff_order(kv[1], order_map), kv[0])):
        if isinstance(value, str):
            items.append({title_map.get(value) or _nav_label(Path(name).stem): value})
        else:
            items.append({_nav_label(name): _tree_to_nav(value, order_map, title_map)})
    return items


def _build_entries(
    category: str, src_uris: list[str], order_map: dict[str, float], title_map: dict[str, str]
) -> list[dict[str, Any]]:
    prefix = f"examples/{category}/"
    tree: dict[str, Any] = {}
    for src in src_uris:
        rel = src[len(prefix):] if src.startswith(prefix) else Path(src).name
        parts = rel.split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = src
    return _tree_to_nav(tree, order_map, title_map)


def _build_examples_section(
    existing_children: list[Any],
    by_category: dict[str, list[str]],
    order_map: dict[str, float],
    title_map: dict[str, str],
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
                entries = _build_entries(category, by_category.get(category, []), order_map, title_map)
                result.append({label: [{"Overview": target}, *entries]})
                continue
        result.append(child)

    for category in sorted(c for c in by_category if c not in used):
        result.append({_nav_label(category): _build_entries(category, by_category[category], order_map, title_map)})

    return result


def _examples_nav_tiers(nav: list[Any]) -> list[str]:
    r"""Tier categories referenced as ``examples/<cat>/index.md`` in the nav."""
    tiers: list[str] = []
    for item in nav:
        if isinstance(item, dict) and "Examples" in item:
            for child in item["Examples"]:
                if isinstance(child, dict) and len(child) == 1:
                    (_, target), = child.items()
                    parts = str(target).split("/")
                    if len(parts) == 3 and parts[0] == "examples" and parts[2] == "index.md":
                        tiers.append(parts[1])
    return tiers


def _inject_nav(
    config: Any, by_category: dict[str, list[str]], order_map: dict[str, float], title_map: dict[str, str]
) -> None:
    nav = config.get("nav") or []
    new_nav: list[Any] = []
    for item in nav:
        if isinstance(item, dict) and "Examples" in item:
            section = _build_examples_section(item["Examples"], by_category, order_map, title_map)
            new_nav.append({"Examples": section})
        else:
            new_nav.append(item)
    config["nav"] = new_nav


def on_config(config: Any) -> Any:
    r"""Discover examples, render virtual pages + tier galleries, rebuild the nav."""
    base = str(config.get("repo_url") or "").rstrip("/")
    _generated.clear()

    order_map: dict[str, float] = {}
    title_map: dict[str, str] = {}
    by_cat: dict[str, list[tuple[str, Example]]] = {}

    for example in _discover_examples():
        src = _doc_src_path(example)
        if (DOCS_DIR / src).exists():
            continue  # curated on-disk page wins
        _generated[src] = example.generate(base)
        order_map[src] = example.order
        title_map[src] = example.title
        category = src.split("/")[1]
        by_cat.setdefault(category, []).append((src, example))

    # Section overview (from examples/index.md) and tier index pages (from
    # examples/<tier>/index.md + the example table). A curated on-disk page,
    # should one ever be added, still wins.
    if not (DOCS_DIR / "examples/index.md").exists():
        _generated["examples/index.md"] = _render_overview(base)

    nav = config.get("nav") or []
    for category in dict.fromkeys(_examples_nav_tiers(nav) + list(by_cat)):
        idx = f"examples/{category}/index.md"
        if idx in _generated or (DOCS_DIR / idx).exists():
            continue
        _generated[idx] = _render_tier_index(category, by_cat.get(category, []), base)

    by_category_uris = {cat: [src for src, _ in items] for cat, items in by_cat.items()}
    _inject_nav(config, by_category_uris, order_map, title_map)
    logger.info(
        "gen_examples: prepared %d example pages across %d tiers",
        sum(len(v) for v in by_cat.values()), len(by_cat),
    )
    return config


@event_priority(100)  # inject before plugins that validate the file set (redirects)
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
