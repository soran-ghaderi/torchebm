"""
MkDocs hook: auto-generates benchmark pages entirely in memory at build time.

Creates virtual files for the benchmark section using ``File.generated()``,
reading JSON results from ``benchmarks/results/``. No physical markdown
files are needed under ``docs/benchmarks/``.

Results are grouped by device (GPU/CPU) — comparisons only happen within
the same device. Multi-version support lets users compare any two versions
via content tabs when three or more versions are available.

Registered in mkdocs.yml under ``hooks:``.
"""

from __future__ import annotations

import importlib.util
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from mkdocs.structure.files import File

try:
    from packaging.version import Version, InvalidVersion
except ImportError:
    Version = None

_RESULTS_DIR = Path("benchmarks/results")
_MAX_COMPARE_TABS = 4


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _clean_version(version: str) -> str:
    v = version.split("+")[0]
    return v.lstrip("v") if v.startswith("v") else v


def _sort_versions(versions: list[str]) -> list[str]:
    if Version is not None:
        def _key(v: str):
            try:
                return (0, Version(v))
            except InvalidVersion:
                return (1, v)
        return sorted(versions, key=_key, reverse=True)
    return sorted(versions, reverse=True)


def _scan_results() -> dict[str, dict[str, dict]]:
    r"""Scan all JSON result files grouped by device and version.

    Returns:
        ``{device: {version: data}}`` with versions sorted newest-first.
    """
    if not _RESULTS_DIR.is_dir():
        return {}

    entries: dict[tuple[str, str], list[tuple[dict, str]]] = {}

    for path in _RESULTS_DIR.glob("*.json"):
        data = _load_json(path)
        if not data or "results" not in data or "environment" not in data:
            continue

        env = data["environment"]
        results = data["results"]
        version_raw = env.get("torchebm_version", "")
        if not version_raw:
            continue

        version = _clean_version(version_raw)
        timestamp = env.get("timestamp", "")

        devices: dict[str, list[dict]] = {}
        for r in results:
            if "error" not in r:
                devices.setdefault(r.get("device", "cpu"), []).append(r)

        for device, device_results in devices.items():
            device_errors = [r for r in results if "error" in r]
            device_data = {
                "environment": env,
                "results": device_results + device_errors,
            }
            entries.setdefault((version, device), []).append(
                (device_data, timestamp)
            )

    deduped: dict[str, dict[str, dict]] = {}
    for (version, device), items in entries.items():
        best = max(items, key=lambda x: x[1])
        deduped.setdefault(device, {})[version] = best[0]

    result = {}
    for device, ver_map in deduped.items():
        sorted_vers = _sort_versions(list(ver_map.keys()))
        result[device] = {v: ver_map[v] for v in sorted_vers}

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Comparison helpers
# ═══════════════════════════════════════════════════════════════════════════


def _match_pairs(before: dict, after: dict) -> list[tuple[dict, dict]]:
    bm = {r["name"]: r for r in before["results"] if "error" not in r}
    return [
        (bm[r["name"]], r)
        for r in after["results"]
        if "error" not in r and r["name"] in bm
    ]


def _speedup(b: dict, a: dict) -> dict[str, Any]:
    b_med, a_med = b["median_ms"], a["median_ms"]
    sp = b_med / a_med if a_med > 0 else float("inf")
    return {
        "name": b["name"],
        "module": b["module"],
        "batch_size": b["batch_size"],
        "before_ms": b_med,
        "after_ms": a_med,
        "speedup": sp,
        "pct": (1 - a_med / b_med) * 100 if b_med > 0 else 0,
        "before_mem": b.get("peak_memory_mb", 0),
        "after_mem": a.get("peak_memory_mb", 0),
        "before_tp": b.get("samples_per_sec", 0),
        "after_tp": a.get("samples_per_sec", 0),
    }


def _geomean(values: list[float]) -> float:
    if not values:
        return 1.0
    prod = 1.0
    for v in values:
        prod *= v
    return prod ** (1.0 / len(values))


def _indicator(sp: float) -> str:
    if sp >= 1.5:
        return ":material-arrow-up-bold:{ .text-green } **significant**"
    if sp >= 1.1:
        return ":material-arrow-up:{ .text-green }"
    if sp >= 0.95:
        return ":material-minus:{ .text-grey }"
    if sp >= 0.8:
        return ":material-arrow-down:{ .text-orange }"
    return ":material-arrow-down-bold:{ .text-red } **regression**"


def _bar(value: float, max_value: float, width: int = 20) -> str:
    if max_value <= 0:
        return ""
    return "█" * max(1, int((value / max_value) * width))


def _device_label(device: str) -> str:
    if device == "cuda":
        return ":material-expansion-card: GPU (CUDA)"
    return ":material-cpu-64-bit: CPU"


def _fmt_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return ts


# ═══════════════════════════════════════════════════════════════════════════
# Section renderers
# ═══════════════════════════════════════════════════════════════════════════


def _render_environment(data: dict, version: str) -> list[str]:
    env = data.get("environment", {})
    ts = _fmt_ts(env.get("timestamp", "unknown"))
    lines = [
        f'!!! abstract "Environment — {version}"',
        "",
        "    | | |",
        "    |---|---|",
        f'    | **PyTorch** | `{env.get("torch_version", "N/A")}` |',
        f'    | **TorchEBM** | `{version}` |',
    ]
    gpu = env.get("gpu_name")
    if gpu:
        mem = env.get("gpu_memory_gb")
        mem_str = f" ({mem:.1f} GB)" if mem else ""
        lines.append(f"    | **GPU** | `{gpu}`{mem_str} |")
        lines.append(
            f'    | **CUDA** | `{env.get("cuda_version", "N/A")}` |'
        )
    lines += [
        f'    | **Platform** | `{env.get("platform", "N/A")}` |',
        f"    | **Date** | {ts} |",
        "",
    ]
    return lines


def _render_results_table(
    data: dict, version: str, *, include_heading: bool = True, html_subheadings: bool = False
) -> list[str]:
    results = [r for r in data["results"] if "error" not in r]
    errors = [r for r in data["results"] if "error" in r]

    modules: dict[str, list[dict]] = {}
    for r in results:
        modules.setdefault(r["module"], []).append(r)

    lines: list[str] = []
    if include_heading:
        lines += [f"## Results — {version}", ""]
    lines += [f"**{len(results)}** benchmarks across **{len(modules)}** modules",
        "",
    ]

    def _subheading(text: str, level: int = 3) -> str:
        if html_subheadings:
            tag = f"h{level}"
            return f"<{tag}>{text}</{tag}>"
        return f"{'#' * level} {text}"

    for mod in sorted(modules):
        items = sorted(modules[mod], key=lambda r: r["name"])
        lines += [
            _subheading(mod.title()),
            "",
            "| Benchmark | Batch | Median (ms) | Throughput (samp/s) | Peak Mem (MB) |",
            "|-----------|------:|------------:|--------------------:|--------------:|",
        ]
        for r in items:
            mem = (
                f'{r.get("peak_memory_mb", 0):.1f}'
                if r.get("peak_memory_mb", 0) > 0
                else "—"
            )
            lines.append(
                f'| `{r["name"]}` '
                f'| {r["batch_size"]} '
                f'| {r["median_ms"]:.2f} '
                f'| {r.get("samples_per_sec", 0):.0f} '
                f"| {mem} |"
            )
        lines.append("")

    if errors:
        lines += [
            _subheading("Errors"),
            "",
            "| Benchmark | Error |",
            "|-----------|-------|",
        ]
        for r in errors:
            lines.append(
                f'| `{r.get("name", "?")}` | {r.get("error", "?")} |'
            )
        lines.append("")

    return lines


def _render_comparison(
    before: dict,
    after: dict,
    before_ver: str,
    after_ver: str,
    *,
    include_heading: bool = True,
) -> list[str]:
    pairs = _match_pairs(before, after)
    if not pairs:
        return [
            f'!!! warning "No matching benchmarks between {before_ver} and {after_ver}"',
            "",
        ]

    comparisons = [_speedup(b, a) for b, a in pairs]
    speedups = [c["speedup"] for c in comparisons]
    geo = _geomean(speedups)
    total_before = sum(c["before_ms"] for c in comparisons)
    total_after = sum(c["after_ms"] for c in comparisons)
    net_saved = total_before - total_after
    pct_saved = (1 - total_after / total_before) * 100 if total_before > 0 else 0
    summary_type = "success" if geo >= 1.0 else "warning"

    lines: list[str] = []

    if include_heading:
        lines += [
            f"## Comparison: {after_ver} vs {before_ver}",
            "",
        ]

    # Overall summary
    lines += [
        f'!!! {summary_type} "Overall: {geo:.2f}x geometric mean speedup"',
        "",
        f"    - **Benchmarks compared**: {len(comparisons)}",
        f"    - **Total baseline time**: {total_before:.1f} ms",
        f"    - **Total optimized time**: {total_after:.1f} ms",
        f"    - **Net time saved**: {net_saved:.1f} ms ({pct_saved:+.1f}%)",
        "",
    ]

    # Group by module
    modules: dict[str, list[dict]] = {}
    for c in comparisons:
        modules.setdefault(c["module"], []).append(c)

    # Speedup-by-module chart (vertical Mermaid — few categories, readable)
    if len(modules) > 1:
        max_geo = max(
            _geomean([c["speedup"] for c in items])
            for items in modules.values()
        )
        chart_max = max(3, math.ceil(max_geo + 0.5))
        x_labels = ", ".join(
            f'"{m.title()}"' for m in sorted(modules)
        )
        bars = ", ".join(
            f"{_geomean([c['speedup'] for c in modules[m]]):.2f}"
            for m in sorted(modules)
        )
        baseline_line = ", ".join("1" for _ in modules)
        lines += [
            "### Speedup by Module",
            "",
            "```mermaid",
            "%%{init: {'theme': 'dark'}}%%",
            "xychart-beta",
            '    title "Geometric Mean Speedup by Module"',
            f"    x-axis [{x_labels}]",
            f'    y-axis "Speedup (x)" 0 --> {chart_max}',
            f"    bar [{bars}]",
            f"    line [{baseline_line}]",
            "```",
            "",
        ]

    # Per-module detail tables
    for mod in sorted(modules):
        items = sorted(modules[mod], key=lambda c: c["name"])
        mod_geo = _geomean([c["speedup"] for c in items])
        icon = ":material-trending-up:" if mod_geo >= 1.0 else ":material-trending-down:"
        lines += [
            f"### {mod.title()} {icon} {mod_geo:.2f}x",
            "",
            f"| Benchmark | Batch | {before_ver} (ms) | {after_ver} (ms) | Speedup | Throughput | Status |",
            "|-----------|------:|------------------:|-----------------:|--------:|-----------:|--------|",
        ]
        for c in items:
            ind = _indicator(c["speedup"])
            tp = f'{c["before_tp"]:.0f} &rarr; {c["after_tp"]:.0f}'
            lines.append(
                f'| `{c["name"]}` '
                f'| {c["batch_size"]} '
                f'| {c["before_ms"]:.2f} '
                f'| {c["after_ms"]:.2f} '
                f'| **{c["speedup"]:.2f}x** '
                f"| {tp} "
                f"| {ind} |"
            )
        lines.append("")

    # Top improvements — horizontal bar table (names readable on left)
    top = sorted(comparisons, key=lambda c: c["speedup"], reverse=True)[:10]
    if top:
        max_sp = max(c["speedup"] for c in top)
        lines += [
            "### Top Improvements",
            "",
            "| Benchmark | Speedup | |",
            "|-----------|--------:|---|",
        ]
        for c in top:
            lines.append(
                f'| `{c["name"]}` '
                f'| **{c["speedup"]:.2f}x** '
                f"| {_bar(c['speedup'], max_sp)} |"
            )
        lines.append("")

    # Regressions
    regressions = [c for c in comparisons if c["speedup"] < 0.95]
    if regressions:
        lines += ['!!! warning "Regressions Detected"', ""]
        for c in regressions:
            lines.append(
                f'    - **`{c["name"]}`**: {c["speedup"]:.2f}x '
                f'({c["before_ms"]:.2f} ms &rarr; {c["after_ms"]:.2f} ms)'
            )
        lines.append("")

    # Memory comparison
    mem = [c for c in comparisons if c["before_mem"] > 0 and c["after_mem"] > 0]
    if mem:
        lines += [
            "### Memory Usage",
            "",
            "| Benchmark | Before (MB) | After (MB) | Ratio |",
            "|-----------|------------:|-----------:|------:|",
        ]
        for c in mem:
            ratio = c["before_mem"] / c["after_mem"] if c["after_mem"] > 0 else 0
            lines.append(
                f'| `{c["name"]}` '
                f'| {c["before_mem"]:.1f} '
                f'| {c["after_mem"]:.1f} '
                f"| {ratio:.2f}x |"
            )
        lines.append("")

    return lines


def _render_history_table(versions_data: dict[str, dict]) -> list[str]:
    if len(versions_data) < 2:
        return []

    lines = [
        "## Version History",
        "",
        "| Version | Date | Benchmarks | Avg Median (ms) | Avg Throughput (samp/s) |",
        "|---------|------|----------:|----------------:|------------------------:|",
    ]
    for ver, data in versions_data.items():
        env = data.get("environment", {})
        ts = _fmt_ts(env.get("timestamp", ""))
        results = [r for r in data["results"] if "error" not in r]
        if results:
            avg_med = sum(r["median_ms"] for r in results) / len(results)
            avg_tp = sum(r.get("samples_per_sec", 0) for r in results) / len(
                results
            )
        else:
            avg_med = avg_tp = 0
        lines.append(
            f"| **{ver}** | {ts} | {len(results)} | {avg_med:.2f} | {avg_tp:.0f} |"
        )
    lines.append("")
    return lines


# ═══════════════════════════════════════════════════════════════════════════
# Device section
# ═══════════════════════════════════════════════════════════════════════════


def _render_device_section(
    device: str, versions_data: dict[str, dict], *,
    include_heading: bool = True, html_subheadings: bool = False
) -> list[str]:
    versions = list(versions_data.keys())
    latest_ver = versions[0]
    latest_data = versions_data[latest_ver]

    lines: list[str] = []

    # Environment + latest results
    lines += _render_environment(latest_data, latest_ver)
    lines += _render_results_table(
        latest_data, latest_ver,
        include_heading=include_heading,
        html_subheadings=html_subheadings,
    )

    # Comparison (only when ≥2 versions exist for this device)
    if len(versions) >= 2:
        prev_versions = versions[1:]

        if len(prev_versions) == 1:
            # Single previous version — render comparison directly
            lines += _render_comparison(
                versions_data[prev_versions[0]],
                latest_data,
                prev_versions[0],
                latest_ver,
            )
        else:
            # Multiple previous versions — tabs to select comparison target
            lines += ["## Performance Comparison", ""]
            for i, prev_ver in enumerate(
                prev_versions[:_MAX_COMPARE_TABS]
            ):
                label = prev_ver
                if i == 0:
                    label += " (Previous)"
                lines.append(f'=== "{label}"')
                lines.append("")
                comp_lines = _render_comparison(
                    versions_data[prev_ver],
                    latest_data,
                    prev_ver,
                    latest_ver,
                    include_heading=False,
                )
                lines += [f"    {ln}" for ln in comp_lines]
                lines.append("")

        lines += _render_history_table(versions_data)

    return lines


# ═══════════════════════════════════════════════════════════════════════════
# Full page generators
# ═══════════════════════════════════════════════════════════════════════════


def _render_no_results() -> str:
    return "\n".join(
        [
            "<!-- GENERATED AT BUILD TIME — do not edit -->",
            "",
            "# Benchmark Results",
            "",
            '!!! info "No benchmark results found"',
            "",
            "    Run the benchmark suite to populate this page:",
            "",
            "    ```bash",
            "    bash benchmarks/run.sh",
            "    ```",
            "",
            "    Results will appear here automatically on the next `mkdocs serve`.",
        ]
    )


def _render_results_page(device_data: dict[str, dict[str, dict]]) -> str:
    if not device_data:
        return _render_no_results()

    lines = [
        "<!-- GENERATED AT BUILD TIME — do not edit -->",
        "",
        "# Benchmark Results",
        "",
    ]

    # GPU first, then CPU
    devices = sorted(device_data.keys(), key=lambda d: (d != "cuda", d))

    if len(devices) == 1:
        lines += _render_device_section(devices[0], device_data[devices[0]])
    else:
        # Shared heading — versions may differ per device, collect unique ones
        all_versions = sorted(
            {v for dd in device_data.values() for v in dd}, reverse=True
        )
        lines += [
            f"## Results — {', '.join(all_versions)}",
            "",
        ]
        for i, device in enumerate(devices):
            lines.append(f'=== "{_device_label(device)}"')
            lines.append("")
            section = _render_device_section(
                device, device_data[device],
                include_heading=False, html_subheadings=(i > 0)
            )
            lines += [f"    {ln}" for ln in section]
            lines.append("")

    return "\n".join(lines)


def _render_index(device_data: dict[str, dict[str, dict]]) -> str:
    has_results = bool(device_data)
    lines = [
        "---",
        "title: Benchmarks",
        "icon: material/speedometer",
        "---",
        "",
        "# Benchmarks",
        "",
        "Automated performance benchmarks for every TorchEBM module.",
        "Results are generated by the benchmark suite and rendered here",
        "automatically at documentation build time.",
        "",
        "## Quick Start",
        "",
        '=== "First Run (Baseline)"',
        "",
        "    ```bash",
        "    bash benchmarks/run.sh",
        "    ```",
        "",
        "    Captures baseline performance numbers.",
        "",
        '=== "After Optimization"',
        "",
        "    ```bash",
        "    bash benchmarks/run.sh",
        "    ```",
        "",
        "    Automatically compares against baseline and generates report.",
        "",
        '=== "Quick Test"',
        "",
        "    ```bash",
        "    bash benchmarks/run.sh --quick",
        "    ```",
        "",
        "    Small-scale only (~1 min).",
        "",
        '=== "CI Mode"',
        "",
        "    ```bash",
        "    bash benchmarks/run.sh --ci",
        "    ```",
        "",
        "    Exits non-zero on regression > 5%.",
        "",
        "## Coverage",
        "",
        '<div class="grid cards" markdown>',
        "",
        '-   :material-calculator-variant:{ .lg .middle } **Losses**',
        "",
        "    Exact SM, Approx SM, Sliced SM, Contrastive Divergence, Equilibrium Matching",
        "",
        '-   :material-chart-scatter-plot:{ .lg .middle } **Samplers**',
        "",
        "    Langevin Dynamics, HMC, Flow ODE",
        "",
        '-   :material-math-integral:{ .lg .middle } **Integrators**',
        "",
        "    Leapfrog, Euler-Maruyama, RK4, DOPRI5 (adaptive)",
        "",
        '-   :material-brain:{ .lg .middle } **Models**',
        "",
        "    Transformer forward, Transformer forward+backward",
        "",
        '-   :material-sine-wave:{ .lg .middle } **Interpolants**',
        "",
        "    Linear, Cosine, Variance-Preserving",
        "",
        "</div>",
        "",
        "## Scale Configurations",
        "",
        "| Scale | Batch Size | Dimensions | Steps |",
        "|-------|----------:|----------:|------:|",
        "| `small` | 64 | 8 | 50 |",
        "| `medium` | 256 | 32 | 100 |",
        "| `large` | 1024 | 128 | 200 |",
        "",
        "## Measurement Methodology",
        "",
        "| Aspect | GPU | CPU |",
        "|--------|-----|-----|",
        "| **Timer** | `torch.cuda.Event` | `time.perf_counter` |",
        "| **Warm-up** | 10 iterations | 3 iterations |",
        "| **Measured** | 50 iterations | 20 iterations |",
        "| **Memory** | `cuda.max_memory_allocated` | N/A |",
        "| **Statistics** | median, mean, std, IQR, min, max | same |",
        "| **Comparison** | geometric mean speedup | same |",
        "",
    ]

    if has_results:
        devices = sorted(device_data.keys(), key=lambda d: (d != "cuda", d))
        device_str = ", ".join(d.upper() for d in devices)
        total_versions = sum(len(v) for v in device_data.values())
        lines += [
            "## Results",
            "",
            f"**{total_versions}** version(s) across **{device_str}**",
            "",
            "[View benchmark results :material-arrow-right:](results.md){ .md-button }",
            "",
        ]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Interactive dashboard page (Plotly-based)
# ═══════════════════════════════════════════════════════════════════════════


def _import_bench_data():
    path = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "data.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_bench_data", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_DASHBOARD_CSS = """
#bench-app * { box-sizing: border-box; }
.bench-sum-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; margin-bottom: 20px; }
.bench-sum-card { background: var(--md-code-bg-color, #161616); border: 1px solid var(--md-default-fg-color--lightest, #30363d); border-radius: 8px; padding: 16px; text-align: center; }
.bench-sum-label { font-size: 0.75em; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.5px; }
.bench-sum-value { font-size: 1.6em; font-weight: 700; margin: 4px 0; }
.bench-sum-detail { font-size: 0.8em; opacity: 0.6; }
.bench-nav { display: flex; gap: 4px; border-bottom: 1px solid var(--md-default-fg-color--lightest, #30363d); margin-bottom: 16px; overflow-x: auto; }
.bench-nav-tab { padding: 8px 16px; border: none; background: none; color: var(--md-default-fg-color--light, #8b949e); cursor: pointer; font-size: 0.9em; border-bottom: 2px solid transparent; white-space: nowrap; transition: color 0.2s, border-color 0.2s; }
.bench-nav-tab:hover { color: var(--md-default-fg-color, #e6edf3); }
.bench-nav-tab.bench-active { color: var(--md-accent-fg-color, #a8cc3b); border-bottom-color: var(--md-accent-fg-color, #a8cc3b); }
.bench-tab-content { display: none; }
.bench-tab-content.bench-active { display: block; }
.bench-controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 16px; font-size: 0.9em; }
.bench-controls select, .bench-controls input { background: var(--md-code-bg-color, #161616); color: var(--md-default-fg-color, #e6edf3); border: 1px solid var(--md-default-fg-color--lightest, #30363d); border-radius: 6px; padding: 6px 10px; font-size: 0.9em; }
.bench-controls label { opacity: 0.7; font-size: 0.85em; }
.bench-card { background: var(--md-code-bg-color, #161616); border: 1px solid var(--md-default-fg-color--lightest, #30363d); border-radius: 8px; padding: 16px; margin-bottom: 12px; }
.bench-card h3 { margin: 0 0 12px 0; font-size: 1em; opacity: 0.8; }
.bench-chart { width: 100%; min-height: 420px; }
.bench-grid-2 { display: grid; grid-template-columns: 1fr; gap: 12px; margin-bottom: 12px; }
.bench-mod-section { margin-bottom: 24px; }
.bench-mod-hdr { font-size: 1.3em; font-weight: 700; text-transform: capitalize; margin-bottom: 12px; padding-bottom: 6px; border-bottom: 1px solid var(--md-default-fg-color--lightest, #30363d); }
.bench-tbl-scroll { overflow-x: auto; }
.bench-tbl { width: 100%; border-collapse: collapse; font-size: 0.85em; }
.bench-tbl th, .bench-tbl td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--md-default-fg-color--lightest, #30363d); }
.bench-tbl th { font-weight: 600; opacity: 0.7; font-size: 0.85em; text-transform: uppercase; }
.bench-tbl td:nth-child(n+3) { text-align: right; font-family: monospace; }
.bench-tag { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; background: var(--md-default-fg-color--lightest, #30363d); }
.bench-pill { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.85em; font-weight: 600; }
.bench-pill-green { color: #3fb950; background: rgba(63,185,80,0.15); }
.bench-pill-red { color: #f85149; background: rgba(248,81,73,0.15); }
.bench-pill-yellow { color: #d29922; background: rgba(210,153,34,0.15); }
.bench-pill-neutral { opacity: 0.5; }
.bench-delta-bar { display: inline-block; height: 8px; border-radius: 4px; vertical-align: middle; margin-right: 6px; }
.bench-delta-positive { background: #3fb950; }
.bench-delta-negative { background: #f85149; }
.bench-mcs { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
.bench-mc { background: var(--md-code-bg-color, #161616); border: 1px solid var(--md-default-fg-color--lightest, #30363d); border-radius: 6px; padding: 8px 14px; }
.bench-mc-l { font-size: 0.75em; opacity: 0.6; display: block; }
.bench-mc-v { font-size: 1.1em; font-weight: 600; }
.bench-ok { color: #3fb950 !important; }
.bench-warn { color: #d29922 !important; }
"""


def _render_no_dashboard() -> str:
    return "\n".join([
        "---",
        "title: Interactive Dashboard",
        "icon: material/chart-box",
        "---",
        "",
        "# Interactive Dashboard",
        "",
        '!!! info "No interactive benchmark data available"',
        "",
        "    Run benchmarks with pytest-benchmark to populate this page:",
        "",
        "    ```bash",
        "    bash benchmarks/run.sh",
        "    ```",
    ])


def _render_dashboard_page() -> str:
    bench_data = _import_bench_data()
    if bench_data is None:
        return _render_no_dashboard()

    runs = bench_data.load_benchmark_files(str(_RESULTS_DIR))
    if not runs:
        return _render_no_dashboard()

    benchmarks, run_meta = bench_data.extract_all_data(runs)
    bench_data.backfill_gpu_info(benchmarks, run_meta)

    for b in benchmarks:
        b.pop("_extra_raw", None)

    modules = sorted(set(b["module"] for b in benchmarks))
    scales = sorted(
        set(b["scale"] for b in benchmarks),
        key=lambda s: {"small": 0, "medium": 1, "large": 2}.get(s, 3),
    )

    data_json = json.dumps({
        "benchmarks": benchmarks,
        "run_meta": run_meta,
        "modules": modules,
        "scales": scales,
    }).replace("</", "<\\/")

    last_idx = len(run_meta) - 1
    before_idx = max(0, last_idx - 1)

    def _opts(selected):
        parts = []
        for rm in run_meta:
            sel = " selected" if rm["index"] == selected else ""
            parts.append(f'<option value="{rm["index"]}"{sel}>{rm["label"]}</option>')
        return "".join(parts)

    def _mod_opts(include_all=False):
        h = '<option value="all">All Modules</option>' if include_all else ""
        return h + "".join(f'<option value="{m}">{m}</option>' for m in modules)

    def _scale_opts():
        return '<option value="all">All Scales</option>' + "".join(
            f'<option value="{s}">{s}</option>' for s in scales
        )

    latest = [b for b in benchmarks if b["run_index"] == last_idx]
    total = len(latest)
    n_mods = len(set(b["module"] for b in latest))
    rm_last = run_meta[last_idx] if run_meta else {}
    device = rm_last.get("gpu_name") or rm_last.get("device") or "N/A"

    return "\n".join([
        "---",
        "title: Interactive Dashboard",
        "icon: material/chart-box",
        "hide:",
        "  - toc",
        "---",
        "",
        "# Interactive Dashboard",
        "",
        "<style>",
        _DASHBOARD_CSS,
        "</style>",
        "",
        '<div id="bench-app">',
        "",
        '<div class="bench-sum-grid">',
        f'<div class="bench-sum-card"><div class="bench-sum-label">Benchmarks</div>'
        f'<div class="bench-sum-value">{total}</div></div>',
        f'<div class="bench-sum-card"><div class="bench-sum-label">Modules</div>'
        f'<div class="bench-sum-value">{n_mods}</div></div>',
        f'<div class="bench-sum-card"><div class="bench-sum-label">Runs</div>'
        f'<div class="bench-sum-value">{len(run_meta)}</div></div>',
        f'<div class="bench-sum-card"><div class="bench-sum-label">Device</div>'
        f'<div class="bench-sum-value" style="font-size:0.9em">{device}</div></div>',
        "</div>",
        "",
        '<div class="bench-nav">',
        "<button class=\"bench-nav-tab bench-active\" data-tab=\"overview\" onclick=\"benchSwitchTab('overview')\">Overview</button>",
        "<button class=\"bench-nav-tab\" data-tab=\"comparison\" onclick=\"benchSwitchTab('comparison')\">Run Comparison</button>",
        "<button class=\"bench-nav-tab\" data-tab=\"scaling\" onclick=\"benchSwitchTab('scaling')\">Scale Analysis</button>",
        "<button class=\"bench-nav-tab\" data-tab=\"history\" onclick=\"benchSwitchTab('history')\">History</button>",
        "<button class=\"bench-nav-tab\" data-tab=\"details\" onclick=\"benchSwitchTab('details')\">All Stats</button>",
        "</div>",
        "",
        '<div id="bench-tab-overview" class="bench-tab-content bench-active">',
        f'<div class="bench-controls"><label>Run:</label>'
        f'<select id="bench-ov-run" onchange="benchRenderOverview()">{_opts(last_idx)}</select></div>',
        '<div id="bench-overview-container"></div>',
        "</div>",
        "",
        '<div id="bench-tab-comparison" class="bench-tab-content">',
        f'<div class="bench-controls"><label>Before:</label>'
        f'<select id="bench-cmp-before" onchange="benchRenderComparison()">{_opts(before_idx)}</select>'
        f'<label>After:</label>'
        f'<select id="bench-cmp-after" onchange="benchRenderComparison()">{_opts(last_idx)}</select></div>',
        '<div id="bench-cmp-summary" class="bench-sum-grid"></div>',
        '<div id="bench-cmp-modules"></div>',
        "</div>",
        "",
        '<div id="bench-tab-scaling" class="bench-tab-content">',
        f'<div class="bench-controls"><label>Module:</label>'
        f'<select id="bench-sc-module" onchange="benchRenderScaling()">{_mod_opts()}</select>'
        f'<label>Run:</label>'
        f'<select id="bench-sc-run" onchange="benchRenderScaling()">{_opts(last_idx)}</select></div>',
        '<div class="bench-grid-2">',
        '<div class="bench-card"><h3>Absolute Scaling</h3><div id="bench-sc-abs" class="bench-chart"></div></div>',
        '<div class="bench-card"><h3>Relative Scaling (normalized to small)</h3><div id="bench-sc-rel" class="bench-chart"></div></div>',
        "</div>",
        '<div class="bench-card"><h3>Scale Factors</h3><div class="bench-tbl-scroll"><table id="bench-sc-table" class="bench-tbl">',
        "<thead><tr><th>Benchmark</th><th>Small</th><th>Medium</th><th>Large</th><th>Med/Small</th><th>Large/Small</th></tr></thead>",
        "<tbody></tbody></table></div></div>",
        "</div>",
        "",
        '<div id="bench-tab-history" class="bench-tab-content">',
        f'<div class="bench-controls"><label>Module:</label>'
        f'<select id="bench-hist-module" onchange="benchRenderHistory()">{_mod_opts()}</select>'
        f'<label>Scale:</label>'
        f'<select id="bench-hist-scale" onchange="benchRenderHistory()">{_scale_opts()}</select></div>',
        '<h3 id="bench-hist-title"></h3>',
        '<div id="bench-hist-chart" class="bench-chart" style="min-height:500px"></div>',
        "</div>",
        "",
        '<div id="bench-tab-details" class="bench-tab-content">',
        f'<div class="bench-controls"><label>Run:</label>'
        f'<select id="bench-dt-run" onchange="benchRenderDetails()">{_opts(last_idx)}</select>'
        f'<label>Module:</label>'
        f'<select id="bench-dt-module" onchange="benchRenderDetails()">{_mod_opts(include_all=True)}</select>'
        '<input id="bench-dt-search" type="text" placeholder="Search benchmarks..." oninput="benchRenderDetails()">'
        "</div>",
        '<div id="bench-details-container"></div>',
        "</div>",
        "",
        "</div>",
        "",
        "<script>",
        f"window.__BENCH_DATA__ = {data_json};",
        "</script>",
    ])


# ═══════════════════════════════════════════════════════════════════════════
# MkDocs hook entry point
# ═══════════════════════════════════════════════════════════════════════════


def on_files(files, config, **kwargs):
    r"""Inject virtual benchmark pages into the file collection."""
    device_data = _scan_results()

    pages = {
        "benchmarks/index.md": _render_index(device_data),
        "benchmarks/results.md": _render_results_page(device_data),
        "benchmarks/dashboard.md": _render_dashboard_page(),
    }

    for src_uri, content in pages.items():
        files.append(File.generated(config, src_uri, content=content))

    return files
