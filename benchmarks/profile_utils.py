"""Shared helpers for `benchmarks/profile.py`.

Kept framework-agnostic so the CLI stays tiny: extraction of top-N ops
from a `torch.profiler.profile` instance, serialization, and diff
rendering.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Top-N ops extraction
# ---------------------------------------------------------------------------


_TOP_OPS_COLUMNS = [
    "name",
    "count",
    "cpu_time_total_us",
    "self_cpu_time_total_us",
    "cuda_time_total_us",
    "self_cuda_time_total_us",
    "self_cpu_mem_b",
    "self_cuda_mem_b",
]


def extract_top_ops(prof, n: int = 20, sort_by: str = "auto") -> List[Dict[str, Any]]:
    """Return the top-N ops as plain dicts.

    Args:
        prof: A finished `torch.profiler.profile` instance.
        n: Number of rows to keep.
        sort_by: "auto" picks self_cuda_time if any row has CUDA time, else
            self_cpu_time. Or pass one of the keys in `_TOP_OPS_COLUMNS`.
    """
    events = prof.key_averages()
    rows: List[Dict[str, Any]] = []
    for ev in events:
        rows.append(
            {
                "name": ev.key,
                "count": int(getattr(ev, "count", 0)),
                "cpu_time_total_us": float(getattr(ev, "cpu_time_total", 0.0)),
                "self_cpu_time_total_us": float(
                    getattr(ev, "self_cpu_time_total", 0.0)
                ),
                "cuda_time_total_us": float(
                    getattr(
                        ev, "cuda_time_total", getattr(ev, "device_time_total", 0.0)
                    )
                ),
                "self_cuda_time_total_us": float(
                    getattr(
                        ev,
                        "self_cuda_time_total",
                        getattr(ev, "self_device_time_total", 0.0),
                    )
                ),
                "self_cpu_mem_b": int(getattr(ev, "self_cpu_memory_usage", 0) or 0),
                "self_cuda_mem_b": int(
                    getattr(
                        ev,
                        "self_cuda_memory_usage",
                        getattr(ev, "self_device_memory_usage", 0),
                    )
                    or 0
                ),
            }
        )

    if sort_by == "auto":
        has_cuda = any(r["self_cuda_time_total_us"] > 0 for r in rows)
        sort_by = "self_cuda_time_total_us" if has_cuda else "self_cpu_time_total_us"

    rows.sort(key=lambda r: r.get(sort_by, 0.0), reverse=True)
    return rows[:n]


def render_top_ops_md(rows: List[Dict[str, Any]], title: str = "Top ops") -> str:
    """Render a markdown table of top-N rows."""
    headers = [
        "op",
        "calls",
        "self CUDA (ms)",
        "CUDA (ms)",
        "self CPU (ms)",
        "CPU (ms)",
        "self CUDA mem (MB)",
        "self CPU mem (MB)",
    ]
    lines = [f"## {title}", "", "| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _truncate(r["name"], 60),
                    str(r["count"]),
                    f"{r['self_cuda_time_total_us'] / 1e3:.3f}",
                    f"{r['cuda_time_total_us'] / 1e3:.3f}",
                    f"{r['self_cpu_time_total_us'] / 1e3:.3f}",
                    f"{r['cpu_time_total_us'] / 1e3:.3f}",
                    f"{r['self_cuda_mem_b'] / (1024 ** 2):.3f}",
                    f"{r['self_cpu_mem_b'] / (1024 ** 2):.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def dump_top_ops_json(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w") as f:
        json.dump({"columns": _TOP_OPS_COLUMNS, "rows": rows}, f, indent=2)


def load_top_ops_json(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    return data["rows"]


# ---------------------------------------------------------------------------
# Diff between two runs
# ---------------------------------------------------------------------------


def diff_top_ops(
    rows_a: List[Dict[str, Any]],
    rows_b: List[Dict[str, Any]],
    n: int = 20,
    metric: str = "self_cuda_time_total_us",
) -> List[Dict[str, Any]]:
    """Join two top-N rowsets by op name and sort by |Δ|."""
    by_name_a = {r["name"]: r for r in rows_a}
    by_name_b = {r["name"]: r for r in rows_b}
    all_names = set(by_name_a) | set(by_name_b)

    out: List[Dict[str, Any]] = []
    for name in all_names:
        a = by_name_a.get(name)
        b = by_name_b.get(name)
        va = float(a[metric]) if a else 0.0
        vb = float(b[metric]) if b else 0.0
        delta = vb - va
        pct = (delta / va * 100.0) if va > 0 else (float("inf") if vb > 0 else 0.0)
        out.append(
            {
                "name": name,
                "a_us": va,
                "b_us": vb,
                "delta_us": delta,
                "delta_pct": pct,
                "calls_a": (a["count"] if a else 0),
                "calls_b": (b["count"] if b else 0),
                "status": (
                    "new" if a is None else "dropped" if b is None else "changed"
                ),
            }
        )
    out.sort(key=lambda r: abs(r["delta_us"]), reverse=True)
    return out[:n]


def render_diff_md(
    diff: List[Dict[str, Any]],
    label_a: str = "A",
    label_b: str = "B",
    metric: str = "self_cuda_time_total_us",
) -> str:
    headers = [
        "op",
        f"{label_a} (ms)",
        f"{label_b} (ms)",
        "Δ (ms)",
        "Δ %",
        f"calls {label_a}",
        f"calls {label_b}",
        "status",
    ]
    lines = [
        f"## Diff ({metric})",
        "",
        f"A = `{label_a}`  •  B = `{label_b}`",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in diff:
        pct = r["delta_pct"]
        pct_s = "∞" if pct == float("inf") else f"{pct:+.1f}%"
        lines.append(
            "| "
            + " | ".join(
                [
                    _truncate(r["name"], 60),
                    f"{r['a_us'] / 1e3:.3f}",
                    f"{r['b_us'] / 1e3:.3f}",
                    f"{r['delta_us'] / 1e3:+.3f}",
                    pct_s,
                    str(r["calls_a"]),
                    str(r["calls_b"]),
                    r["status"],
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def write_meta(
    out_dir: str,
    argv: List[str],
    label: Optional[str],
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    import torch

    try:
        import torchebm

        tebm_ver = getattr(torchebm, "__version__", "")
    except Exception:
        tebm_ver = ""

    meta: Dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch_version": torch.__version__,
        "torchebm_version": tebm_ver,
        "cuda_available": torch.cuda.is_available(),
        "python_version": sys.version.split()[0],
        "git_sha": _git_sha(),
        "argv": argv,
        "label": label or "",
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        meta["gpu_name"] = props.name
        meta["gpu_arch"] = f"sm_{props.major}{props.minor}"
        meta["cuda_version"] = torch.version.cuda or ""
    if extra:
        meta.update(extra)

    path = os.path.join(out_dir, "meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path
