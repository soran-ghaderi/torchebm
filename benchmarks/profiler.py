#!/usr/bin/env python
"""TorchEBM profiling CLI.

Two sub-commands:

    run    Run torch.profiler on a component or arbitrary callable and
           emit chrome trace, top-N ops table, memory snapshot, and/or
           NVTX ranges.

    diff   Compare two ``top_ops.json`` files produced by ``run`` and
           render a markdown diff sorted by |Δ|.

Profiling never touches ``torchebm/*`` source. The target callable is
either (a) a benchmarked component resolved via ``benchmarks/registry.py``
(same code path exercised by pytest-benchmark) or (b) any importable
zero-arg callable via ``module:attr``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from typing import Callable, Dict, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from conftest import SCALES, make_pedantic_setup  # noqa: E402
from registry import TEMPLATE_BUILDERS, apply_mode, discover_components  # noqa: E402
from profile_utils import (  # noqa: E402
    diff_top_ops,
    dump_top_ops_json,
    extract_top_ops,
    load_top_ops_json,
    render_diff_md,
    render_top_ops_md,
    write_meta,
)


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


def _resolve_component(
    name: str, scale: str, device: torch.device, dtype: torch.dtype
) -> Tuple[Callable, Dict, str]:
    """Build the same (fn, info) the benchmarks time, for a named component."""
    specs = [s for s in discover_components() if s.name == name]
    if not specs:
        avail = sorted({s.name for s in discover_components()})
        raise SystemExit(
            f"Unknown component '{name}'.\nAvailable:\n  " + "\n  ".join(avail)
        )
    spec = specs[0]
    cfg = SCALES[scale]
    dim, bs, n_steps = cfg["dim"], cfg["batch_size"], cfg["n_steps"]

    builder = TEMPLATE_BUILDERS[spec.module]
    if spec.module == "models":
        fn, info = builder(spec, dim, bs, n_steps, device, dtype, scale=scale)
    else:
        fn, info = builder(spec, dim, bs, n_steps, device, dtype)

    # Registry returns private keys used by test harness — drop them.
    info.pop("_quality_fn", None)
    info.pop("_counting_drift", None)

    label = f"{spec.module}_{spec.name}_{scale}"
    return fn, info, label


def _resolve_callable(dotted: str) -> Tuple[Callable, Dict, str]:
    """Import ``module:attr`` and return a zero-arg callable."""
    if ":" not in dotted:
        raise SystemExit(f"--callable must be 'module:attr', got {dotted!r}")
    mod_path, attr = dotted.split(":", 1)
    mod = importlib.import_module(mod_path)
    target = getattr(mod, attr)
    if callable(target) and not _is_zero_arg(target):
        # Treat as factory: call once to produce the zero-arg step fn.
        target = target()
    if not callable(target):
        raise SystemExit(f"{dotted} is not callable")
    label = dotted.replace(":", "_").replace(".", "_")
    return target, {}, label


def _is_zero_arg(fn: Callable) -> bool:
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    required = [
        p
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    return len(required) == 0


# ---------------------------------------------------------------------------
# run subcommand
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[profile] CUDA requested but unavailable — falling back to CPU")
        device = torch.device("cpu")
    dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.randn(1, device=device)  # warm up context

    if args.component:
        fn, info, label = _resolve_component(args.component, args.scale, device, dtype)
    elif args.callable:
        fn, info, label = _resolve_callable(args.callable)
    else:
        raise SystemExit("provide --component NAME or --callable module:attr")

    mode = "eager"
    if args.compile:
        mode = "compiled"
    elif args.amp:
        mode = "amp_fp16"
    fn = apply_mode(fn, mode, device)

    # Output dir
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suffix = f"_{args.label}" if args.label else ""
    run_name = f"{label}_{device.type}_{mode}_{stamp}{suffix}"
    out_dir = os.path.join(args.out, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[profile] → {out_dir}")

    # NVTX wrapper (optional)
    use_nvtx = args.nvtx and device.type == "cuda"
    if args.nvtx and device.type != "cuda":
        print("[profile] --nvtx requires CUDA — skipping NVTX")

    orig_fn = fn
    if use_nvtx:

        def _nvtx_fn():
            torch.cuda.nvtx.range_push(label)
            try:
                orig_fn()
            finally:
                torch.cuda.nvtx.range_pop()

        fn = _nvtx_fn

    # Memory snapshot (optional)
    snapshot_requested = args.memory and device.type == "cuda"
    if args.memory and device.type != "cuda":
        print("[profile] --memory requires CUDA — skipping memory snapshot")

    if snapshot_requested:
        torch.cuda.memory._record_memory_history(max_entries=100_000)

    # Warm up
    setup = make_pedantic_setup(device)
    setup()
    for _ in range(args.warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # torch.profiler
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    want_trace = args.trace or args.all
    want_top = args.top is not None or args.all
    top_n = args.top if args.top is not None else 20

    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.steps):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # Artifacts
    if want_trace:
        trace_path = os.path.join(out_dir, "trace.json")
        prof.export_chrome_trace(trace_path)
        print(f"[profile]   trace.json          ({_size_mb(trace_path):.1f} MB)")

    if want_top:
        rows = extract_top_ops(prof, n=top_n)
        md_path = os.path.join(out_dir, "top_ops.md")
        json_path = os.path.join(out_dir, "top_ops.json")
        with open(md_path, "w") as f:
            f.write(
                render_top_ops_md(rows, title=f"Top {top_n} ops — {label} [{mode}]")
            )
        dump_top_ops_json(rows, json_path)
        print(f"[profile]   top_ops.md/.json    (top {top_n})")

    if snapshot_requested:
        snap_path = os.path.join(out_dir, "memory_snapshot.pickle")
        torch.cuda.memory._dump_snapshot(snap_path)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(
            f"[profile]   memory_snapshot.pickle  "
            f"({_size_mb(snap_path):.1f} MB) — view at https://pytorch.org/memory_viz"
        )

    if use_nvtx:
        print(
            "[profile]   NVTX ranges emitted. Example:\n"
            f"             nsys profile -o {out_dir}/nsys "
            f"--trace=cuda,nvtx python {sys.argv[0]} run "
            + " ".join(a for a in sys.argv[2:] if a != "--nvtx")
        )

    # Line profiler (optional)
    if args.line:
        _run_line_profiler(args.line, fn, out_dir)

    # Meta
    meta_extra = {
        "device": device.type,
        "dtype": args.dtype,
        "mode": mode,
        "component": args.component,
        "callable": args.callable,
        "scale": args.scale if args.component else None,
        "warmup": args.warmup,
        "steps": args.steps,
        "extra_info": {k: v for k, v in info.items() if _jsonable(v)},
    }
    write_meta(out_dir, sys.argv, args.label, extra=meta_extra)

    # Concise summary to stdout
    if want_top:
        print()
        print(render_top_ops_md(rows[: min(10, top_n)], title=f"Top 10 ops summary"))

    return 0


def _run_line_profiler(dotted: str, fn: Callable, out_dir: str) -> None:
    try:
        from line_profiler import LineProfiler
    except ImportError:
        print(
            "[profile] --line requested but `line_profiler` is not installed.\n"
            "          Install with: pip install line_profiler"
        )
        return

    if ":" not in dotted:
        print(f"[profile] --line expects 'module:attr', got {dotted!r}")
        return
    mod_path, attr = dotted.split(":", 1)
    try:
        mod = importlib.import_module(mod_path)
        target = getattr(mod, attr)
    except (ImportError, AttributeError) as e:
        print(f"[profile] --line could not resolve {dotted}: {e}")
        return

    lp = LineProfiler()
    lp.add_function(target)
    lp_wrapper = lp(fn)
    lp_wrapper()
    path = os.path.join(out_dir, "line_prof.txt")
    with open(path, "w") as f:
        lp.print_stats(stream=f)
    print(f"[profile]   line_prof.txt       (target: {dotted})")


def _jsonable(v) -> bool:
    import json

    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def _size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024**2)


# ---------------------------------------------------------------------------
# diff subcommand
# ---------------------------------------------------------------------------


def cmd_diff(args: argparse.Namespace) -> int:
    a_path = _locate_top_ops(args.run_a)
    b_path = _locate_top_ops(args.run_b)
    rows_a = load_top_ops_json(a_path)
    rows_b = load_top_ops_json(b_path)

    label_a = args.label_a or os.path.basename(os.path.dirname(a_path))
    label_b = args.label_b or os.path.basename(os.path.dirname(b_path))

    diff = diff_top_ops(rows_a, rows_b, n=args.top, metric=args.metric)
    md = render_diff_md(diff, label_a=label_a, label_b=label_b, metric=args.metric)

    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
        print(f"[profile] wrote {args.out}")
    print(md)
    return 0


def _locate_top_ops(path: str) -> str:
    """Accept either a run directory or a path to top_ops.json."""
    if os.path.isdir(path):
        return os.path.join(path, "top_ops.json")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="profile", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="profile a component or callable")
    tgt = run.add_mutually_exclusive_group(required=True)
    tgt.add_argument("--component", help="name of a registered component")
    tgt.add_argument("--callable", help="importable 'module:attr' zero-arg callable")
    run.add_argument("--scale", default="small", choices=list(SCALES.keys()))
    run.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    run.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64", "float16", "bfloat16"],
    )
    run.add_argument("--warmup", type=int, default=5)
    run.add_argument("--steps", type=int, default=20)
    run.add_argument("--compile", action="store_true", help="wrap with torch.compile")
    run.add_argument("--amp", action="store_true", help="run under autocast float16")
    run.add_argument("--trace", action="store_true", help="write chrome trace.json")
    run.add_argument("--top", type=int, default=None, help="write top-N ops table")
    run.add_argument(
        "--memory", action="store_true", help="record CUDA memory snapshot pickle"
    )
    run.add_argument("--nvtx", action="store_true", help="emit NVTX ranges per step")
    run.add_argument(
        "--line",
        default=None,
        help="run line_profiler on given 'module:attr' (optional dep)",
    )
    run.add_argument(
        "--all",
        action="store_true",
        help="enable --trace, --top 20, --memory, --nvtx",
    )
    run.add_argument("--label", default=None, help="free-form tag for output dir")
    run.add_argument(
        "--out",
        default=os.path.join(_HERE, "profiles"),
        help="output root directory",
    )
    run.set_defaults(func=cmd_run)

    diff = sub.add_parser("diff", help="diff two top_ops.json files or run dirs")
    diff.add_argument("run_a")
    diff.add_argument("run_b")
    diff.add_argument("--top", type=int, default=20)
    diff.add_argument(
        "--metric",
        default="self_cuda_time_total_us",
        choices=[
            "self_cuda_time_total_us",
            "cuda_time_total_us",
            "self_cpu_time_total_us",
            "cpu_time_total_us",
            "self_cuda_mem_b",
            "self_cpu_mem_b",
        ],
    )
    diff.add_argument("--label-a", default=None)
    diff.add_argument("--label-b", default=None)
    diff.add_argument("--out", default=None, help="write markdown here")
    diff.set_defaults(func=cmd_diff)

    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
