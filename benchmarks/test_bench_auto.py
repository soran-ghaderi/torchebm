"""
Auto-generated benchmarks from the TorchEBM component registry.

pytest collects one test per (component × scale × mode) combination.  New
components exported in ``torchebm.*.__init__`` are picked up
automatically; only components with unusual construction requirements
need an entry in ``registry.COMPONENT_OVERRIDES``.
"""

import pytest
import torch

from conftest import (
    SCALES,
    get_bench_params,
    is_benchmark_excluded,
    is_module_enabled,
    make_pedantic_setup,
)
from registry import TEMPLATE_BUILDERS, BenchSpec, apply_mode, discover_components


# ---------------------------------------------------------------------------
# Collect specs once at import time (runs during pytest collection)
# ---------------------------------------------------------------------------

_ALL_SPECS = discover_components()


def _spec_id(spec: BenchSpec, scale: str, mode: str = "eager") -> str:
    base = f"{spec.module}/{spec.name}[{scale}]"
    if mode != "eager":
        base += f"/{mode}"
    return base


# ---------------------------------------------------------------------------
# Dynamic parametrization
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    if "bench_spec" not in metafunc.fixturenames:
        return
    module_filter = metafunc.config.getoption("--bench-module", default=None)
    scale_filter = set(
        metafunc.config.getoption("--bench-scales", default=list(SCALES.keys()))
    )

    modes = ["eager"]
    if metafunc.config.getoption("--bench-compile", default=False):
        modes.append("compiled")
    if metafunc.config.getoption("--bench-amp", default=False):
        modes.append("amp_fp16")

    params = []
    ids = []
    for spec in _ALL_SPECS:
        if module_filter and spec.module != module_filter:
            continue
        if not is_module_enabled(spec.module):
            continue
        if is_benchmark_excluded(spec.name):
            continue
        for scale in SCALES:
            if scale not in scale_filter:
                continue
            for mode in modes:
                params.append((spec, scale, mode))
                ids.append(_spec_id(spec, scale, mode))

    metafunc.parametrize("bench_spec", params, ids=ids, indirect=True)


@pytest.fixture
def bench_spec(request):
    return request.param


# ---------------------------------------------------------------------------
# Single test entry-point — all categories
# ---------------------------------------------------------------------------


class TestBenchmarks:

    def test_component(self, benchmark, bench_device, bench_dtype, bench_spec):
        spec, scale, mode = bench_spec
        cfg = SCALES[scale]
        dim, bs, n_steps = cfg["dim"], cfg["batch_size"], cfg["n_steps"]

        builder = TEMPLATE_BUILDERS[spec.module]

        if spec.module == "models":
            fn, info = builder(
                spec, dim, bs, n_steps, bench_device, bench_dtype, scale=scale
            )
        else:
            fn, info = builder(spec, dim, bs, n_steps, bench_device, bench_dtype)

        # Keep eager fn for FLOPS estimation (compile/AMP alter execution)
        eager_fn = fn

        # Extract internal fields before updating extra_info
        quality_fn = info.pop("_quality_fn", None)
        counting_drift = info.pop("_counting_drift", None)

        # Apply mode transformations
        fn = apply_mode(fn, mode, bench_device)
        info["mode"] = mode

        info["scale"] = scale
        benchmark.extra_info.update(info)
        warmup, rounds, iters = get_bench_params(bench_device, mode=mode)
        benchmark.pedantic(
            fn,
            setup=make_pedantic_setup(bench_device),
            warmup_rounds=warmup,
            rounds=rounds,
            iterations=iters,
        )

        # -- Post-benchmark metrics --

        # Throughput (samples/sec)
        if benchmark.stats and benchmark.stats.get("median", 0) > 0:
            batch_size = benchmark.extra_info.get("batch_size", 0)
            if batch_size > 0:
                benchmark.extra_info["samples_per_sec"] = round(
                    batch_size / benchmark.stats["median"], 2
                )

        # FLOPS estimation (always use eager fn)
        try:
            from torch.utils.flop_counter import FlopCounterMode

            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                eager_fn()
            total_flops = flop_counter.get_total_flops()
            if total_flops > 0:
                benchmark.extra_info["total_flops"] = total_flops
                if benchmark.stats and benchmark.stats.get("median", 0) > 0:
                    benchmark.extra_info["gflops_per_sec"] = round(
                        total_flops / benchmark.stats["median"] / 1e9, 2
                    )
        except (ImportError, Exception):
            pass

        # Quality metrics (sampler quality, loss sanity)
        if quality_fn is not None:
            try:
                quality = quality_fn()
                if quality:
                    benchmark.extra_info.update(quality)
                    # ESS per second
                    if (
                        "ess" in quality
                        and benchmark.stats
                        and benchmark.stats.get("median", 0) > 0
                    ):
                        benchmark.extra_info["ess_per_sec"] = round(
                            quality["ess"] / benchmark.stats["median"], 2
                        )
            except Exception:
                pass

        # -- GPU-centric derived metrics --

        # Gradient throughput (samplers / losses)
        grad_evals = benchmark.extra_info.get("gradient_evals", 0)
        if grad_evals > 0 and benchmark.stats and benchmark.stats.get("median", 0) > 0:
            benchmark.extra_info["gradient_evals_per_sec"] = round(
                grad_evals / benchmark.stats["median"], 2
            )

        # Adaptive integrator drift evaluation efficiency
        if counting_drift is not None:
            try:
                counting_drift.count = 0
                eager_fn()
                n_steps_cfg = benchmark.extra_info.get("n_steps", 1)
                benchmark.extra_info["drift_evals_total"] = counting_drift.count
                benchmark.extra_info["drift_evals_per_step"] = round(
                    counting_drift.count / max(n_steps_cfg, 1), 2
                )
            except Exception:
                pass

        # Memory per sample
        peak_mem = benchmark.extra_info.get("peak_memory_mb")
        batch_size = benchmark.extra_info.get("batch_size", 0)
        if peak_mem is not None and batch_size > 0:
            benchmark.extra_info["memory_per_sample_kb"] = round(
                peak_mem * 1024 / batch_size, 2
            )
