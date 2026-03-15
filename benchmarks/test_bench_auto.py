"""
Auto-generated benchmarks from the TorchEBM component registry.

pytest collects one test per (component × scale) combination.  New
components exported in ``torchebm.*.__init__`` are picked up
automatically; only components with unusual construction requirements
need an entry in ``registry.COMPONENT_OVERRIDES``.
"""

import copy

import pytest

from conftest import (
    SCALES,
    get_bench_params,
    is_benchmark_excluded,
    is_module_enabled,
    make_pedantic_setup,
)
from registry import TEMPLATE_BUILDERS, BenchSpec, discover_components


# ---------------------------------------------------------------------------
# Collect specs once at import time (runs during pytest collection)
# ---------------------------------------------------------------------------

_ALL_SPECS = discover_components()


def _spec_id(spec: BenchSpec, scale: str) -> str:
    return f"{spec.module}/{spec.name}[{scale}]"


# ---------------------------------------------------------------------------
# Dynamic parametrization
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    if "bench_spec" not in metafunc.fixturenames:
        return
    module_filter = metafunc.config.getoption("--bench-module", default=None)
    scale_filter = set(metafunc.config.getoption("--bench-scales", default=list(SCALES.keys())))

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
            params.append((spec, scale))
            ids.append(_spec_id(spec, scale))

    metafunc.parametrize("bench_spec", params, ids=ids, indirect=True)


@pytest.fixture
def bench_spec(request):
    return request.param


# ---------------------------------------------------------------------------
# Single test entry-point — all categories
# ---------------------------------------------------------------------------


class TestBenchmarks:

    def test_component(self, benchmark, bench_device, bench_dtype, bench_spec):
        spec, scale = bench_spec
        cfg = SCALES[scale]
        dim, bs, n_steps = cfg["dim"], cfg["batch_size"], cfg["n_steps"]

        builder = TEMPLATE_BUILDERS[spec.module]

        if spec.module == "models":
            fn, info = builder(spec, dim, bs, n_steps, bench_device, bench_dtype, scale=scale)
        else:
            fn, info = builder(spec, dim, bs, n_steps, bench_device, bench_dtype)

        info["scale"] = scale
        benchmark.extra_info.update(info)
        warmup, rounds, iters = get_bench_params(bench_device)
        benchmark.pedantic(
            fn,
            setup=make_pedantic_setup(bench_device),
            warmup_rounds=warmup,
            rounds=rounds,
            iterations=iters,
        )
