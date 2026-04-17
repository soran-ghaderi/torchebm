---
sidebar_position: 10
title: Benchmarking System
description: How TorchEBM measures and tracks end-to-end performance
icon: material/chart-timeline-variant
---

# Benchmarking System

TorchEBM uses [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) to measure end-to-end wall time, throughput, FLOPS, and memory for every registered component across releases. Results are saved as JSON, compared against a baseline, and published to the [TorchEBM Benchmarks Dashboard](https://soran-ghaderi.github.io/torchebm-benchmarks/).

See the [Profiling & Benchmarks](profiling.md) page for the complementary per-op drill-down tool.

---

## How it works

```
                       ┌────────────────────────────┐
  registry discovers ──▶  benchmarks/registry.py    │
  components from      │  (TEMPLATE_BUILDERS)        │
  torchebm.*           └──────────────┬──────────────┘
                                      ▼
                       ┌────────────────────────────┐
                       │  test_bench_auto.py        │
                       │  parametrizes              │
                       │  (component × scale × mode)│
                       └──────────────┬──────────────┘
                                      ▼
                       ┌────────────────────────────┐
                       │  benchmark.pedantic(fn)    │
                       │  CUDA sync + warmup        │
                       └──────────────┬──────────────┘
                                      ▼
           benchmarks/results/<date>_<sha>.json  →  dashboard
```

- **Zero test boilerplate.** `test_bench_auto.py` picks up every component exported from `torchebm.*.__init__` and times its standard workload (forward + backward for losses, step for samplers/integrators, `.interpolate` for interpolants, etc.).
- **Scales.** Each benchmark runs at `small`, `medium`, `large` (batch size / dim / steps, defined in `benchmarks/conftest.py::SCALES`).
- **Modes.** `eager` by default; `--compile` and `--amp` add `torch.compile` and `float16` autocast variants via the same `registry.apply_mode()` the profiler uses, so results stay directly comparable.
- **CUDA-aware.** `benchmark.pedantic()` calls `torch.cuda.synchronize()` and tracks peak/reserved memory per run.

---

## Quick start

```bash
# Full suite
bash benchmarks/run.sh

# Smoke test (small scale only)
bash benchmarks/run.sh --quick

# One module
bash benchmarks/run.sh --module losses

# One benchmark by name (pytest -k)
bash benchmarks/run.sh --filter "ScoreMatching"

# With compile / AMP variants
bash benchmarks/run.sh --compile --amp
```

Outputs land in `benchmarks/results/`. The wrapper also keeps a versioned copy `v{torchebm_version}_{device}_{timestamp}.json` and updates `baseline_{device}.json` on first run.

---

## Comparing runs

```bash
# Save current results as the new baseline
bash benchmarks/run.sh --baseline

# Compare the latest two runs (uses pytest-benchmark's built-in compare)
bash benchmarks/run.sh --compare

# Regenerate the local HTML dashboard
bash benchmarks/run.sh --dashboard
```

For CI-style regression checks:

```bash
bash benchmarks/run.sh --ci
```

This exits non-zero if the geometric-mean speedup versus baseline is below 0.95×.

!!! tip "Benchmarks vs. Profiler"
    Use benchmarks to **detect** a regression; use [`profiler.py diff`](profiling.md#the-diff-command) on the same component to **explain** it op by op.

---

## Enabling in CI / editor

Benchmarks are **disabled by default** (`--benchmark-disable` in `pyproject.toml`) so regular `pytest tests/` never runs them. They opt in explicitly:

```bash
pytest benchmarks/ --benchmark-only --benchmark-enable
```

GitHub Actions triggers the GPU runner on `workflow_dispatch`, merges to `main`, or any commit containing `#bench` in the message.

---

## Adding a benchmark

You usually do **nothing**. Any class exported from a `torchebm.*` subpackage that inherits a known base (`BaseLoss`, `BaseSampler`, `BaseIntegrator`, `BaseInterpolant`) is auto-discovered.

Add an entry to `benchmarks/registry.py::COMPONENT_OVERRIDES` only if your component needs non-default construction. The dataclass keys are short and self-documenting; existing entries are the best reference:

```python
COMPONENT_OVERRIDES["MyNewLoss"] = {
    "model_type": "ebm",          # or "velocity", "ebm_double_well", "none"
    "init_kwargs": {"k_steps": 5},
    "needs_grad": True,            # x.requires_grad_(True)
    "max_dim": 64,                 # cap for O(dim²) methods
}
```

Variants (e.g. `exact` vs `approx` modes) use the `variants` key:

```python
COMPONENT_OVERRIDES["ScoreMatching"] = {
    "model_type": "ebm",
    "needs_grad": True,
    "variants": {
        "score_matching_exact":  {"init_kwargs": {"hessian_method": "exact"}, "max_dim": 16},
        "score_matching_approx": {"init_kwargs": {"hessian_method": "approx"}},
    },
}
```

To temporarily exclude a benchmark, edit `benchmarks/benchmark.toml`:

```toml
[modules]
models = false                 # disable a whole module

[exclude]
benchmarks = ["dopri5"]        # exclude a single benchmark by name
```

---

## Publishing results

Benchmark JSONs and the dashboard live in a [separate repo](https://github.com/soran-ghaderi/torchebm-benchmarks) so this one stays lean. After a run:

1. Copy the autosaved JSON from `benchmarks/results/Linux-CPython-*/` to the `torchebm-benchmarks` repo.
2. In that repo, run `bash scripts/publish.sh <path-to-json>`. it updates `manifest.json`, commits, and pushes.
3. GitHub Pages auto-deploys the dashboard.

---

## Files in this system

| File | Role |
|------|------|
| `benchmarks/registry.py` | Auto-discovers components, defines `BenchSpec`, builds `(fn, info)` pairs, exposes `apply_mode()` shared with the profiler. |
| `benchmarks/test_bench_auto.py` | Parametrizes pytest over (component × scale × mode) and drives `benchmark.pedantic`. |
| `benchmarks/conftest.py` | CLI options (`--bench-*`), `SCALES`, CUDA sync hooks, pedantic setup. |
| `benchmarks/benchmark.toml` | Enable / disable modules and exclude benchmarks by name. |
| `benchmarks/run.sh` | Shell wrapper with `--quick`, `--baseline`, `--compare`, `--dashboard`, `--ci`. |
| `benchmarks/dashboard.py` | Generates a local `dashboard.html` from the result JSONs (the published version lives in the separate benchmarks repo). |
| `benchmarks/torchebm_benchmarking.md` | Full architecture reference (for repo maintainers). |

!!! note
    For internal details. CI workflow, `extra_info` metadata schema, Plotly dashboard structure. see `benchmarks/torchebm_benchmarking.md` in the repo.
