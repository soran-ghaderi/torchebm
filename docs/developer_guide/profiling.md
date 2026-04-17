---
sidebar_position: 9
title: Profiling
description: Drilling into where time and memory go inside a TorchEBM component
icon: material/chart-line
---

# Profiling

TorchEBM ships with two complementary performance tools under `benchmarks/`:

| Tool | Answers | Tracks change over time? |
|------|---------|--------------------------|
| [Benchmarking](benchmarking.md) (`benchmarks/run.sh`) | *How fast is this component?*. wall time, throughput, FLOPS | yes, via baselines and the dashboard |
| **`benchmarks/profiler.py`** (this page) | *Where does time and memory go inside it?*. per-op CPU/CUDA time, kernels, memory events | yes, via `profiler.py diff` |

Use benchmarks to detect a regression; use the profiler to explain it.

---

## How it works

`profiler.py` is a thin wrapper around `torch.profiler` that reuses the benchmark component registry, so the callable you profile is **identical** to the one pytest-benchmark times.

```
                          ┌────────────────────────┐
  --component NAME ──────▶│  benchmarks/registry   │──┐
                          └────────────────────────┘  │
                          ┌────────────────────────┐  ├──▶ zero-arg fn()
  --callable mod:attr ───▶│  importlib             │──┘        │
                          └────────────────────────┘           ▼
                                              ┌─ torch.profiler.profile ──┐
                                              │   + memory snapshot       │
                                              │   + NVTX ranges           │
                                              └─────────────┬─────────────┘
                                                            ▼
                          benchmarks/profiles/<run>/{meta,trace,top_ops,…}
```

There is **zero instrumentation inside `torchebm/*`**: all profiling lives in `benchmarks/`. Nothing in the library is aware that it is being profiled.

---

## Quick start

Profile one component at a scale defined by the benchmark suite:

```bash
python benchmarks/profiler.py run \
    --component LangevinDynamics \
    --scale medium \
    --trace --top 20 --memory
```

Profile a full training step (any importable zero-arg callable or factory):

```bash
python benchmarks/profiler.py run \
    --callable examples.eqm.train_2d:profile_one_step \
    --trace --top 20
```

Compare two runs to find which op regressed:

```bash
python benchmarks/profiler.py diff \
    benchmarks/profiles/<run_a> \
    benchmarks/profiles/<run_b>
```

---

## Outputs

Each `run` creates one directory under `benchmarks/profiles/`:

```
<target>_<device>_<mode>_<timestamp>[_<label>]/
├── meta.json              # device, dtype, torch/torchebm versions, git sha, argv
├── trace.json             # --trace   → open in ui.perfetto.dev or chrome://tracing
├── top_ops.md             # --top N   → human-readable top-N table
├── top_ops.json           # --top N   → machine-readable (source of truth for diff)
├── memory_snapshot.pickle # --memory  → view at pytorch.org/memory_viz
└── line_prof.txt          # --line module:fn (requires line_profiler)
```

The directory is gitignored; profiles are local by design.

!!! tip "Pick one lens at a time"
    Start with `--top 20` to see the hot ops. Add `--trace` when you need a timeline, `--memory` when you suspect allocations, `--nvtx` when you are going deeper with Nsight Systems.

---

## The `run` command

| Flag | Purpose |
|------|---------|
| `--component NAME` | Profile a registered component (e.g. `LangevinDynamics`, `ScoreMatching`). |
| `--callable mod:attr` | Profile any zero-arg callable. If `attr` has required args it is invoked once as a factory; the returned callable is profiled. |
| `--scale {small,medium,large}` | Applies to `--component`. Matches `benchmarks/conftest.py::SCALES`. |
| `--device {cpu,cuda}` | Auto-falls back to CPU when CUDA is unavailable. |
| `--dtype {float32,float64,float16,bfloat16}` | Target dtype for the component. |
| `--warmup N` / `--steps N` | Warmup iterations (untraced) and profiled iterations. |
| `--compile` / `--amp` | Wrap the callable with `torch.compile` or `torch.autocast(float16)`. same transform pytest-benchmark uses. |
| `--trace` | Export Chrome-format trace. |
| `--top N` | Write `top_ops.md` + `top_ops.json` with the top N ops. |
| `--memory` | CUDA only. Record and dump a memory history snapshot. |
| `--nvtx` | CUDA only. Wrap each step in NVTX ranges; prints the matching `nsys profile` command. |
| `--line module:fn` | Optional. Line-level profile of a specific function (requires `pip install line_profiler`). |
| `--all` | Shortcut for `--trace --top 20 --memory --nvtx`. |
| `--label TAG` | Free-form tag appended to the output directory name. |

---

## The `diff` command

`diff` joins two `top_ops.json` rowsets by op name, sorts by \( |\Delta t| \), and prints a markdown table:

```bash
python benchmarks/profiler.py diff \
    benchmarks/profiles/<before> \
    benchmarks/profiles/<after> \
    --metric self_cuda_time_total_us \
    --top 20
```

Columns: op, time in run A, time in run B, absolute Δ, percent Δ, call counts, and a `status` of `changed` / `new` / `dropped`. `diff` also accepts paths directly to `top_ops.json` files.

Available metrics: `self_cuda_time_total_us`, `cuda_time_total_us`, `self_cpu_time_total_us`, `cpu_time_total_us`, `self_cuda_mem_b`, `self_cpu_mem_b`.

---

## Typical workflows

=== "Find a hot op"
    ```bash
    python benchmarks/profiler.py run --component ScoreMatching --scale medium --top 20
    ```
    Open `benchmarks/profiles/*/top_ops.md` and look at the top rows.

=== "A/B a change"
    ```bash
    git checkout main
    python benchmarks/profiler.py run --component LangevinDynamics --scale medium --top 30 --label before
    git checkout my-optimization
    python benchmarks/profiler.py run --component LangevinDynamics --scale medium --top 30 --label after
    python benchmarks/profiler.py diff benchmarks/profiles/*_before benchmarks/profiles/*_after
    ```

=== "Visual timeline"
    ```bash
    python benchmarks/profiler.py run --component FlowSampler --scale small --trace
    ```
    Drag `trace.json` into [ui.perfetto.dev](https://ui.perfetto.dev).

=== "Memory attribution"
    ```bash
    python benchmarks/profiler.py run --component ContrastiveDivergence --device cuda --memory --steps 30
    ```
    Upload `memory_snapshot.pickle` to [pytorch.org/memory_viz](https://pytorch.org/memory_viz).

=== "Nsight Systems"
    ```bash
    python benchmarks/profiler.py run --component HamiltonianMonteCarlo --device cuda --nvtx
    # then run the nsys command printed at the end
    ```

=== "Train-step end-to-end"
    ```bash
    python benchmarks/profiler.py run \
        --callable examples.eqm.train_2d:profile_one_step \
        --trace --top 20 --memory
    ```

---

## Adding a new profiling target

**Components** exported from `torchebm.*` are picked up automatically by `benchmarks/registry.py`, so nothing to do. new components just appear as valid `--component` names.

**Custom callables** only need to be importable and either take no required arguments or act as a factory returning a zero-arg callable:

```python
# anywhere importable, e.g. examples/my_model/train.py
def profile_one_step(batch_size: int = 512):
    model, loss_fn, optim, data = _setup(batch_size)

    def step():
        optim.zero_grad(set_to_none=True)
        loss_fn(data()).backward()
        optim.step()

    return step
```

Then:

```bash
python benchmarks/profiler.py run --callable examples.my_model.train:profile_one_step --trace --top 20
```

See [`examples/eqm/train_2d.py`](https://github.com/soran-ghaderi/torchebm/blob/main/examples/eqm/train_2d.py) for a working reference.

---

## Relation to the benchmark suite

- The **[benchmark suite](benchmarking.md)** produces versioned JSONs and the dashboard for tracking wall time across releases. your primary regression tracker.
- The **profiler** produces per-op evidence for a specific run. Reach for it whenever the dashboard shows a regression or you want to know where to optimize next.

Both tools share the same component builders via `benchmarks/registry.py::apply_mode`, so eager vs. `--compile` vs. `--amp` results stay directly comparable between them.

!!! note
    For full system details (conftest fixtures, scale configs, CI deploy), see `benchmarks/torchebm_benchmarking.md` in the repo.
