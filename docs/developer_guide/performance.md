---
title: Performance and Benchmarking
description: Writing fast TorchEBM code, and measuring it when the evidence matters
icon: material/speedometer
---

# Performance and Benchmarking

Patterns for writing performance-sensitive code, and the two measurement tools
that back any performance claim: the benchmark suite (how fast is it?) and the
profiler (where does the time go?).

## The four hot spots

1. **Sampler steps**: iterative, run 100-1000x, dominate wall time.
2. **Score / energy gradients**: `autograd.grad` calls are frequent and stack.
3. **Loss forward + backward**: called every training batch.
4. **Host-device traffic**: `.item()`, `.cpu()`, repeated `.to()` stall the GPU.

Optimise in that order. Everything else is noise.

## Vectorise, don't loop

Work in batch dimensions; avoid Python-level iteration over samples.

```python
# bad
energies = torch.stack([energy_fn(x[i]) for i in range(batch)])

# good
energies = energy_fn(x)                       # (batch,)
```

Sample many chains in parallel by putting the chain index in the leading dim.

## Stay on device

Keep tensors on one device and dtype for the whole pipeline. Use
`self.device` / `self.dtype` from `TorchEBMModule` inside the library; never
hard-code `"cuda"`.

`.item()`, `.cpu()`, `.tolist()`, and Python `if tensor > 0:` all trigger a
full GPU sync; defer them until after the hot loop:

```python
# bad: syncs every step
for step in range(n_steps):
    loss = loss_fn(x)
    if loss.item() < threshold:               # GPU stall
        break

# good: one sync per epoch
losses = []
for step in range(n_steps):
    losses.append(loss_fn(x).detach())
avg = torch.stack(losses).mean().item()       # single sync
```

## Reuse memory

Pre-allocate buffers once and fill them in place inside loops; for
trajectories, write into a pre-allocated tensor instead of appending to a list:

```python
traj = torch.empty(n_steps + 1, *x.shape, device=device, dtype=x.dtype)
traj[0] = x
for i in range(n_steps):
    x = sampler.step(x)
    traj[i + 1] = x
```

In-place ops (`x.add_`, `x.mul_`) are safe outside autograd-tracked paths.

## Mixed precision and compilation

Inside the library, wrap large matmul blocks with `self.autocast_context()`
rather than calling `torch.autocast` directly; this honours the user's
configured dtype. At the benchmark/application layer, `--amp` and `--compile`
apply the same transforms via `benchmarks/registry.py::apply_mode`, so eager,
compiled, and mixed-precision results stay comparable.

## Common pitfalls

- Implicit host-device copies: `torch.tensor(x_numpy, device=...)` inside a loop.
- Redundant `.to()` calls: `BaseLoss.__call__` already moves inputs; subclass `forward()` must not.
- Missing `torch.no_grad()`: interpolation targets, momentum init, and random projections need no grad tracking.
- Tiny batches on GPU: prefer one big step over many small ones.
- `isinstance` checks inside the inner loop: resolve once before the loop.

## Benchmarks: detecting change

The suite ([pytest-benchmark](https://pytest-benchmark.readthedocs.io/) under
`benchmarks/`) auto-discovers every component exported from
`torchebm.*.__init__` and times its standard workload at three scales. Regular
`pytest tests/` never runs them.

```bash
bash benchmarks/run.sh --quick               # smoke run, small scale
bash benchmarks/run.sh --module losses       # one module
bash benchmarks/run.sh --filter "ScoreMatching"
bash benchmarks/run.sh --baseline            # save current results as baseline
bash benchmarks/run.sh --compare             # compare latest two runs
bash benchmarks/run.sh --ci                  # fail if geo-mean speedup < 0.95x
```

Components needing non-default construction get an entry in
`benchmarks/registry.py::COMPONENT_OVERRIDES`; existing entries are the best
reference. Modules and individual benchmarks can be excluded in
`benchmarks/benchmark.toml`.

**Publishing.** Results and the dashboard live in the separate
[torchebm-benchmarks](https://github.com/soran-ghaderi/torchebm-benchmarks)
repository, deployed at
[soran-ghaderi.github.io/torchebm-benchmarks](https://soran-ghaderi.github.io/torchebm-benchmarks/).
After a run: copy the autosaved JSON from
`benchmarks/results/Linux-CPython-*/` into that repo and run
`bash scripts/publish.sh <path-to-json>`; GitHub Pages auto-deploys.

## Profiling: explaining change

`benchmarks/profiler.py` wraps `torch.profiler` around the same registry
callables the benchmarks time. Profile only when an optimisation is
non-trivial and evidence-driven: a dashboard regression to localise, a hot
path rewrite to justify, or a suspected memory issue. Skip it for one-line
fixes and cleanups.

The whole workflow is one before/after pair plus a diff:

```bash
python benchmarks/profiler.py run --component LangevinDynamics --scale medium --top 20 --label before
# make the change
python benchmarks/profiler.py run --component LangevinDynamics --scale medium --top 20 --label after
python benchmarks/profiler.py diff benchmarks/profiles/*_before benchmarks/profiles/*_after
```

Add `--trace` only when the top-N table is not enough (open in
[ui.perfetto.dev](https://ui.perfetto.dev)), `--memory` for allocator work
(view at [pytorch.org/memory_viz](https://pytorch.org/memory_viz)), `--nvtx`
for Nsight Systems. Arbitrary callables profile via
`--callable module:factory`. Outputs land under `benchmarks/profiles/`
(gitignored; profiles are local by design).

**Division of labour**: benchmarks detect a regression and track it across
releases; the profiler explains it op by op on one run.
