---
sidebar_position: 8
title: Performance
description: Writing fast TorchEBM code. vectorization, memory, and GPU patterns
icon: material/speedometer
---

# Performance

Patterns we follow when writing performance-sensitive code. For *measuring* performance, see [Benchmarking](benchmarking.md); for *debugging* a specific regression, see [Profiling](profiling.md).

---

## The four hot spots

1. **Sampler steps**: iterative, run 100–1000×, dominate wall time.
2. **Score / energy gradients**: `autograd.grad` calls are frequent and stack.
3. **Loss forward + backward**: called every training batch.
4. **Host ↔ device traffic**: `.item()`, `.cpu()`, repeated `.to()` stall the GPU.

Optimise in that order. Everything else is noise.

---

## Vectorise, don't loop

Work in batch dimensions; avoid Python-level iteration over samples.

```python
# bad
energies = torch.stack([energy_fn(x[i]) for i in range(batch)])

# good
energies = energy_fn(x)                       # (batch,)
```

Sample many chains in parallel by putting the chain index in the leading dim:

```python
x = torch.randn(n_chains, dim, device=device)
x, _ = sampler.step(x)                        # all chains advance together
```

---

## Stay on device

Keep tensors on the same device and dtype for the whole pipeline. Use `DeviceMixin`'s `self.device` / `self.dtype` inside the library; never hard-code `cuda`.

!!! warning "Do not sync unnecessarily"
    `.item()`, `.cpu()`, `.tolist()`, and Python `if tensor > 0:` all trigger a full GPU sync. Defer them until after the hot loop, ideally until logging at the epoch boundary.

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

---

## Reuse memory

Pre-allocate buffers once, reuse in the loop:

```python
# bad: allocates every call
def drift(x, t):
    t_batch = torch.ones(x.shape[0], device=x.device) * t

# good: allocate once, fill in place
t_batch = torch.empty(batch, device=device)
for step in range(n_steps):
    t_batch.fill_(t_value)
```

For trajectories, write into a pre-allocated tensor instead of appending to a list and stacking at the end:

```python
traj = torch.empty(n_steps + 1, *x.shape, device=device, dtype=x.dtype)
traj[0] = x
for i in range(n_steps):
    x = sampler.step(x)
    traj[i + 1] = x
```

In-place ops (`x.add_`, `x.mul_`) are safe outside of autograd-tracked paths.

---

## Mixed precision and compilation

Both are opt-in at the benchmark / application layer via the same entry point the profiler uses:

```python
from benchmarks.registry import apply_mode
fn = apply_mode(fn, mode="amp",     device=device)   # float16 autocast
fn = apply_mode(fn, mode="compile", device=device)   # torch.compile
```

Inside the library, wrap large matmul blocks with `self.autocast_context()` (provided by `DeviceMixin`) rather than calling `torch.autocast` directly. this honours the user's configured dtype.

---

## Sampler-specific tips

=== "Langevin dynamics"
    Rough scaling: step size \( \sim d^{-1/3} \), noise scale \( \sigma = \sqrt{2\eta} \).

    ```python
    step_size = min(0.01, 0.1 * dim ** (-1 / 3))
    noise_scale = (2 * step_size) ** 0.5
    ```

=== "HMC"
    Rough scaling: leapfrog steps \( \sim \sqrt{d} \), step size \( \sim d^{-1/4} \). Target acceptance 0.6–0.8.

    ```python
    n_leapfrog = max(5, int(dim ** 0.5))
    step_size  = min(0.01, 0.05 * dim ** (-1 / 4))
    ```

=== "Flow / diffusion"
    Prefer adaptive integrators (`DOPRI5`, `Heun`) for generation; fixed-step for training. Keep the ODE function allocation-free. see the pre-allocation pattern above.

---

## Common pitfalls

- **Implicit host ↔ device copies**: `torch.tensor(x_numpy, device=…)` inside a loop.
- **Redundant `.to()` calls**: `BaseLoss.__call__` already moves inputs; subclass `forward()` should not move them again.
- **Missing `torch.no_grad()`**: interpolation targets, momentum init, and random projection generation don't need grad tracking.
- **Tiny batches on GPU**: under-utilises SMs; prefer one big step over many small ones.
- **Python-level `isinstance` inside the inner loop**: resolve once before the loop.

---

## Next steps

<<<<<<< HEAD
- Evidence first: [Benchmarking](benchmarking.md) tells you *how fast*; [Profiling](profiling.md) tells you *where the time goes*.
- CUDA kernels: planned. See [cuRBLAS](https://github.com/soran-ghaderi/cuRBLAS) for background.
- Multi-GPU scaling: planned.
=======
    Initialize sampling from a distribution close to the target.

    ```python
    # Warm start from approximate distribution
    x = approximate_sampler.sample(n_samples, dim)
    samples = sampler.sample(
      n_steps=n_steps, 
      initial_samples=x
    )
    ```

</div>

## Benchmarking Workflow

TorchEBM ships with a pytest-benchmark suite that auto-discovers components across all modules. Results are stored as JSON files and can be visualized in an interactive HTML dashboard.

### Quick Reference

```bash
# Full run (all modules, all scales — slowest, most thorough)
bash benchmarks/run.sh

# Fast smoke test (small scale only)
bash benchmarks/run.sh --quick

# Benchmark a single module
bash benchmarks/run.sh --module losses
bash benchmarks/run.sh --module samplers
bash benchmarks/run.sh --module integrators

# Benchmark a specific component by name
bash benchmarks/run.sh --filter langevin
bash benchmarks/run.sh --filter "score_matching"

# Choose scale(s)
bash benchmarks/run.sh --module losses --scales "small medium"

# With torch.compile or mixed precision
bash benchmarks/run.sh --compile
bash benchmarks/run.sh --amp

# Force CPU even if CUDA is available
bash benchmarks/run.sh --device cpu

# Save current results as the baseline
bash benchmarks/run.sh --baseline

# Compare latest run against baseline
bash benchmarks/run.sh --compare

# Regenerate dashboard from existing results
bash benchmarks/run.sh --dashboard
```

### Targeted vs Full Benchmarking

**You do not need to run the full suite every time.** The `--module` and `--filter` flags exist specifically for targeted runs:

| Scenario | Command | Time |
|----------|---------|------|
| Changed something in `losses/` | `bash benchmarks/run.sh --module losses --quick` | ~1-2 min |
| Changed a single sampler | `bash benchmarks/run.sh --filter langevin --quick` | ~30 sec |
| Changed core integrator code | `bash benchmarks/run.sh --module integrators` | ~3-5 min |
| Pre-merge full validation | `bash benchmarks/run.sh` | ~15-30 min |
| CI regression gate | `bash benchmarks/run.sh --ci` | ~15-30 min |

**When to run what:**

- **During development**: Use `--module` or `--filter` with `--quick` after each change. This takes seconds, not hours. Enough to catch regressions in the component you touched.
- **Before a PR**: Run the full suite once (`bash benchmarks/run.sh`) to catch cross-module interactions. Only needed once per PR, not per commit.
- **CI (optional)**: The GitHub Actions workflow in `.github/workflows/benchmarks.yml` runs on commits containing `#bench` in the message or on `workflow_dispatch`. It is **not** triggered on every merge — this is intentional for teams without dedicated benchmark hardware.

### The Baseline/Compare Workflow

This is the core loop for validating that a change actually improves performance:

**Step 1 — Save a baseline before making changes:**

```bash
bash benchmarks/run.sh --module losses --baseline
```

This saves the current results as `benchmarks/results/baseline_{device}.json`.

**Step 2 — Make your changes.**

**Step 3 — Run the same benchmarks again:**

```bash
bash benchmarks/run.sh --module losses
```

The runner automatically compares against the saved baseline and prints a summary.

**Step 4 — View the comparison:**

```bash
bash benchmarks/run.sh --compare
# or open benchmarks/results/dashboard.html
```

The dashboard shows per-benchmark median times with percentage change.

!!! warning "Always compare on the same machine under similar conditions"
    Absolute numbers vary across hardware. Always use baseline/compare pairs from the same machine. Close other heavy programs during benchmarking for stable results.

### Benchmark Scales

The suite supports three predefined scales. Larger scales stress GPU utilization but take longer:

| Scale | batch_size | dim | n_steps | Use Case |
|-------|-----------|-----|---------|----------|
| small | 64 | 8 | 50 | Quick smoke tests during development |
| medium | 256 | 32 | 100 | Default development benchmarking |
| large | 1024 | 128 | 200 | Pre-merge performance validation |

Choose your scale with `--scales "small"` (or `--quick` as a shorthand for small-only).

### Configuration

Modules and individual benchmarks can be enabled/disabled in `benchmarks/benchmark.toml`:

```toml
[modules]
losses = true
samplers = true
integrators = true
models = false        # disabled by default (slow)
interpolants = true

[exclude]
benchmarks = [
    # "dopri5",
    # "transformer_fwd_bwd",
]
```

### Regression Detection

In CI mode (`--ci`), the runner computes a **geometric mean speedup** across all benchmarks compared to the previous run:

- \(\geq 0.95\times\): **PASS** — no significant regression
- \(< 0.95\times\): **FAIL** — regression detected, CI exits non-zero

This is conservative: a 5% slowdown anywhere triggers investigation. For PR reviews, the CI workflow posts a speedup table as a PR comment.

## Profiling and Debugging Performance

When benchmark numbers don't improve as expected, use profiling to find the actual bottleneck.

### Import Time

Measure how long `import torchebm` takes and which submodules are slow:

```bash
python -X importtime -c "import torchebm" 2>&1 | head -30
```

This prints a tree with cumulative microseconds per import. Useful for validating lazy-import optimizations.

### Quick Micro-Benchmarks

For validating that a specific code change (e.g., `.detach()` vs `.detach().clone()`) helps, use `torch.utils.benchmark.Timer` in a scratch script:

```python
import torch
import torch.utils.benchmark as bench

x = torch.randn(1024, 128, device="cuda")

t0 = bench.Timer(stmt="x.detach().clone()", globals={"x": x})
t1 = bench.Timer(stmt="x.detach()", globals={"x": x})

print(t0.blocked_autorange())
print(t1.blocked_autorange())
```

This handles CUDA synchronization, warmup, and statistical noise correctly. You don't need to commit these — they're for quick validation during development.

### GPU Kernel Profiling

For deeper analysis (which CUDA kernels dominate, where time is spent):

```python
import torch

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True,
) as prof:
    # Run the workload you want to profile
    result = sampler.sample(dim=32, n_steps=100, n_samples=256)

# Print a summary table sorted by CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Or export for Chrome trace viewer (chrome://tracing)
prof.export_chrome_trace("trace.json")
```

### Memory Profiling

Track peak GPU memory to ensure optimizations don't trade speed for excessive memory:

```python
torch.cuda.reset_peak_memory_stats()
# ... run workload ...
peak_mb = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak GPU memory: {peak_mb:.1f} MB")
```

The benchmark suite tracks this automatically via the `--bench-device=cuda` fixture.

## Performance Benchmarks

The benchmark suite auto-discovers components from `torchebm.*` exports and generates parametrized tests. Current coverage:

| Module | Components | Metrics |
|--------|-----------|---------|
| Integrators | Leapfrog, EulerMaruyama, RK4, DOPRI5, Heun, AdaptiveHeun, Bosh3 | Time, throughput, drift evals/step |
| Losses | ScoreMatching, SlicedScoreMatching, ContrastiveDivergence, EquilibriumMatchingLoss | Forward+backward+zero_grad time |
| Samplers | LangevinDynamics, HMC, FlowSampler | Time, acceptance rate, ESS |
| Interpolants | Linear, Cosine, VariancePreserving | Interpolate time |
| Models | ConditionalTransformer2D (disabled by default) | Forward, forward+backward |

Results are stored locally in `benchmarks/results/` as JSON files (gitignored). Published results and the interactive dashboard are maintained in the separate [`torchebm-benchmarks`](https://github.com/soran-ghaderi/torchebm-benchmarks) repository.

### Benchmark Results Repository

Benchmark results are kept in a dedicated repository, separate from the main codebase:

```
torchebm-benchmarks/
├── results/                  # All historical JSON results
│   ├── v0.5.12_cuda_A100/   # Grouped by version + GPU
│   ├── v0.6.0_cuda_A100/
│   └── ...
├── dashboard/                # Generated dashboard HTML
├── site/                     # GitHub Pages source
├── scripts/
│   └── publish.sh            # Copies JSON to results/, rebuilds site
└── README.md
```

**Why separate?**

- Main repo stays lean — no benchmark data in git history
- Results grow indefinitely without affecting clone speed
- The benchmark site can deploy independently
- Clear separation: library code vs performance tracking

**Workflow:**

1. Run benchmarks locally or on a remote GPU (see below)
2. Copy the versioned JSON (`v{version}_{device}_{timestamp}.json`) to the benchmark repo
3. Push to the benchmark repo — site auto-deploys

### Remote GPU Benchmarking (vast.ai)

If you don't have a suitable GPU locally, rent one on-demand from [vast.ai](https://vast.ai):

1. **Create a vast.ai account** and add credits (~$5 is enough for many runs)

2. **Rent an instance** — search for an A100 or A6000 (~$0.20/hr), select a PyTorch template

3. **SSH into the instance** (vast.ai provides the command, different each time):
    ```bash
    ssh -p <port> root@<host>
    ```

4. **Set up and run benchmarks on the instance:**
    ```bash
    git clone https://github.com/soran-ghaderi/torchebm.git
    cd torchebm && git checkout <your-branch>
    pip install -e ".[dev]"
    bash benchmarks/run.sh                    # full run
    bash benchmarks/run.sh --module losses    # or targeted
    ```

5. **Download results to your local machine** (from a separate terminal):
    ```bash
    scp -P <port> root@<host>:torchebm/benchmarks/results/v*.json benchmarks/results/
    ```

6. **Destroy the instance** — you're done

**Cost:** ~$0.10–0.50 per benchmark run. Each instance is ephemeral — there's no persistent state between runs.


## Performance Tips and Best Practices

<div class="grid" markdown>
<div markdown>

### General Tips

1. **Use the right device**: Always move computation to GPU when available
2. **Batch processing**: Process data in batches rather than individually
3. **Reuse tensors**: Avoid creating new tensors in inner loops
4. **Monitor memory**: Use `torch.cuda.memory_summary()` to track memory usage

</div>
<div markdown>

### Sampling Tips

1. **Tune step sizes**: Optimal step sizes balance exploration and stability
2. **Parallel chains**: Use multiple chains to improve sample diversity
3. **Adaptive methods**: Use adaptive samplers for complex distributions
4. **Mixed precision**: Consider using mixed precision for larger models

</div>
</div>

!!! warning "Common Pitfalls"
    Avoid these common performance issues:

    1. **Unnecessary CPU-GPU transfers**: Keep data on the same device
    2. **Small batch sizes**: Too small batches underutilize hardware
    3. **Unneeded gradient tracking**: Disable gradients when not training
    4. **Excessive logging**: Logging every step can significantly slow down sampling

## Algorithm-Specific Optimizations

### Langevin Dynamics

```python
# Optimize step size for Langevin dynamics
# Rule of thumb: step_size ≈ O(d^(-1/3)) where d is dimension
step_size = min(0.01, 0.1 * dim**(-1/3))

# Noise scale should be sqrt(2 * step_size) for standard Langevin
noise_scale = np.sqrt(2 * step_size)
```

### Hamiltonian Monte Carlo

```python
# Optimize HMC parameters
# Leapfrog steps should scale with dimension
n_leapfrog_steps = max(5, int(np.sqrt(dim)))

# Step size should decrease with dimension
step_size = min(0.01, 0.05 * dim**(-1/4))
```

## Multi-GPU Scaling (Planned)

## Conclusion

Performance optimization in TorchEBM involves careful attention to vectorization, GPU acceleration, memory management, and algorithm-specific tuning. By following these guidelines, you can achieve significant speedups in your energy-based modeling workflows. 
>>>>>>> ec7b9371 (perf: Implement lazy loading for subpackages and remove unused image and visualization utilities)
