---
sidebar_position: 8
title: Performance Optimization
description: Techniques for optimizing performance in TorchEBM
icon: material/speedometer
---

# Performance Optimization

This document provides guidance on optimizing the performance of TorchEBM for both development and usage.

## Performance Considerations

!!! tip "Key Performance Areas"
    When working with TorchEBM, pay special attention to these performance-critical areas:

    1. **Sampling algorithms**: These are iterative and typically the most compute-intensive
    2. **Gradient calculations**: Computing energy gradients is fundamental to many algorithms
    3. **Batch processing**: Effective vectorization for parallel processing
    4. **GPU utilization**: Proper device management and memory usage

## Vectorization Techniques

<div class="grid" markdown>
<div markdown>

### Batched Operations

TorchEBM extensively uses batching to improve performance:

```python
# Instead of looping over samples
for i in range(n_samples):
    energy_i = energy_function(x[i])  # Slow

# Use batched computation
energy = energy_function(x)  # Fast
```

</div>
<div markdown>

### Parallel Sampling

Sample multiple chains in parallel by using batch dimensions:

```python
# Initialize batch of samples
x = torch.randn(n_samples, dim, device=device)

# One sampling step (all chains update together)
x_new, _ = sampler.step(x)
```

</div>
</div>

## GPU Acceleration

TorchEBM is designed to work efficiently on GPUs:

### Device Management

```python
# Create energy function and move to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
energy_fn = GaussianEnergy(mean, cov).to(device)

# Create sampler with the same device
sampler = LangevinDynamics(energy_fn, device=device)

# Generate samples (automatically on the correct device)
samples, _ = sampler.sample(dim=2, n_steps=1000, n_samples=10000)
```

### Memory Management

Memory management is critical for performance, especially on GPUs:

```python
# Avoid creating new tensors in loops
for step in range(n_steps):
    # Bad: Creates new tensors each iteration
    x = x - step_size * energy_fn.gradient(x) + noise_scale * torch.randn_like(x)
    
    # Good: In-place operations
    grad = energy_fn.gradient(x)
    x.sub_(step_size * grad)
    x.add_(noise_scale * torch.randn_like(x))
```

## Custom CUDA Kernels (to be Added--See Also: [cuRBLAS](https://github.com/soran-ghaderi/cuRBLAS))

## Sampling Efficiency

Sampling efficiency can be improved using several techniques:

<div class="grid cards" markdown>

-   :material-speedometer:{ .lg .middle } __Step Size Adaptation__

    ---

    Automatically adjust step sizes based on acceptance rates or other metrics.

    ```python
    if acceptance_rate < 0.3:
        step_size *= 0.9  # Decrease step size
    elif acceptance_rate > 0.7:
        step_size *= 1.1  # Increase step size
    ```

-   :material-fire:{ .lg .middle } __Burn-in Period__

    ---

    Discard initial samples to reduce the impact of initialization.

    ```python
    # Run burn-in period
    x = torch.randn(n_samples, dim)
    for _ in range(burn_in_steps):
        x, _ = sampler.step(x)
        
    # Start collecting samples
    samples = []
    for _ in range(n_steps):
        x, _ = sampler.step(x)
        samples.append(x.clone())
    ```

-   :material-selection-drag:{ .lg .middle } __Thinning__

    ---

    Reduce correlation between samples by keeping only every Nth sample.

    ```python
    # Collect samples with thinning
    samples = []
    for i in range(n_steps):
        x, _ = sampler.step(x)
        if i % thinning == 0:
            samples.append(x.clone())
    ```

- :material-chart-bell-curve:{ .lg .middle } __Warm Starting__

    ---

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

Results are stored in `benchmarks/results/` as JSON files, with a versioned naming scheme (`v{version}_{device}_{timestamp}.json`) and an interactive dashboard at `benchmarks/results/dashboard.html`.


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