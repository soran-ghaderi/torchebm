---
sidebar_position: 8
title: Performance Optimization
description: Techniques for optimizing performance in TorchEBM
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
samples, _ = sampler.sample_chain(dim=2, n_steps=1000, n_samples=10000)
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

## Custom CUDA Kernels

For the most performance-critical operations, TorchEBM provides custom CUDA kernels:

```python
# Standard PyTorch implementation
def langevin_step_pytorch(x, energy_fn, step_size, noise_scale):
    grad = energy_fn.gradient(x)
    noise = torch.randn_like(x) * noise_scale
    return x - step_size * grad + noise

# Using custom CUDA kernel when available
from torchebm.cuda import langevin_step_cuda

def langevin_step(x, energy_fn, step_size, noise_scale):
    if x.is_cuda and torch.cuda.is_available():
        return langevin_step_cuda(x, energy_fn, step_size, noise_scale)
    else:
        return langevin_step_pytorch(x, energy_fn, step_size, noise_scale)
```

## Sampling Efficiency

Sampling efficiency can be improved using several techniques:

<div class="grid cards" markdown>

-   :material-speedometer:{ .lg .middle } __Step Size Adaptation__

    ---

    Automatically adjust step sizes based on acceptance rates or other metrics.

    ```python
    # Adaptive step size example
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

-   :material-chart-bell-curve:{ .lg .middle } __Warm Starting__

    ---

    Initialize sampling from a distribution close to the target.

    ```python
    # Warm start from approximate distribution
    x = approximate_sampler.sample(n_samples, dim)
    samples = sampler.sample_chain(
        n_steps=n_steps, 
        initial_samples=x
    )
    ```

</div>

## Profiling and Benchmarking

To identify performance bottlenecks, TorchEBM includes profiling utilities:

```python
from torchebm.utils.profiling import profile_sampling

# Profile a sampling run
profiling_results = profile_sampling(
    sampler, 
    dim=10, 
    n_steps=1000, 
    n_samples=100
)

# Print results
print(f"Total time: {profiling_results['total_time']:.2f} seconds")
print(f"Time per step: {profiling_results['time_per_step']:.5f} seconds")
print("Component breakdown:")
for component, time_pct in profiling_results['component_times'].items():
    print(f"  {component}: {time_pct:.1f}%")
```

## Performance Benchmarks

Here are some performance benchmarks for common operations:

| Operation | CPU Time (ms) | GPU Time (ms) | Speedup |
|-----------|---------------|---------------|---------|
| Langevin step (1,000 samples, dim=10) | 8.2 | 0.41 | 20x |
| HMC step (1,000 samples, dim=10) | 15.4 | 0.76 | 20.3x |
| Energy gradient (10,000 samples, dim=100) | 42.1 | 1.8 | 23.4x |
| Full sampling (10,000 samples, 100 steps) | 820 | 38 | 21.6x |

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

## Multi-GPU Scaling

For extremely large sampling tasks, TorchEBM supports multi-GPU execution:

```python
# Distribution across GPUs using DataParallel
import torch.nn as nn

class ParallelSampler(nn.DataParallel):
    def __init__(self, sampler, device_ids=None):
        super().__init__(sampler, device_ids=device_ids)
        self.module = sampler
        
    def sample_chain(self, dim, n_steps, n_samples):
        # Distribute samples across GPUs
        return self.forward(dim, n_steps, n_samples)
        
# Create parallel sampler
devices = list(range(torch.cuda.device_count()))
parallel_sampler = ParallelSampler(sampler, device_ids=devices)

# Generate samples using all available GPUs
samples = parallel_sampler.sample_chain(dim=100, n_steps=1000, n_samples=100000)
```

## Conclusion

Performance optimization in TorchEBM involves careful attention to vectorization, GPU acceleration, memory management, and algorithm-specific tuning. By following these guidelines, you can achieve significant speedups in your energy-based modeling workflows. 