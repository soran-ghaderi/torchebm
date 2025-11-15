#!/usr/bin/env python3
"""
Benchmark script to demonstrate performance improvements.

This script can be run to see the performance of key operations:
- GaussianModel forward pass
- BaseModel gradient computation
- Langevin Dynamics sampling
- HMC sampling
"""
import time
import torch
from torchebm.core.base_model import GaussianModel, DoubleWellModel
from torchebm.samplers import LangevinDynamics, HamiltonianMonteCarlo


def benchmark_gaussian_forward():
    """Benchmark GaussianModel forward pass."""
    print("=" * 70)
    print("Benchmarking GaussianModel.forward()")
    print("=" * 70)
    
    device = "cpu"
    dim = 100
    batch_size = 1000
    n_iterations = 200
    
    mean = torch.zeros(dim, device=device)
    cov = torch.eye(dim, device=device)
    model = GaussianModel(mean=mean, cov=cov).to(device)
    
    x = torch.randn(batch_size, dim, device=device)
    
    # Warmup
    for _ in range(10):
        _ = model(x)
    
    # Benchmark
    start = time.time()
    for _ in range(n_iterations):
        energy = model(x)
    elapsed = time.time() - start
    
    print(f"  Dimensions: {dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Time per iteration: {elapsed/n_iterations*1000:.2f}ms")
    print(f"  Throughput: {n_iterations*batch_size/elapsed:.0f} samples/sec")
    print()


def benchmark_gradient_computation():
    """Benchmark BaseModel gradient computation."""
    print("=" * 70)
    print("Benchmarking BaseModel.gradient()")
    print("=" * 70)
    
    device = "cpu"
    dim = 50
    batch_size = 100
    n_iterations = 100
    
    model = DoubleWellModel(barrier_height=2.0).to(device)
    x = torch.randn(batch_size, dim, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(5):
        _ = model.gradient(x)
    
    # Benchmark
    start = time.time()
    for _ in range(n_iterations):
        grad = model.gradient(x)
    elapsed = time.time() - start
    
    print(f"  Dimensions: {dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Time per iteration: {elapsed/n_iterations*1000:.2f}ms")
    print(f"  Throughput: {n_iterations*batch_size/elapsed:.0f} samples/sec")
    print()


def benchmark_langevin_sampling():
    """Benchmark LangevinDynamics sampling."""
    print("=" * 70)
    print("Benchmarking LangevinDynamics.sample()")
    print("=" * 70)
    
    device = "cpu"
    dim = 20
    n_samples = 100
    n_steps = 200
    
    mean = torch.zeros(dim, device=device)
    cov = torch.eye(dim, device=device)
    model = GaussianModel(mean=mean, cov=cov).to(device)
    
    sampler = LangevinDynamics(
        model=model,
        step_size=0.01,
        device=device
    )
    
    # Warmup
    _ = sampler.sample(dim=dim, n_steps=10, n_samples=10)
    
    # Benchmark
    start = time.time()
    samples = sampler.sample(dim=dim, n_steps=n_steps, n_samples=n_samples)
    elapsed = time.time() - start
    
    print(f"  Dimensions: {dim}")
    print(f"  Samples: {n_samples}")
    print(f"  Steps: {n_steps}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Time per step: {elapsed/n_steps*1000:.2f}ms")
    print(f"  Samples generated: {n_samples * n_steps}")
    print(f"  Throughput: {n_samples*n_steps/elapsed:.0f} samples/sec")
    print()


def benchmark_hmc_sampling():
    """Benchmark HamiltonianMonteCarlo sampling."""
    print("=" * 70)
    print("Benchmarking HamiltonianMonteCarlo.sample()")
    print("=" * 70)
    
    device = "cpu"
    dim = 20
    n_samples = 50
    n_steps = 100
    n_leapfrog = 10
    
    mean = torch.zeros(dim, device=device)
    cov = torch.eye(dim, device=device)
    model = GaussianModel(mean=mean, cov=cov).to(device)
    
    sampler = HamiltonianMonteCarlo(
        model=model,
        step_size=0.1,
        n_leapfrog_steps=n_leapfrog,
        device=device
    )
    
    # Warmup
    _ = sampler.sample(dim=dim, n_steps=5, n_samples=5)
    
    # Benchmark
    start = time.time()
    samples = sampler.sample(dim=dim, n_steps=n_steps, n_samples=n_samples)
    elapsed = time.time() - start
    
    print(f"  Dimensions: {dim}")
    print(f"  Samples: {n_samples}")
    print(f"  Steps: {n_steps}")
    print(f"  Leapfrog steps: {n_leapfrog}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Time per step: {elapsed/n_steps*1000:.2f}ms")
    print(f"  Total gradient calls: {n_steps * n_leapfrog * 2}")
    print(f"  Throughput: {n_samples*n_steps/elapsed:.0f} samples/sec")
    print()


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("TORCHEBM PERFORMANCE BENCHMARKS")
    print("=" * 70)
    print()
    
    benchmark_gaussian_forward()
    benchmark_gradient_computation()
    benchmark_langevin_sampling()
    benchmark_hmc_sampling()
    
    print("=" * 70)
    print("BENCHMARKS COMPLETE")
    print("=" * 70)
    print()
    print("These optimizations include:")
    print("  - Removed unnecessary dtype conversions in gradient computation")
    print("  - Replaced expand+bmm with efficient einsum in GaussianModel")
    print("  - Optimized diagnostics using broadcasting instead of expand")
    print("  - Cached repeated computations in integrators")
    print("  - Removed redundant device/dtype conversions")
    print()


if __name__ == "__main__":
    main()
