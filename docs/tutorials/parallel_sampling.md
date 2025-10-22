---
sidebar_position: 5
title: Parallel Sampling
description: Guide to efficient parallel sampling with TorchEBM
icon: material/vector-polyline-plus
---

# Parallel Sampling

This guide explains how to efficiently sample from models in parallel using TorchEBM.

## Overview

Parallel sampling allows you to generate multiple samples simultaneously, leveraging modern hardware like GPUs for significant speedups. TorchEBM is designed for efficient parallel sampling, making it easy to generate thousands or even millions of samples with minimal code.

## Basic Parallel Sampling

The simplest way to perform parallel sampling is to initialize multiple chains and let TorchEBM handle the parallelization:

```python
import torch
from torchebm.core import BaseModel
from torchebm.samplers import LangevinDynamics
import torch.nn as nn

class MLPModel(BaseModel):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPModel(input_dim=2, hidden_dim=32).to(device)

sampler = LangevinDynamics(
    model=model,
    step_size=0.1,
    noise_scale=0.01,
    device=device
)

n_samples = 10000
dim = 2
initial_points = torch.randn(n_samples, dim, device=device)

samples = sampler.sample(
    x=initial_points,
    n_steps=1000,
    return_trajectory=False
)

print(f"Generated {samples.shape[0]} samples of dimension {samples.shape[1]}")
```

## GPU Acceleration

For maximum performance, TorchEBM leverages GPU acceleration when available. This provides dramatic speedups for parallel sampling:

```python
import time
import torch
from torchebm.core import DoubleWellModel
from torchebm.samplers import LangevinDynamics

model = DoubleWellModel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

sampler = LangevinDynamics(
    model=model,
    step_size=0.01,
    device=device
)

n_samples = 50000
dim = 2

initial_points = torch.randn(n_samples, dim, device=device)

start_time = time.time()
samples = sampler.sample(
    x=initial_points,
    n_steps=1000,
    return_trajectory=False
)
end_time = time.time()

print(f"Generated {n_samples} samples in {end_time - start_time:.2f} seconds")
print(f"Average time per sample: {(end_time - start_time) / n_samples * 1000:.4f} ms")
```

## Batch Processing for Large Sample Sets

When generating a very large number of samples, you might need to process them in batches to avoid memory issues:

```python
import torch
import numpy as np
from torchebm.core import BaseModel
from torchebm.samplers import LangevinDynamics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPModel(input_dim=2, hidden_dim=64).to(device)
sampler = LangevinDynamics(
    model=model,
    step_size=0.01,
    device=device
)

total_samples = 1000000
dim = 2
batch_size = 10000
num_batches = total_samples // batch_size

all_samples = np.zeros((total_samples, dim))

for i in range(num_batches):
    print(f"Generating batch {i+1}/{num_batches}")

    initial_points = torch.randn(batch_size, dim, device=device)

    batch_samples = sampler.sample(
        x=initial_points,
        n_steps=1000,
        return_trajectory=False
    )

    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    all_samples[start_idx:end_idx] = batch_samples.cpu().numpy()

print(f"Generated {total_samples} samples in total")
```

## Multi-GPU Sampling

For even larger-scale sampling, you can distribute the workload across multiple GPUs:

```python
import torch
import torch.multiprocessing as mp
from torchebm.core import DoubleWellModel
from torchebm.samplers import LangevinDynamics

def sample_on_device(rank, n_samples, n_steps, result_queue):
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    model = DoubleWellModel().to(device)
    sampler = LangevinDynamics(
        model=model,
        step_size=0.01,
        device=device
    )

    initial_points = torch.randn(n_samples, 2, device=device)
    samples = sampler.sample(
        x=initial_points,
        n_steps=n_steps,
        return_trajectory=False
    )

    result_queue.put(samples.cpu())

def main():
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("No GPUs available, using CPU")
        n_gpus = 1

    print(f"Using {n_gpus} device(s) for sampling")

    total_samples = 100000
    samples_per_device = total_samples // n_gpus
    n_steps = 1000

    result_queue = mp.Queue()

    processes = []
    for rank in range(n_gpus):
        p = mp.Process(
            target=sample_on_device,
            args=(rank, samples_per_device, n_steps, result_queue)
        )
        p.start()
        processes.append(p)

    all_samples = []
    for _ in range(n_gpus):
        all_samples.append(result_queue.get())

    for p in processes:
        p.join()

    all_samples = torch.cat(all_samples, dim=0)
    print(f"Generated {all_samples.shape[0]} samples")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
```

## Performance Tips for Parallel Sampling

1. **Use the correct device**: Always specify the device when creating samplers to ensure proper hardware acceleration.

2. **Batch size tuning**: Find the optimal batch size for your hardware. Too small wastes parallelism, too large may cause memory issues.

3. **Data type optimization**: Consider using `torch.float16` (half precision) for even faster sampling on compatible GPUs:

```python
initial_points = torch.randn(10000, 2, device=device, dtype=torch.float16)
model = model.half()
sampler = LangevinDynamics(
    model=model,
    step_size=0.01,
    device=device
)
samples = sampler.sample(x=initial_points, n_steps=1000)
```

4. **Minimize data transfers**: Keep data on the GPU as much as possible. CPU-GPU transfers are slow.

5. **Pre-allocate memory**: For repetitive sampling, reuse the same tensor to avoid repeated allocations.

## Conclusion

Parallel sampling in TorchEBM allows you to efficiently generate large numbers of samples from your energy-based models. By leveraging GPU acceleration and batch processing, you can significantly speed up sampling, enabling more efficient model evaluation and complex applications.

Whether you're generating samples for visualization, evaluation, or downstream tasks, TorchEBM's parallel sampling capabilities provide the performance and scalability you need. 