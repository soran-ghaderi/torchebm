---
sidebar_position: 7
title: Parallel Sampling
description: A guide to efficient parallel sampling with TorchEBM.
---

# Parallel Sampling

TorchEBM is designed for efficient parallel sampling, allowing you to generate thousands or even millions of samples simultaneously by leveraging modern hardware like GPUs.

## Batch Sampling

The key to parallel sampling is to provide a batch of initial points to the sampler. Each point in the batch is treated as an independent MCMC chain, and all chains are updated in parallel.

This is highly efficient on GPUs, where the vectorized operations can be processed simultaneously.

### Example

Here's how to generate 10,000 samples in parallel.

```python
import torch
import torch.nn as nn
import time
from torchebm.core import BaseModel
from torchebm.samplers import LangevinDynamics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPModel(BaseModel):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.network(x).squeeze(-1)

model = MLPModel().to(device)

sampler = LangevinDynamics(
    model=model,
    step_size=0.1
)

n_samples = 10000
dim = 2
initial_points = torch.randn(n_samples, dim, device=device)

start_time = time.time()
samples = sampler.sample(
    x=initial_points,
    n_steps=1000
)
end_time = time.time()

print(f"Generated {samples.shape[0]} samples in {end_time - start_time:.2f} seconds on {device}.")
```

## Tips for Efficient Parallel Sampling

-   **Use a GPU**: For significant speedups, always run parallel sampling on a CUDA-enabled GPU.
-   **Batch Size**: Experiment with the number of samples (`n_samples`) to find the optimal batch size for your hardware. Larger batches can lead to better hardware utilization, but also increase memory usage.
-   **Minimize Data Transfers**: Keep your model and data on the same device to avoid costly CPU-GPU memory transfers.
-   **Use Half Precision**: For GPUs that support it, using `dtype=torch.float16` for your model and initial points can provide a significant speed boost. 