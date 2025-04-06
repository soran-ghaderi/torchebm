---
sidebar_position: 2
title: Getting Started
description: Get started with TorchEBM by installing it and running some basic examples
---

# Getting Started

This guide will help you get started with TorchEBM by walking you through the installation process and demonstrating some basic usage examples.

## Installation

TorchEBM can be installed directly from PyPI:

```bash
pip install torchebm
```

### Prerequisites

- Python 3.8 or newer
- PyTorch 1.10.0 or newer
- CUDA (optional, but recommended for performance)

### Installation from Source

If you wish to install the development version:

```bash
git clone https://github.com/soran-ghaderi/torchebm.git
cd torchebm
pip install -e .
```

## Quick Start

Here's a simple example to get you started with TorchEBM:

```python
import torch
from torchebm.core import GaussianEnergy
from torchebm.samplers.langevin_dynamics import LangevinDynamics

# Set device for computation
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define a 2D Gaussian energy function for visualization
energy_fn = GaussianEnergy(
    mean=torch.zeros(2, device=device),
    cov=torch.eye(2, device=device)
)

# Initialize Langevin dynamics sampler
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.01,
    device=device
).to(device)

# Generate 1000 samples
samples = sampler.sample_chain(
    dim=2,
    n_steps=100,
    n_samples=1000,
    return_trajectory=False
)

print(f"Generated {samples.shape[0]} samples of dimension {samples.shape[1]}")
```

## Next Steps

- Learn about [Energy Functions](./guides/energy_functions.md) available in TorchEBM
- Explore different [Sampling Algorithms](./guides/samplers.md)
- Try out the [Examples](./examples/index.md) for visualizations and advanced usage
- Check the [API Reference](./api/index.md) for detailed documentation

## Common Issues

### CUDA Out of Memory

If you encounter CUDA out of memory errors, try:
- Reducing the number of samples
- Reducing the dimension of the problem
- Switching to CPU if needed

### Support

If you encounter any issues or have questions:
- Check the [FAQ](./faq.md)
- Open an issue on [GitHub](https://github.com/soran-ghaderi/torchebm/issues)

