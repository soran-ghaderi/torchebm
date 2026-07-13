---
title: Langevin Dynamics Sampling with TorchEBM
date: 2024-04-05
authors:
  - soran-ghaderi
categories:
  - Tutorials
  - Examples
tags:
  - langevin
  - sampling
  - tutorial
readtime: 10
comments: true
---

# Langevin Dynamics Sampling with TorchEBM

Langevin dynamics is a powerful sampling technique that allows us to draw samples from complex probability distributions. In this tutorial, we'll explore how to use TorchEBM's implementation of Langevin dynamics for sampling from various energy landscapes.

<!-- more -->

## Basic Example: Sampling from a 2D Gaussian

Let's start with a simple example of sampling from a 2D Gaussian distribution:

```python title="Basic Langevin Dynamics Sampling" linenums="1"
import torch
import matplotlib.pyplot as plt
from torchebm.core import GaussianModel
from torchebm.samplers import LangevinDynamics

# Create an energy model for a 2D Gaussian
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dim = 2  # dimension of the state space
n_steps = 100  # sampling steps
n_samples = 1000  # number of parallel chains
mean = torch.tensor([1.0, -1.0])
cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])
model = GaussianModel(mean, cov, device=device)

# Initialize sampler
sampler = LangevinDynamics(
    model=model,
    step_size=0.01,
    noise_scale=0.1,
    device=device,
)

# Generate samples
initial_state = torch.zeros(n_samples, dim, device=device)
samples = sampler.sample(
    x=initial_state,
    n_steps=n_steps,
)

# Plot results
samples = samples.cpu().numpy()
plt.figure(figsize=(10, 5))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1)
plt.title("Samples from 2D Gaussian using Langevin Dynamics")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.show()
```

## Advanced Example: Double Well Potential

For a more interesting example, let's sample from a double well potential, which has two local minima:

```python title="Double Well Energy Sampling" linenums="1"
from torchebm.core import DoubleWellModel

# Create energy model and sampler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DoubleWellModel(barrier_height=2.0)
sampler = LangevinDynamics(
    model=model,
    step_size=0.001,
    noise_scale=0.1,
    device=device,
)

# Generate a trajectory with diagnostics (dict of per-step tensors)
initial_state = torch.zeros(1, 2, device=device)
trajectory, diagnostics = sampler.sample(
    x=initial_state,
    n_steps=1000,
    return_trajectory=True,
    return_diagnostics=True,
)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot trajectory of the first chain's first coordinate
ax1.plot(trajectory[0, :, 0].cpu().numpy())
ax1.set_title("Single Chain Trajectory")
ax1.set_xlabel("Step")
ax1.set_ylabel("Position")

# Plot energy over time
ax2.plot(diagnostics["energy"].cpu().numpy())
ax2.set_title("Energy Evolution")
ax2.set_xlabel("Step")
ax2.set_ylabel("Energy")

plt.tight_layout()
plt.show()
```

## Key Benefits of TorchEBM's Langevin Dynamics Implementation

1. **GPU Acceleration** - Sampling is performed efficiently on GPUs when available; chains are a batch dimension
2. **Flexible API** - Easy to use with analytic models or custom neural energies
3. **Diagnostic Tools** - Track mean, variance, and energy during sampling via `return_diagnostics=True`
4. **Schedulable Parameters** - Step size and noise scale accept schedulers for annealed sampling

## Conclusion

Langevin dynamics is a versatile sampling method for energy-based models, and TorchEBM makes it easy to use in your projects. Whether you're sampling from simple analytical distributions or complex neural network energy functions, the same API works seamlessly.

For a runnable version, see the
[Langevin Dynamics 101 example](../../examples/10-sampling/01-mcmc/01-langevin-101.md);
the theory lives in [Sampling and Integration](../../concepts/sampling.md).