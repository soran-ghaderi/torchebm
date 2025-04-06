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
from torchebm.core import GaussianEnergy
from torchebm.samplers.langevin_dynamics import LangevinDynamics

# Create energy function for a 2D Gaussian
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dim = 2  # dimension of the state space
n_steps = 100  # steps between samples
n_samples = 1000  # num of samples
mean = torch.tensor([1.0, -1.0])
cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])
energy_fn = GaussianEnergy(mean, cov, device=device)

# Initialize sampler
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.01,
    noise_scale=0.1,
    device=device,
)

# Generate samples
initial_state = torch.zeros(n_samples, dim, device=device)
samples = sampler.sample_chain(
    x=initial_state,
    n_steps=n_steps,
    n_samples=n_samples,
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
from torchebm.core import DoubleWellEnergy

# Create energy function and sampler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
energy_fn = DoubleWellEnergy(barrier_height=2.0)
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.001,
    noise_scale=0.1,
    decay=0.1,  # for stability
    device=device,
)

# Generate trajectory with diagnostics
initial_state = torch.tensor([0.0], device=device)
trajectory, diagnostics = sampler.sample(
    x=initial_state,
    n_steps=1000,
    return_trajectory=True,
    return_diagnostics=True,
)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot trajectory
ax1.plot(trajectory[0, :, 0].cpu().numpy())
ax1.set_title("Single Chain Trajectory")
ax1.set_xlabel("Step")
ax1.set_ylabel("Position")

# Plot energy over time
ax2.plot(diagnostics[:, 2, 0, 0].cpu().numpy())
ax2.set_title("Energy Evolution")
ax2.set_xlabel("Step")
ax2.set_ylabel("Energy")

plt.tight_layout()
plt.show()
```

## Key Benefits of TorchEBM's Langevin Dynamics Implementation

1. **GPU Acceleration** - Sampling is performed efficiently on GPUs when available
2. **Flexible API** - Easy to use with various energy functions and initialization strategies
3. **Diagnostic Tools** - Track energy, gradient norms, and acceptance rates during sampling
4. **Configurable Parameters** - Fine-tune step size, noise scale, and decay for optimal performance

## Conclusion

Langevin dynamics is a versatile sampling method for energy-based models, and TorchEBM makes it easy to use in your projects. Whether you're sampling from simple analytical distributions or complex neural network energy functions, the same API works seamlessly.

Stay tuned for more tutorials on other samplers and energy functions!
```

```