---
sidebar_position: 3
title: Samplers
description: Understanding and using sampling algorithms in TorchEBM
---

# Sampling Algorithms

Sampling from energy-based models is a core task in TorchEBM. This guide explains the different sampling algorithms available and how to use them effectively.

## Overview of Sampling

In energy-based models, we need to sample from the probability distribution defined by the energy function:

$$p(x) = \frac{e^{-E(x)}}{Z}$$

Since the normalizing constant Z is typically intractable, we use Markov Chain Monte Carlo (MCMC) methods to generate samples without needing to compute Z.

## Available Samplers

### Langevin Dynamics

Langevin Dynamics is a gradient-based MCMC method that updates samples using the energy gradient plus Gaussian noise:

```python
from torchebm.samplers.langevin_dynamics import LangevinDynamics
from torchebm.core import GaussianEnergy
import torch

# Create an energy function
energy_fn = GaussianEnergy(
    mean=torch.zeros(10),
    cov=torch.eye(10)
)

# Create a Langevin dynamics sampler
langevin_sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.01,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Generate samples
samples = langevin_sampler.sample_chain(
    dim=10,
    n_steps=1000,
    n_samples=100,
    return_trajectory=False
)
```

#### Parameters

- `energy_function`: The energy function to sample from
- `step_size`: Step size for gradient updates (controls exploration vs. stability)
- `noise_scale`: Scale of the noise (default is sqrt(2*step_size))
- `device`: The device to perform sampling on (e.g., "cuda" or "cpu")

### Hamiltonian Monte Carlo (HMC)

HMC uses Hamiltonian dynamics to make more efficient proposals, leading to better exploration of the distribution:

```python
from torchebm.samplers.hmc import HamiltonianMonteCarlo
from torchebm.core import DoubleWellEnergy
import torch

# Create an energy function
energy_fn = DoubleWellEnergy()

# Create an HMC sampler
hmc_sampler = HamiltonianMonteCarlo(
    energy_function=energy_fn,
    step_size=0.1,
    n_leapfrog_steps=10,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Generate samples
samples = hmc_sampler.sample_chain(
    dim=2,
    n_steps=500,
    n_samples=100,
    return_trajectory=False
)
```

#### Parameters

- `energy_function`: The energy function to sample from
- `step_size`: Step size for leapfrog integration
- `n_leapfrog_steps`: Number of leapfrog steps per iteration
- `device`: The device to perform sampling on

## Advanced Sampling Usage

### Tracking Sampling Progress

You can track diagnostics during sampling by setting `return_diagnostics=True`:

```python
samples, diagnostics = sampler.sample_chain(
    dim=10,
    n_steps=1000,
    n_samples=100,
    return_trajectory=True,
    return_diagnostics=True
)

# Diagnostics shape: [n_steps, n_diagnostics, n_samples, dim]
# Includes: Mean, Variance, Energy, Acceptance rate (for HMC)
```

### Custom Initialization

You can start the sampling chain from a specific point:

```python
# Custom initialization
x_init = torch.randn(100, 10)  # [n_samples, dim]
samples = sampler.sample_chain(
    x=x_init,
    n_steps=1000,
    return_trajectory=False
)
```

### Burn-in and Thinning

For better samples, you can implement burn-in and thinning:

```python
# Perform burn-in and thinning manually
samples, trajectory = sampler.sample_chain(
    dim=10,
    n_steps=2000,
    n_samples=100,
    return_trajectory=True
)

# Discard the first 1000 steps (burn-in)
# Keep every 10th sample (thinning)
thinned_samples = trajectory[:, 1000::10, :]
```

## Choosing a Sampler

- **Langevin Dynamics**: Good for general-purpose sampling, especially in high dimensions
- **Hamiltonian Monte Carlo**: Better exploration of complex energy landscapes, but more expensive per step

## Sampler Performance Tips

1. **Adjust step size**: Too large → unstable sampling; too small → slow mixing
2. **Use GPU acceleration**: For large batches of samples or high dimensions
3. **Monitor acceptance rates**: For HMC, aim for 60-90% acceptance rate
4. **Check sample quality**: Correlation between successive samples should be low
5. **Burn-in**: Discard initial samples before the chain reaches its stationary distribution

## Implementing Custom Samplers

You can create custom samplers by subclassing `BaseSampler`:

```python
from torchebm.core import BaseSampler
import torch

class MyCustomSampler(BaseSampler):
    def __init__(self, energy_function, param1, param2, device="cpu"):
        super().__init__(energy_function, device)
        self.param1 = param1
        self.param2 = param2
    
    def step(self, x, step_idx=None):
        # Implement your sampling step here
        # x shape: [n_samples, dim]
        
        # Example: simple random walk
        noise = torch.randn_like(x) * self.param1
        x_new = x + noise
        
        # Return updated samples and any diagnostics
        return x_new, {"diagnostic1": value1, "diagnostic2": value2}
``` 