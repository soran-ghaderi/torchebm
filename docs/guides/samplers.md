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

## Langevin Dynamics

Langevin Dynamics is a gradient-based MCMC method that updates samples using the energy gradient plus Gaussian noise. It's one of the most commonly used samplers in energy-based models due to its simplicity and effectiveness.

### Basic Usage

```python
import torch
from torchebm.core import BaseEnergyFunction
from torchebm.samplers import LangevinDynamics
import torch.nn as nn

# Define a custom energy function
class MLPEnergy(BaseEnergyFunction):
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

# Create an energy function
energy_fn = MLPEnergy(input_dim=2, hidden_dim=32)
        
# Create a Langevin dynamics sampler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
langevin_sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.1,
    noise_scale=0.01,
    device=device
)

# Generate samples
initial_points = torch.randn(100, 2, device=device)  # 100 samples of dimension 2
samples = langevin_sampler.sample(
    x=initial_points,
    n_steps=1000,
    return_trajectory=False
)

print(samples.shape)  # Shape: [100, 2]
```

### Parameters

- `energy_function`: The energy function to sample from
- `step_size`: Step size for gradient updates (controls exploration vs. stability)
- `noise_scale`: Scale of the noise (default is sqrt(2*step_size))
- `device`: The device to perform sampling on (e.g., "cuda" or "cpu")

### Advanced Features

The `LangevinDynamics` sampler in TorchEBM comes with several advanced features:

#### Burn-in and Thinning

You can automatically discard the initial samples (burn-in) and keep only every n-th sample (thinning):

```python
samples = langevin_sampler.sample(
    x=initial_points,
    n_steps=2000,
    burn_in=1000,  # Discard the first 1000 steps
    thinning=10,   # Keep every 10th sample after burn-in
    return_trajectory=False
)
```

#### Returning Trajectories

For visualization or analysis, you can get the full trajectory of the sampling process:

```python
trajectory = langevin_sampler.sample(
    x=initial_points,
    n_steps=1000,
    return_trajectory=True
)

print(trajectory.shape)  # Shape: [n_samples, n_steps, dim]
```

#### Dynamic Parameter Scheduling

TorchEBM allows you to dynamically adjust the step size and noise scale during sampling using schedulers:

```python
from torchebm.core import CosineScheduler, LinearScheduler, ExponentialDecayScheduler

# Create schedulers
step_size_scheduler = CosineScheduler(
    start_value=3e-2,
    end_value=5e-3,
    n_steps=100
)

noise_scheduler = CosineScheduler(
    start_value=3e-1,
    end_value=1e-2,
    n_steps=100
)

# Create sampler with schedulers
dynamic_sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=step_size_scheduler,
    noise_scale=noise_scheduler,
    device=device
)
```

## Hamiltonian Monte Carlo (HMC)

HMC uses Hamiltonian dynamics to make more efficient proposals, leading to better exploration of the distribution:

```python
from torchebm.samplers import HamiltonianMonteCarlo
from torchebm.core import DoubleWellEnergy

# Create an energy function
energy_fn = DoubleWellEnergy()

# Create an HMC sampler
hmc_sampler = HamiltonianMonteCarlo(
    energy_function=energy_fn,
    step_size=0.1,
    n_leapfrog_steps=10,
    device=device
)

# Generate samples
samples = hmc_sampler.sample(
    x=torch.randn(100, 2, device=device),
    n_steps=500,
    return_trajectory=False
)
```

## Integration with Loss Functions

Samplers in TorchEBM are designed to work seamlessly with loss functions for training energy-based models:

```python
from torchebm.losses import ContrastiveDivergence

# Create a loss function that uses the sampler internally
loss_fn = ContrastiveDivergence(
    energy_function=energy_fn,
    sampler=langevin_sampler,
    k_steps=10,
    persistent=True,
    buffer_size=1024
)

# During training, the loss function will use the sampler to generate negative samples
optimizer.zero_grad()
loss, negative_samples = loss_fn(data_batch)
loss.backward()
optimizer.step()
```

## Parallel Sampling

TorchEBM supports parallel sampling to speed up the generation of multiple samples:

```python
# Generate multiple chains in parallel
n_samples = 1000
dim = 2
initial_points = torch.randn(n_samples, dim, device=device)

# All chains are processed in parallel on the GPU
samples = langevin_sampler.sample(
    x=initial_points,
    n_steps=1000,
    return_trajectory=False
)
```

## Sampler Visualizations

Visualizing the sampling process can help understand the behavior of your model. Here's an example showing how to visualize Langevin Dynamics trajectories:

```python
import numpy as np
import matplotlib.pyplot as plt
from torchebm.core import DoubleWellEnergy
from torchebm.samplers import LangevinDynamics

# Create energy function and sampler
energy_fn = DoubleWellEnergy(barrier_height=2.0)
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.01
)

# Initial point
initial_point = torch.tensor([[-2.0, 0.0]], dtype=torch.float32)

# Run sampling and get trajectory
trajectory = sampler.sample(
    x=initial_point,
    n_steps=1000,
    return_trajectory=True
)

# Background energy landscape
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32).unsqueeze(0)
        Z[i, j] = energy_fn(point).item()

# Visualize
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.7)
plt.colorbar(label='Energy')

# Extract trajectory coordinates
traj_x = trajectory[0, :, 0].numpy()
traj_y = trajectory[0, :, 1].numpy()

# Plot trajectory
plt.plot(traj_x, traj_y, 'r-', linewidth=1, alpha=0.7)
plt.scatter(traj_x[0], traj_y[0], c='black', s=50, marker='o', label='Start')
plt.scatter(traj_x[-1], traj_y[-1], c='red', s=50, marker='*', label='End')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Langevin Dynamics Trajectory')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('docs/assets/images/samplers/langevin_trajectory.png')
plt.show()
```

![Langevin Trajectory](../assets/images/samplers/langevin_trajectory.png)

## Choosing a Sampler

- **Langevin Dynamics**: Good for general-purpose sampling, especially with neural network energy functions
- **Hamiltonian Monte Carlo**: Better exploration of complex energy landscapes, but more computationally expensive
- **Metropolis-Adjusted Langevin Algorithm (MALA)**: Similar to Langevin Dynamics but with an accept/reject step

## Performance Tips

1. **Use GPU acceleration**: Batch processing of samples on GPU can significantly speed up sampling
2. **Adjust step size**: Too large → unstable sampling; too small → slow mixing
3. **Dynamic scheduling**: Use parameter schedulers to automatically adjust step size and noise during sampling
4. **Monitor energy values**: Track energy values to ensure proper mixing and convergence
5. **Burn-in and thinning**: Use appropriate burn-in and thinning to improve sample quality
6. **Multiple chains**: Run multiple chains from different starting points to better explore the distribution

## Custom Samplers

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
        # Implement a single sampling step
        # x shape: [n_samples, dim]
        
        # Example: simple random walk with energy gradient
        grad = self.energy_function.grad(x)
        noise = torch.randn_like(x) * self.param1
        x_new = x - self.param2 * grad + noise
        
        return x_new
    
    def sample(self, x=None, n_steps=1000, n_samples=None, dim=None, 
               return_trajectory=False, burn_in=0, thinning=1):
        # You can customize the sampling logic or use the implementation
        # from the parent class BaseSampler
        return super().sample(x, n_steps, n_samples, dim, 
                              return_trajectory, burn_in, thinning) 