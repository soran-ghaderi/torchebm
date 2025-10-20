---
sidebar_position: 3
title: Samplers
description: Understanding and using sampling algorithms in TorchEBM.
---

# Sampling Algorithms

Sampling is a core task in TorchEBM. Since the normalizing constant \(Z\) in the Boltzmann distribution \(p(x) = \frac{e^{-E(x)}}{Z}\) is typically intractable, we use Markov Chain Monte Carlo (MCMC) methods to generate samples from the distribution defined by the model.

## Langevin Dynamics

Langevin Dynamics is a gradient-based MCMC method that is simple and effective for sampling from EBMs. It updates sample positions using the energy gradient plus some Gaussian noise.

### Basic Usage

```python
import torch
import torch.nn as nn
from torchebm.core import BaseModel
from torchebm.samplers import LangevinDynamics

class MLPModel(BaseModel):
    def __init__(self, input_dim=2, hidden_dim=64):
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
model = MLPModel().to(device)
        
langevin_sampler = LangevinDynamics(
    model=model,
    step_size=0.1,
    noise_scale=0.01
)

initial_points = torch.randn(100, 2, device=device)
samples = langevin_sampler.sample(
    x=initial_points,
    n_steps=1000
)

print(samples.shape)
```

### Key Parameters

- `model`: The EBM model to sample from.
- `step_size`: The step size for the gradient update. A smaller step size leads to more accurate integration but slower exploration.
- `noise_scale`: The scale of the Gaussian noise added at each step.

### Advanced Usage: Schedulers

You can dynamically adjust `step_size` and `noise_scale` during sampling using schedulers. This is useful for techniques like simulated annealing.

```python
from torchebm.core import CosineScheduler

step_size_scheduler = CosineScheduler(start_value=3e-2, end_value=5e-3, n_steps=1000)
noise_scheduler = CosineScheduler(start_value=3e-1, end_value=1e-2, n_steps=1000)

dynamic_sampler = LangevinDynamics(
    model=model,
    step_size=step_size_scheduler,
    noise_scale=noise_scheduler
)

samples = dynamic_sampler.sample(x=initial_points, n_steps=1000)
```

### Visualizing Sampler Trajectories

By setting `return_trajectory=True` in the `sample` method, you can obtain the full path of the sampler. This is useful for diagnosing sampler behavior and visualizing the energy landscape exploration.

![Langevin Trajectory](../assets/images/samplers/langevin_trajectory.png)

## Hamiltonian Monte Carlo (HMC)

HMC is a more advanced MCMC method that uses Hamiltonian dynamics to propose more efficient moves, often leading to faster exploration of the state space compared to Langevin Dynamics.

```python
from torchebm.samplers import HamiltonianMonteCarlo

hmc_sampler = HamiltonianMonteCarlo(
    model=model,
    step_size=0.1,
    n_leapfrog_steps=10
)

samples = hmc_sampler.sample(
    x=torch.randn(100, 2, device=device),
    n_steps=500
)
```

### Key Parameters

- `model`: The EBM model to sample from.
- `step_size`: The step size for the leapfrog integrator.
- `n_leapfrog_steps`: The number of leapfrog steps to perform for each proposal.

## Choosing a Sampler

-   **Langevin Dynamics**: A great default choice. It's fast, simple, and works well for a wide range of problems, especially when paired with a good scheduler.
-   **Hamiltonian Monte Carlo**: Can be more efficient at exploring complex, high-dimensional energy landscapes, but is more computationally intensive per step and has more hyperparameters to tune.

For more details on parallelizing sampling and other advanced techniques, see the [Parallel Sampling](parallel_sampling.md) guide.