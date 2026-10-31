---
title: Getting Started
description: Install TorchEBM and run your first sampler, your first trained energy, and your first generative flow.
icon: material/rocket-launch-outline
---

# Getting Started

TorchEBM provides composable PyTorch primitives for generative models that
learn dynamics on data space, from energy-based models to diffusion, flow
matching, and equilibrium and energy matching: energies and fields, samplers,
losses, interpolants, couplings, integrators, and schedulers. This page takes you from installation to the core workflows the
library is built around.

## Installation

```bash
pip install torchebm
```

The only hard dependencies are PyTorch and NumPy. A CUDA-enabled PyTorch build
is recommended for training; every snippet below also runs on CPU.

## Sample from an energy

An energy \(E(x)\) defines a density \(p(x) \propto e^{-E(x)}\). Samplers draw
from it using only the energy's gradient, here with Langevin dynamics. The
model is an argument of the call, as the function is for `torchdiffeq`:

```python
import torch
from torchebm.core import GaussianModel
from torchebm.samplers import LangevinDynamics

model = GaussianModel(mean=torch.zeros(2), cov=torch.tensor([[1.0, 0.8], [0.8, 1.0]]))
sampler = LangevinDynamics(step_size=0.02, noise_scale=1.0)

samples = sampler(model, torch.randn(2000, 2), n_steps=500)
print(torch.cov(samples.T))
```

Every sampler is vectorized over chains: the call above runs 2000 independent
chains as one tensor program and recovers the target covariance. Swap in
`HamiltonianMonteCarlo` or change `integrator=` without touching the rest.

## Train an energy from data

Any differentiable module mapping `(N, d)` to `(N,)` is a valid energy. Train
it with contrastive divergence: the sampler draws negatives from the current
model and the loss compares the energies of data and negatives:

```python
import torch
from torch import nn
from torchebm.core import BaseModel
from torchebm.datasets import TwoMoonsDataset
from torchebm.losses import ContrastiveDivergenceLoss
from torchebm.samplers import LangevinDynamics

class MLP(BaseModel):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

data = TwoMoonsDataset(n_samples=3000, noise=0.05, seed=0).get_data()
model = MLP()
sampler = LangevinDynamics(step_size=0.1, noise_scale=1.0)
loss_fn = ContrastiveDivergenceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(1000):
    batch = data[torch.randint(len(data), (256,))]
    negatives = sampler(model, batch, n_steps=10)
    loss = loss_fn(model(batch), model(negatives))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Train a generative flow

The same library expresses continuous-time generation. Equilibrium matching
regresses a time-invariant field onto the velocity of the interpolant path,
and `FlowSampler` integrates it from noise:

```python
import torch
from torch import nn
from torchebm.datasets import TwoMoonsDataset
from torchebm.interpolants import LinearInterpolant
from torchebm.losses import EquilibriumMatchingLoss
from torchebm.samplers import FlowSampler

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 2),
        )
    def forward(self, x, t=None, **kwargs):
        return self.net(x)

data = TwoMoonsDataset(n_samples=4000, noise=0.05, seed=0).get_data()
model = MLP()
path = LinearInterpolant()
loss_fn = EquilibriumMatchingLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(3000):
    x1 = data[torch.randint(len(data), (256,))]
    x0 = torch.randn_like(x1)
    t = torch.rand(len(x1))
    x_t, u_t = path(x0, x1, t)
    loss = loss_fn(model(x_t), u_t, t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

sampler = FlowSampler(integrator="euler")
samples = sampler(model, torch.randn(4000, 2), n_steps=100)
```

Each piece is chosen independently: the interpolant selects the probability
path, the integrator string selects the numerics, and a coupling called before
the interpolant selects how noise is paired with data.

## Where to go next

- [Concepts](../concepts/index.md): the theory, and how the components compose
  into EBMs, diffusion, flow matching, Schrödinger bridges, and beyond.
- [Examples](../examples/index.md): a runnable, CI-tested curriculum from energy
  landscapes to full training recipes.
- [API Reference](../api/index.md): signatures and details for every component.
