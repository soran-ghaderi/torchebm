# Getting Started

TorchEBM provides composable PyTorch primitives for energy-based generative modeling: energies, samplers, losses, interpolants, couplings, integrators, and schedulers. This page takes you from installation to the core workflows the library is built around.

## Installation

```bash
pip install torchebm
```

The only hard dependencies are PyTorch and NumPy. A CUDA-enabled PyTorch build is recommended for training; every snippet below also runs on CPU.

## Sample from an energy

An energy (E(x)) defines a density (p(x) \\propto e^{-E(x)}). Samplers draw from it using only the energy's gradient, here with Langevin dynamics:

```python
import torch
from torchebm.core import GaussianModel
from torchebm.samplers import LangevinDynamics

model = GaussianModel(mean=torch.zeros(2), cov=torch.tensor([[1.0, 0.8], [0.8, 1.0]]))
sampler = LangevinDynamics(model=model, step_size=0.02, noise_scale=1.0)

samples = sampler.sample(dim=2, n_samples=2000, n_steps=500)  # (2000, 2)
print(torch.cov(samples.T))  # recovers the target covariance
```

Every sampler is vectorized over chains: the call above runs 2000 independent chains as one tensor program. Swap in `HamiltonianMonteCarlo` or change `integrator=` without touching the rest.

## Train an energy from data

Any differentiable module mapping `(N, d)` to `(N,)` is a valid energy. Train it with contrastive divergence, using a sampler to draw negatives:

```python
import torch
from torch import nn
from torchebm.core import BaseModel
from torchebm.datasets import TwoMoonsDataset
from torchebm.losses import ContrastiveDivergence
from torchebm.samplers import LangevinDynamics

class MLPEnergy(BaseModel):
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
energy = MLPEnergy()
sampler = LangevinDynamics(model=energy, step_size=0.1, noise_scale=1.0)
cd = ContrastiveDivergence(model=energy, sampler=sampler, k_steps=10)
opt = torch.optim.Adam(energy.parameters(), lr=1e-3)

for step in range(1000):
    batch = data[torch.randint(len(data), (256,))]
    loss, _ = cd(batch)
    opt.zero_grad(); loss.backward(); opt.step()
```

## Train a generative flow

The same library expresses continuous-time generation. Equilibrium matching trains a time-invariant field, and `FlowSampler` integrates it from noise:

```python
import torch
from torch import nn
from torchebm.datasets import TwoMoonsDataset
from torchebm.losses import EquilibriumMatchingLoss
from torchebm.samplers import FlowSampler

class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 2),
        )
    def forward(self, x, t, **kwargs):
        return self.net(x)

data = TwoMoonsDataset(n_samples=4000, noise=0.05, seed=0).get_data()
model = Field()
loss_fn = EquilibriumMatchingLoss(model=model, interpolant="linear", energy_type="dot")
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

for step in range(3000):
    batch = data[torch.randint(len(data), (256,))]
    loss = loss_fn(batch)
    opt.zero_grad(); loss.backward(); opt.step()

flow = FlowSampler(model=model, interpolant="linear",
                   negate_velocity=True, integrator="euler")
samples = flow.sample(x=torch.randn(4000, 2), n_steps=100)
```

The pieces are interchangeable by construction: the interpolant string selects the probability path, the integrator string selects the numerics, and a `coupling=` argument on transport-based losses selects how noise is paired with data.

## Where to go next

- [Concepts](https://soran-ghaderi.github.io/torchebm/latest/concepts/index.md): the theory, and how the components compose into EBMs, diffusion, flow matching, Schrödinger bridges, and beyond.
- [Examples](https://soran-ghaderi.github.io/torchebm/latest/examples/index.md): a runnable, CI-tested curriculum from energy landscapes to full training recipes.
- [API Reference](https://soran-ghaderi.github.io/torchebm/latest/api/index.md): signatures and details for every component.
