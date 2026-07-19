TorchEBM🍓

TorchEBM: Simulation-free, GPU-first generative modeling in PyTorch\
Composable primitives for scalable, stable training of modern **EBMs**, **diffusion**, **flow matching**, and **Schrödinger bridges**.

[Get Started](https://soran-ghaderi.github.io/torchebm/latest/getting_started/index.md)

______________________________________________________________________

## Overview

TorchEBM is a PyTorch library for simulation-free, GPU-first generative modeling: scalable, stable training of modern energy-based models, diffusion, flow matching, and Schrödinger bridges. Energy-based models define probability distributions through a scalar energy function, and the formulation is general enough that much of modern generative modeling, from MCMC sampling and score matching to simulation-free transport along probability paths, factors into the same components, i.e. fields, probability paths, couplings, objectives, and integrators. TorchEBM implements these components as composable, high-throughput PyTorch primitives.

The [Design and Scope](https://soran-ghaderi.github.io/torchebm/latest/concepts/design/index.md) page states this framing precisely and places each method family within it.

______________________________________________________________________

## In Action

Eight-gaussians distribution

Circles distribution

Equilibrium matching with different interpolants transporting noise onto structured distributions.

______________________________________________________________________

## Core Components

- [**Core**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/core/index.md)
- [**Samplers**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/samplers/index.md)
- [**Losses**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/losses/index.md)
- [**Interpolants**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/interpolants/index.md)
- [**Couplings**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/couplings/index.md)
- [**Integrators**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/integrators/index.md)
- [**Models**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/models/index.md)
- [**Datasets**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/datasets/index.md)
- [**Utils**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/utils/index.md)
- [**CUDA**](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/cuda/index.md)

______________________________________________________________________

## Quick Start

```bash
pip install torchebm
```

Train a generative model with **equilibrium matching**, then sample the same network both as an ODE flow and as a scalar energy:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchebm.losses import EquilibriumMatchingLoss
from torchebm.samplers import FlowSampler, NesterovSampler
from torchebm.core import BaseModel
from torchebm.datasets import EightGaussiansDataset

dataset = EightGaussiansDataset(n_samples=8192)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# Any nn.Module with forward(x, t) works
class VelocityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2))
    def forward(self, x, t, **kwargs):
        return self.net(x)

model = VelocityNet()

loss_fn = EquilibriumMatchingLoss(
    model=model, interpolant="linear", energy_type="dot",
)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(50):
    for x in loader:
        loss = loss_fn(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Sample via ODE flow
flow = FlowSampler(model=model, interpolant="linear", negate_velocity=True)
flow_samples = flow.sample(x=torch.randn(1000, 2), n_steps=100)

# Same model as a scalar energy: g(x) = x · f(x)
class LearnedEnergy(BaseModel):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        t = torch.zeros(x.shape[0], device=x.device)
        return (x * self.net(x, t)).sum(-1)

nesterov = NesterovSampler(LearnedEnergy(model), step_size=0.01, momentum=0.9)
energy_samples = nesterov.sample(n_samples=1000, dim=2, n_steps=200)
```

One model, two views: the flow view generates in a fixed number of steps, the energy view supports gradient-based refinement. That interchangeability is the library's central design property.

See the [concepts](https://soran-ghaderi.github.io/torchebm/latest/concepts/index.md) and [examples](https://soran-ghaderi.github.io/torchebm/latest/examples/index.md) for the theory behind each component and a runnable, CI-tested curriculum.

**Enjoying TorchEBM?** A GitHub star helps others discover the project and motivates continued development.

[Star on GitHub](https://github.com/soran-ghaderi/torchebm)

______________________________________________________________________

## Citation

If TorchEBM is useful in your research, please cite it:

```bibtex
@misc{torchebm_library_2025,
  author       = {Ghaderi, Soran and Contributors},
  title        = {{TorchEBM}: Simulation-Free, {GPU}-First Generative Modeling in {PyTorch}},
  year         = {2025},
  url          = {https://github.com/soran-ghaderi/torchebm},
}
```

MIT License · [Issues](https://github.com/soran-ghaderi/torchebm/issues) · [Contributing](https://soran-ghaderi.github.io/torchebm/latest/developer_guide/index.md) · [GitHub](https://github.com/soran-ghaderi/torchebm)
