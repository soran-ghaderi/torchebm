TorchEBM 🍓

A high-performance PyTorch library that makes Energy-Based Models **accessible** and **efficient** for researchers and practitioners alike.

[Get Started](https://soran-ghaderi.github.io/torchebm/latest/tutorials/index.md)

______________________________________________________________________

## Overview

Energy-based models assign a scalar energy to each input, implicitly defining a probability distribution where lower energy means higher probability. **TorchEBM** gives you composable PyTorch building blocks that span this landscape, from energy functions and MCMC samplers to flow matching and diffusion-based generation.

______________________________________________________________________

## In Action

Eight-gaussians distribution

Circles distribution

Equilibrium matching with different interpolants transforming noise into structured distributions.

______________________________________________________________________

## Core Components

- **Core**

  ______________________________________________________________________

  Base classes, energy models, schedulers, and the device/dtype management layer shared across all components.

  [API Reference](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/core/index.md)

- **Samplers**

  ______________________________________________________________________

  Draw samples from energy landscapes via MCMC methods, gradient-based optimization, or learned flow/diffusion dynamics.

  [API Reference](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/samplers/index.md)

- **Loss Functions**

  ______________________________________________________________________

  Training objectives including contrastive divergence, score matching variants, and equilibrium matching.

  [API Reference](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/losses/index.md)

- **Interpolants**

  ______________________________________________________________________

  Define how noise and data are mixed along a continuous time path for flow matching and diffusion.

  [API Reference](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/interpolants/index.md)

- **Integrators**

  ______________________________________________________________________

  Numerical solvers for SDEs, ODEs, and Hamiltonian systems. Pluggable into samplers and generation pipelines.

  [API Reference](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/integrators/index.md)

- **Models**

  ______________________________________________________________________

  Neural architectures for energy functions and velocity fields, including vision transformers and guidance wrappers.

  [API Reference](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/models/index.md)

- **Datasets**

  ______________________________________________________________________

  Synthetic 2D distributions for rapid prototyping and visual evaluation. All PyTorch `Dataset` objects.

  [API Reference](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/datasets/index.md)

- **CUDA**

  ______________________________________________________________________

  CUDA-accelerated kernels and mixed precision support for performance-critical sampling and training.

  [API Reference](https://soran-ghaderi.github.io/torchebm/latest/api/torchebm/cuda/index.md)

______________________________________________________________________

## Quick Start

```bash
pip install torchebm
```

Train a generative model with **Equilibrium Matching**, then sample with both a flow solver and an energy-based sampler:

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
flow_samples = flow.sample_ode(torch.randn(1000, 2), num_steps=100)

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

See the [tutorials](https://soran-ghaderi.github.io/torchebm/latest/tutorials/index.md) and [examples](https://soran-ghaderi.github.io/torchebm/latest/examples/index.md) for CIFAR-10 generation, score matching, and more.

**Enjoying TorchEBM?** A GitHub star helps others discover the project and motivates continued development.

[Star on GitHub](https://github.com/soran-ghaderi/torchebm)

______________________________________________________________________

## Citation

If TorchEBM is useful in your research, please cite it:

```bibtex
@misc{torchebm_library_2025,
  author       = {Ghaderi, Soran and Contributors},
  title        = {TorchEBM: A PyTorch Library for Training Energy-Based Models},
  year         = {2025},
  url          = {https://github.com/soran-ghaderi/torchebm},
}
```

MIT License · [Issues](https://github.com/soran-ghaderi/torchebm/issues) · [Contributing](https://soran-ghaderi.github.io/torchebm/latest/developer_guide/index.md) · [GitHub](https://github.com/soran-ghaderi/torchebm)
