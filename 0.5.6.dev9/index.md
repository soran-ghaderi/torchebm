🍓

# **PyTorch Library for Generative Modeling**

A high-performance PyTorch library that makes Energy-Based Models **accessible** and **efficient** for researchers and practitioners alike.

[⭐ Star on GitHub](https://github.com/soran-ghaderi/torchebm)

A PyTorch library for energy-based modeling, with support for flow and diffusion methods.

[Getting Started](https://soran-ghaderi.github.io/torchebm/0.5.6.dev9/tutorials/index.md) [Examples](https://soran-ghaderi.github.io/torchebm/0.5.6.dev9/examples/index.md) [API Reference](https://soran-ghaderi.github.io/torchebm/0.5.6.dev9/api/index.md) [Development](https://soran-ghaderi.github.io/torchebm/0.5.6.dev9/developer_guide/index.md)

Training an energy-based model to capture a target distribution.

______________________________________________________________________

## What is TorchEBM 🍓?

Energy-based models assign a scalar energy to each input, implicitly defining a probability distribution where lower energy means higher probability. This formulation is remarkably general. MCMC sampling, score matching, contrastive divergence, and even flow/diffusion-based generation all operate within or connect naturally to the energy-based framework.

**TorchEBM** gives you composable PyTorch building blocks that span this entire landscape. You can define energy functions, train models with different learning objectives, and generate samples via MCMC, energy minimization, or learned continuous-time dynamics.

______________________________________________________________________

## In Action

Eight-gaussians distribution.

Circles distribution.

Equilibrium matching with different interpolants transforming noise into structured distributions.

______________________________________________________________________

## Core Components

- **Core**

  ______________________________________________________________________

  Base classes, energy models (analytical potentials and custom neural networks), schedulers, and the device/dtype management layer shared across all components.

  [API Reference](https://soran-ghaderi.github.io/torchebm/0.5.6.dev9/api/torchebm/core/index.md)

- **Samplers**

  ______________________________________________________________________

  Draw samples from energy landscapes via MCMC methods, gradient-based optimization, or learned flow/diffusion dynamics (ODE/SDE).

  [API Reference](https://soran-ghaderi.github.io/torchebm/0.5.6.dev9/api/torchebm/samplers/index.md)

- **Loss Functions**

  ______________________________________________________________________

  Training objectives for energy-based and flow-based models, including contrastive divergence variants, score matching variants, and equilibrium matching.

  [API Reference](https://soran-ghaderi.github.io/torchebm/0.5.6.dev9/api/torchebm/losses/index.md)

- **Interpolants**

  ______________________________________________________________________

  Define how noise and data are mixed along a continuous time path. Used in flow matching, diffusion, and related generative schemes.

  [API Reference](https://soran-ghaderi.github.io/torchebm/0.5.6.dev9/api/torchebm/interpolants/index.md)

- **Integrators**

  ______________________________________________________________________

  Numerical solvers for SDEs, ODEs, and Hamiltonian systems. Pluggable into samplers and flow-based generation pipelines.

  [API Reference](https://soran-ghaderi.github.io/torchebm/0.5.6.dev9/api/torchebm/integrators/index.md)

- **Models**

  ______________________________________________________________________

  Neural network architectures for parameterizing energy functions and velocity fields, including vision transformers and guidance wrappers.

  [API Reference](https://soran-ghaderi.github.io/torchebm/0.5.6.dev9/api/torchebm/models/index.md)

- **Datasets**

  ______________________________________________________________________

  Synthetic 2D distributions for rapid prototyping and visual evaluation. All are PyTorch `Dataset` objects.

  [API Reference](https://soran-ghaderi.github.io/torchebm/0.5.6.dev9/api/torchebm/datasets/index.md)

- **CUDA**

  ______________________________________________________________________

  CUDA-accelerated kernels and mixed precision support for performance-critical sampling and training.

  [API Reference](https://soran-ghaderi.github.io/torchebm/0.5.6.dev9/api/torchebm/cuda/index.md)

______________________________________________________________________

## Energy Landscapes

|            |             |           |
| ---------- | ----------- | --------- |
|            |             |           |
| Gaussian   | Double Well | Rastrigin |
|            |             |           |
| Rosenbrock | Ackley      | Harmonic  |

______________________________________________________________________

## Synthetic Datasets

|                  |                 |           |            |
| ---------------- | --------------- | --------- | ---------- |
|                  |                 |           |            |
| Gaussian Mixture | Eight Gaussians | Two Moons | Swiss Roll |
|                  |                 |           |            |
| Checkerboard     | Pinwheel        | Circle    | Grid       |

______________________________________________________________________

## Quick Start

```bash
pip install torchebm
```

Define an energy model, create a sampler, and draw samples in a few lines:

```python
import torch
from torchebm.core import GaussianModel
from torchebm.samplers import LangevinDynamics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2), device=device)

sampler = LangevinDynamics(model=model, step_size=0.01, device=device)
samples = sampler.sample(x=torch.randn(500, 2, device=device), n_steps=100)
```

See the [tutorials](https://soran-ghaderi.github.io/torchebm/0.5.6.dev9/tutorials/index.md) and [examples](https://soran-ghaderi.github.io/torchebm/0.5.6.dev9/examples/index.md) for training loops, flow-based generation, and more.

## Community & Contribution

TorchEBM is open-source and developed with the research community in mind.

- **Issues & feature requests** on [GitHub Issues](https://github.com/soran-ghaderi/torchebm/issues)
- **Contributing** via [developer guide](https://soran-ghaderi.github.io/torchebm/0.5.6.dev9/developer_guide/index.md) and [code guidelines](https://soran-ghaderi.github.io/torchebm/0.5.6.dev9/developer_guide/code_guidelines/index.md)
- **Star the repo** on [GitHub](https://github.com/soran-ghaderi/torchebm) if you find it useful

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

______________________________________________________________________

## License

MIT License. See the [LICENSE file](https://github.com/soran-ghaderi/torchebm/blob/master/LICENSE) for details.
