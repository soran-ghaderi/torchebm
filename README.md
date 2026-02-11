<p align="center">
  <img src="docs/assets/images/logo_with_text.svg" alt="TorchEBM Logo" width="350">
</p>

<p align="center" style="margin-bottom: 20px;">
    <a href="https://pypi.org/project/torchebm/" target="_blank" title="PyPI version">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/torchebm?style=flat-square&color=blue">
    </a>
    <a href="https://github.com/soran-ghaderi/torchebm/blob/master/LICENSE" target="_blank" title="License">
        <img alt="License" src="https://img.shields.io/github/license/soran-ghaderi/torchebm?style=flat-square&color=brightgreen">
    </a>
    <a href="https://github.com/soran-ghaderi/torchebm" target="_blank" title="GitHub Repo Stars">
        <img alt="GitHub Stars" src="https://img.shields.io/github/stars/soran-ghaderi/torchebm?style=social">
    </a>
    <a href="https://deepwiki.com/soran-ghaderi/torchebm"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
    <!-- Consider adding: build status, documentation status, code coverage -->
    <a href="https://github.com/soran-ghaderi/torchebm/actions" target="_blank" title="Build Status">
      <img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/soran-ghaderi/torchebm/tag-release.yml?branch=master&style=flat-square&label=build">
    </a>
    <!-- Docs badge -->
    <a href="https://github.com/soran-ghaderi/torchebm/actions" target="_blank" title="Documentation">
      <img alt="Docs" src="https://img.shields.io/github/actions/workflow/status/soran-ghaderi/torchebm/docs_ci.yml?branch=master&style=flat-square&label=docs">
    </a>
<!--     <a href="https://pepy.tech/project/torchebm" target="_blank" title="Downloads">
        <img alt="Downloads" src="https://img.shields.io/pypi/dm/torchebm?style=flat-square">
    </a> -->
    <a href="https://pepy.tech/project/torchebm" target="_blank" title="Downloads">
        <img alt="Downloads" src="https://static.pepy.tech/badge/torchebm?style=flat-square">
    </a>
    <a href="https://pypi.org/project/torchebm/" target="_blank" title="Python Versions">
        <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/torchebm?style=flat-square">
    </a>
</p>


<p align="center">⚡ A PyTorch library for energy-based modeling, with support for flow and diffusion methods.</p>

<p align="center">
  <img src="docs/assets/animations/ebm_training_animation.gif" alt="EBM Training Animation"/>
</p>

## What is ∇ TorchEBM 🍓? 

Energy-based models define distributions through a scalar energy function, where lower energy means higher probability. This is a very general formulation and many generative approaches, from MCMC sampling to score matching to flow-based generation, can be understood through this lens.

**TorchEBM** is a PyTorch library that gives you composable tools for this entire spectrum. You can define energy landscapes, train models with various learning objectives, and sample via MCMC, optimization, or learned continuous-time dynamics (ODEs/SDEs). The library handles classical EBM training (contrastive divergence, score matching) as well as modern interpolant-based and equilibrium-based generation methods.

📚 For the full documentation, please visit the [official website of TorchEBM 🍓](https://soran-ghaderi.github.io/torchebm/).

## Features

- **Energy models** with built-in analytical potentials and support for custom neural network energy functions
- **MCMC and optimization-based samplers** for drawing samples from energy landscapes
- **Flow and diffusion samplers** that generate via ODE/SDE integration of learned velocity or score fields
- **Training objectives** including contrastive divergence variants, score matching variants, and equilibrium matching
- **Interpolation schemes** for specifying noise-to-data paths in flow and diffusion models
- **Numerical integrators** for SDE, ODE, and Hamiltonian dynamics
- **Neural network architectures** ready for conditional generation
- **Synthetic datasets** for rapid prototyping and benchmarking
- **Hyperparameter schedulers** for step sizes, noise scales, and other training parameters
- **CUDA acceleration** and mixed precision support

<p align="center">
  <img src="docs/assets/animations/8gaussians_flow.gif" alt="8 Gaussians Flow" width="700"/>
</p>
<table align="center">
  <tr>
    <td><img src="docs/assets/images/e_functions/gaussian.png" alt="Gaussian" width="200"/></td>
    <td><img src="docs/assets/images/e_functions/double_well.png" alt="Double Well" width="200"/></td>
    <td><img src="docs/assets/images/e_functions/rastrigin.png" alt="Rastrigin" width="200"/></td>
    <td><img src="docs/assets/images/e_functions/rosenbrock.png" alt="Rosenbrock" width="200"/></td>
  </tr>
  <tr>
    <td align="center">Gaussian</td>
    <td align="center">Double Well</td>
    <td align="center">Rastrigin</td>
    <td align="center">Rosenbrock</td>
  </tr>
</table>

<table align="center">
  <tr>
    <td><img src="docs/assets/images/datasets/gaussian_mixture.png" alt="Gaussian Mixture" width="200"/></td>
    <td><img src="docs/assets/images/datasets/two_moons.png" alt="Two Moons" width="200"/></td>
    <td><img src="docs/assets/images/datasets/swiss_roll.png" alt="Swiss Roll" width="200"/></td>
    <td><img src="docs/assets/images/datasets/checkerboard.png" alt="Checkerboard" width="200"/></td>
  </tr>
  <tr>
    <td align="center">Gaussian Mixture</td>
    <td align="center">Two Moons</td>
    <td align="center">Swiss Roll</td>
    <td align="center">Checkerboard</td>
  </tr>
</table>


## Installation

```bash
pip install torchebm
```

#### Dependencies
- [PyTorch](https://pytorch.org/) (with CUDA support for optimal performance)
- Other dependencies are listed in [requirements.txt](requirements.txt)


## Usage Examples

### MCMC Sampling

```python
import torch
from torchebm.core import GaussianModel
from torchebm.samplers import LangevinDynamics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2), device=device)

sampler = LangevinDynamics(model=model, step_size=0.01, device=device)
samples = sampler.sample(x=torch.randn(500, 2, device=device), n_steps=100)
print(samples.shape)  # torch.Size([500, 2])
```

### Training with Contrastive Divergence

```python
import torch
from torchebm.core import BaseModel
from torchebm.samplers import LangevinDynamics
from torchebm.losses import ContrastiveDivergence
from torchebm.datasets import GaussianMixtureDataset
from torch.utils.data import DataLoader

class MLPEnergy(BaseModel):
    def __init__(self, dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPEnergy(dim=2).to(device)
sampler = LangevinDynamics(model=model, step_size=0.01, device=device)
cd_loss = ContrastiveDivergence(model=model, sampler=sampler, k_steps=10)

data = GaussianMixtureDataset(n_samples=1000, n_components=4).get_data()
loader = DataLoader(data, batch_size=64, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for batch in loader:
        optimizer.zero_grad()
        loss, _ = cd_loss(batch.to(device))
        loss.backward()
        optimizer.step()
```

### Hamiltonian Monte Carlo

```python
import torch
from torchebm.core import GaussianModel
from torchebm.samplers import HamiltonianMonteCarlo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GaussianModel(mean=torch.zeros(10), cov=torch.eye(10), device=device)

hmc = HamiltonianMonteCarlo(model=model, step_size=0.1, n_leapfrog_steps=10, device=device)
samples = hmc.sample(dim=10, n_steps=500, n_samples=1000)
print(samples.shape)  # torch.Size([1000, 10])
```

## Library Structure

```
torchebm/
├── core/           # Base classes, energy models, schedulers, device management
├── samplers/       # MCMC, optimization, and flow/diffusion samplers
├── losses/         # Training objectives (CD, score matching, equilibrium matching)
├── interpolants/   # Noise-to-data interpolation schemes
├── integrators/    # Numerical integrators for SDE/ODE/Hamiltonian dynamics
├── models/         # Neural network architectures
├── datasets/       # Synthetic data generators
├── utils/          # Visualization and training utilities
└── cuda/           # CUDA-accelerated implementations
```

## Visualization Examples

<table align="center">
  <tr>
    <td><img src="docs/assets/images/sampling.jpg" alt="Langevin Dynamics Sampling" width="250"/></td>
    <td><img src="docs/assets/images/trajectory.jpg" alt="Langevin Dynamics Trajectory" width="250"/></td>
    <td><img src="docs/assets/images/parallel.jpg" alt="Parallel Sampling" width="250"/></td>
  </tr>
  <tr>
    <td align="center">Langevin Dynamics Sampling</td>
    <td align="center">Langevin Dynamics Trajectory</td>
    <td align="center">Parallel Sampling</td>
  </tr>
</table>

<p align="center">
  <img src="docs/assets/animations/circles_flow.gif" alt="Flow Comparison" width="700"/>
  <br>
  <em>Equilibrium Matching: Linear, VP, and Cosine interpolants transforming noise into data.</em>
</p>

Check out the `examples/` directory for sample scripts.

## Contributing

Contributions are welcome! Step-by-step instructions for contributing to the project can be found on the [contributing.md](docs/developer_guide/contributing.md) page on the website.

Please check the issues page for current tasks or create a new issue to discuss proposed changes.

## Show your Support for ∇ TorchEBM 🍓

Please ⭐️ this repository if ∇ TorchEBM helped you and spread the word.

Thank you! 🚀

## Citation

If TorchEBM is useful in your research, please cite it:

```bibtex
@misc{torchebm_library_2025,
  author       = {Ghaderi, Soran and Contributors},
  title        = {{TorchEBM}: A PyTorch Library for Training Energy-Based Models},
  year         = {2025},
  url          = {https://github.com/soran-ghaderi/torchebm},
}
```

## Changelog

See [CHANGELOG](CHANGELOG.md) for version history.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Research Collaboration

If you are interested in collaborating on research around energy-based, flow-based, or diffusion models, feel free to reach out. Contributions to TorchEBM 🍓 and discussions that push the field forward are always welcome.
