<h1 align="center">TorchEBM</h1>


<p align="center">⚡ Energy-Based Modeling library for PyTorch, offering tools for 🔬 sampling, 🧠 inference, and 📊 learning in complex distributions.</p>

## About

TorchEBM is a CUDA-accelerated parallel library for Energy-Based Models (EBMs) built on PyTorch. It provides efficient implementations of sampling, inference, and learning algorithms for EBMs, with a focus on scalability and performance.
This is an early version and is under development.
## Features (so far)

- Langevin dynamics sampling
- CUDA-accelerated implementations
- Seamless integration with PyTorch

## Installation

```bash
pip install torchebm
```

## Usage
```python
import torch
from torchebm import EnergyFunction, LangevinDynamics
import matplotlib.pyplot as plt

# You can define your energy function like the following. However you don't need to implement the gradient and it is already automated, but for the sake of the example, we'll include it.
class QuadraticEnergy(EnergyFunction):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(x**2, dim=-1)
    
    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return x

# Instantiate the energy function and the sampler
energy_fn = QuadraticEnergy()
sampler = LangevinDynamics(energy_fn, step_size=0.1, noise_scale=0.1)

# Generate samples
initial_state = torch.tensor([2.0, 2.0])
samples = sampler.sample_chain(initial_state, n_steps=1000, n_samples=500)

# A Single trajectory
trajectory = sampler.sample(initial_state, n_steps, return_trajectory=True)

# Demonstrate parallel sampling
n_chains = 10
initial_states = torch.randn(n_chains, 2) * 2
parallel_samples = sampler.sample_parallel(initial_states, n_steps=1000)
```
### Example Output:
For the visualization codes, check out the examples directory
<table>
  <tr>
    <td><img src="examples/static/images/sampling.jpg" alt="Langevin Dynamics Sampling" width="250"/></td>
    <td><img src="examples/static/images/trajectory.jpg" alt="Single Langevin Dynamics Trajectory" width="250"/></td>
    <td><img src="examples/static/images/parallel.jpg" alt="Parallel Langevin Dynamics Sampling" width="250"/></td>
  </tr>
  <tr>
    <td align="center">Langevin Dynamics Sampling</td>
    <td align="center">Single Langevin Dynamics Trajectory</td>
    <td align="center">Parallel Langevin Dynamics Sampling</td>
  </tr>
</table>

## Contributing
Contributions are welcome! Please check the issues page for current tasks or create a new issue to discuss proposed changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

