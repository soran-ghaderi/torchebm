🍓

# **PyTorch Toolkit for Generative Modeling**

A high-performance PyTorch library that makes Energy-Based Models **accessible** and **efficient** for researchers and practitioners alike.

[⭐ Star on GitHub](https://github.com/soran-ghaderi/torchebm)

**TorchEBM** provides components for 🔬 sampling, 🧠 inference, and 📊 model training.

[Getting Started](https://soran-ghaderi.github.io/torchebm/0.5.4.dev3/tutorials/index.md) [Examples](https://soran-ghaderi.github.io/torchebm/0.5.4.dev3/examples/index.md) [API Reference](https://soran-ghaderi.github.io/torchebm/0.5.4.dev3/api/index.md) [Development](https://soran-ghaderi.github.io/torchebm/0.5.4.dev3/developer_guide/index.md)

______________________________________________________________________

## What is 🍓 TorchEBM?

**TorchEBM** is a PyTorch library for Energy-Based Models (EBMs), a powerful class of generative models. It provides a flexible framework to define, train, and generate samples using energy-based models.

______________________________________________________________________

## Equilibrium Matching in Action

**Equilibrium Matching:** Comparing Linear, VP, and Cosine interpolants transforming noise into structured data distributions.

- **Equilibrium Matching**

  ______________________________________________________________________

  Train generative models by learning velocity fields that transform noise into data.

  [Flow Sampler](https://soran-ghaderi.github.io/torchebm/0.5.4.dev3/api/torchebm/samplers/flow.md)

- **Multiple Interpolants**

  ______________________________________________________________________

  Choose from Linear, Variance-Preserving, and Cosine interpolation schemes.

  [Interpolants](https://soran-ghaderi.github.io/torchebm/0.5.4.dev3/api/torchebm/interpolants/index.md)

______________________________________________________________________

## Core Components

TorchEBM is structured around several key components:

- **Models**

  ______________________________________________________________________

  Define energy functions using `BaseModel`, from analytical forms to custom neural networks.

  [Details](https://soran-ghaderi.github.io/torchebm/0.5.4.dev3/examples/training/index.md)

- **Samplers**

  ______________________________________________________________________

  Generate samples with MCMC samplers like Langevin Dynamics and Hamiltonian Monte Carlo.

  [Details](https://soran-ghaderi.github.io/torchebm/0.5.4.dev3/examples/samplers/index.md)

- **Loss Functions**

  ______________________________________________________________________

  Train models with loss functions like Contrastive Divergence and Score Matching.

  [Details](https://soran-ghaderi.github.io/torchebm/0.5.4.dev3/api/torchebm/losses/index.md)

- **Datasets**

  ______________________________________________________________________

  Use synthetic dataset generators for testing and visualization.

  [Details](https://soran-ghaderi.github.io/torchebm/0.5.4.dev3/examples/datasets/index.md)

- **Visualization**

  ______________________________________________________________________

  Visualize energy landscapes, sampling, and training dynamics.

  [Details](https://soran-ghaderi.github.io/torchebm/0.5.4.dev3/examples/visualization/index.md)

- **Accelerated Computing**

  ______________________________________________________________________

  Accelerate sampling and training with CUDA implementations.

  [Details](https://soran-ghaderi.github.io/torchebm/0.5.4.dev3/api/torchebm/cuda/index.md)

______________________________________________________________________

## Quick Start

Install the library using pip:

```bash
pip install torchebm
```

Here's a minimal example of defining an energy function and a sampler:

```python
import torch
from torchebm.core import GaussianModel
from torchebm.samplers import LangevinDynamics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2), device=device)

sampler = LangevinDynamics(model=model, step_size=0.01, device=device)

initial_points = torch.randn(500, 2, device=device)
samples = sampler.sample(x=initial_points, n_steps=100)

print(f"Output batch_shape: {samples.shape}") # (B, len) -> torch.Size([500, 2])
```

______________________________________________________________________

Latest Release

TorchEBM is currently in early development. Check our [GitHub repository](https://github.com/soran-ghaderi/torchebm) for the latest updates and features.

## Community & Contribution

TorchEBM is an open-source project developed with the research community in mind.

- **Bug Reports & Feature Requests:** Please use the [GitHub Issues](https://github.com/soran-ghaderi/torchebm/issues).
- **Contributing Code:** We welcome contributions! Please see the [Contributing Guidelines](https://soran-ghaderi.github.io/torchebm/0.5.4.dev3/developer_guide/index.md). Consider following the [Commit Conventions](https://soran-ghaderi.github.io/torchebm/0.5.4.dev3/developer_guide/code_guidelines/index.md).
- **Show Support:** If you find TorchEBM helpful for your work, please consider starring the repository on [GitHub](https://github.com/soran-ghaderi/torchebm)!

______________________________________________________________________

## Citation

Please consider citing the TorchEBM repository if it contributes to your research:

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

TorchEBM is available under the **MIT License**. See the [LICENSE file](https://github.com/soran-ghaderi/torchebm/blob/master/LICENSE) for details.
