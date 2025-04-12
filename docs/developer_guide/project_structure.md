---
title: Project Structure
description: Overview of TorchEBM's project structure and organization
icon: material/folder-outline
---

# Project Structure

!!! abstract "Codebase Organization"
    Understanding the TorchEBM project structure helps you navigate the codebase and contribute effectively. This guide provides an overview of the repository organization.

## Repository Overview

The TorchEBM repository is organized as follows:

```
torchebm/
├── torchebm/              # Main package source code
│   ├── core/              # Core functionality and base classes
│   ├── samplers/          # Sampling algorithms implementation
│   ├── losses/            # BaseLoss functions for training
│   ├── models/            # Neural network model implementations
│   ├── cuda/              # CUDA optimized implementations
│   └── utils/             # Utility functions and helpers
├── tests/                 # Test directory
├── docs/                  # Documentation
├── examples/              # Example applications
├── benchmarks/            # Performance benchmarks
├── setup.py               # Package setup script
└── README.md              # Project README
```

## Main Package Structure

<div class="grid" markdown>

<div markdown>
### `torchebm.core`

Contains the core functionality including base classes and essential components:

* `base.py` - Base classes for the entire library
* `energy_function.py` - Base energy function class and interface
* `analytical_functions.py` - Analytical energy function implementations
* `distributions.py` - Probability distribution implementations
</div>

<div markdown>
### `torchebm.samplers`

Implementations of various sampling algorithms:

* `base.py` - Base sampler class
* `langevin_dynamics.py` - Langevin dynamics implementation
* `hmc.py` - Hamiltonian Monte Carlo
* `metropolis_hastings.py` - Metropolis-Hastings
* [Other sampler implementations]
</div>

</div>

<div class="grid" markdown>

<div markdown>
### `torchebm.losses`

BaseLoss functions for training energy-based models:

* `base.py` - Base loss class
* `contrastive_divergence.py` - Contrastive divergence implementations
* `score_matching.py` - Score matching methods
* [Other loss implementations]
</div>

<div markdown>
### `torchebm.models`

Neural network model implementations:

* `mlp.py` - Multi-layer perceptron energy models
* `cnn.py` - Convolutional neural network energy models
* `ebm.py` - Generic energy-based model implementations
* [Other model architectures]
</div>

</div>

<div class="grid" markdown>

<div markdown>
### `torchebm.cuda`

CUDA-optimized implementations for performance-critical operations:

* `kernels/` - CUDA kernel implementations
* `bindings.cpp` - PyTorch C++ bindings
* `ops.py` - Python interfaces to CUDA operations
</div>

<div markdown>
### `torchebm.utils`

Utility functions and helpers:

* `device.py` - Device management utilities
* `visualization.py` - Visualization tools
* `data.py` - Data loading and processing utilities
* `logging.py` - Logging utilities
</div>

</div>

## Tests Structure

The tests directory mirrors the package structure:

```
tests/
├── unit/                  # Unit tests
│   ├── core/              # Tests for core module
│   ├── samplers/          # Tests for samplers module
│   ├── losses/            # Tests for losses module
│   └── utils/             # Tests for utilities
├── integration/           # Integration tests
├── performance/           # Performance benchmarks
├── conftest.py            # Pytest configuration and fixtures
└── utils.py               # Test utilities
```

## Documentation Structure

The documentation is built with MkDocs and organized as follows:

```
docs/
├── index.md               # Home page
├── getting_started.md     # Quick start guide
├── guides/                # User guides
├── api/                   # API reference (auto-generated)
├── examples/              # Example documentation
├── developer_guide/       # Developer documentation
│   ├── contributing.md    # Contributing guidelines
│   ├── code_style.md      # Code style guide
│   └── ...                # Other developer docs
└── assets/                # Static assets
    ├── images/            # Images
    ├── stylesheets/       # Custom CSS
    └── javascripts/       # Custom JavaScript
```

## Examples Structure

Example applications to demonstrate TorchEBM usage:

```
examples/
├── basic/                 # Basic usage examples
├── advanced/              # Advanced examples
└── real_world/            # Real-world applications
```

## Dependencies and Requirements

TorchEBM has the following dependencies:

<div class="grid cards" markdown>

-   :material-torch:{ .lg .middle } __PyTorch__

    ---

    Primary framework for tensor operations and automatic differentiation.

-   :material-numpy:{ .lg .middle } __NumPy__

    ---

    Used for numerical operations when PyTorch isn't needed.

-   :material-chart-line:{ .lg .middle } __Matplotlib__

    ---

    For visualization capabilities.

-   :fontawesome-solid-gears:{ .lg .middle } __tqdm__

    ---

    For progress bars during long operations.

</div>

## Entry Points

The main entry points to the library are:

* `torchebm.core` - Import core functionality
* `torchebm.samplers` - Import samplers
* `torchebm.losses` - Import loss functions
* `torchebm.models` - Import neural network models

Example of typical import patterns:

```python
# Import core components
from torchebm.core import BaseEnergyFunction, GaussianEnergy

# Import samplers
from torchebm.samplers import LangevinDynamics, HamiltonianMC

# Import loss functions
from torchebm.losses import ContrastiveDivergence

# Import utilities
from torchebm.utils import visualize_samples
```

## Package Management

TorchEBM uses setuptools for package management with setup.py:

```python
# setup.py excerpt
setup(
    name="torchebm",
    version=__version__,
    description="Energy-Based Modeling library for PyTorch",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
    ],
    # Additional configuration...
)
```

## Version Control Structure

* We follow [Conventional Commits](https://www.conventionalcommits.org/) for our commit messages
* Feature branches should be named `feature/feature-name`
* Bugfix branches should be named `bugfix/bug-name`
* Release branches should be named `release/vX.Y.Z`

!!! tip "Finding Your Way Around"
    When contributing to TorchEBM, start by exploring the relevant directory for your feature. For example, if you're adding a new sampler, look at the existing implementations in `torchebm/samplers/` to understand the pattern. 