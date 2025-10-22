---
title: API Reference
description: Detailed API reference for TorchEBM
icon: material/file-document
---

# TorchEBM API Reference

Welcome to the TorchEBM API reference documentation. This section provides detailed information about the classes and functions available in TorchEBM.

## Package Structure

TorchEBM is organized into several modules:

<div class="grid cards" markdown>

-   [:material-cube-outline:{ .lg .middle } Core](./torchebm/core)    
-   [:material-database-search:{ .lg .middle } Datasets](./torchebm/datasets)    
-   [:material-chart-scatter-plot:{ .lg .middle } Samplers](./torchebm/samplers)
-   [:material-calculator-variant:{ .lg .middle } Losses](./torchebm/losses)
-   [:octicons-arrow-right-24:{ .lg .middle } Utils](./torchebm/utils)
-   [:material-rocket-launch:{ .lg .middle } CUDA](./torchebm/cuda)

</div>

## Getting Started with the API

If you're new to TorchEBM, we recommend starting with the following classes:

- [`BaseModel`](./torchebm/core/base_model/classes/BaseModel.md): Base class for all models
- [`BaseSampler`](./torchebm/core/basesampler/classes/BaseSampler): Base class for all sampling algorithms
- [`LangevinDynamics`](./torchebm/samplers/langevin_dynamics/classes/LangevinDynamics): Implementation of Langevin dynamics sampling

## Core Components

### Models

TorchEBM provides various built-in models:

| Model | Description |
| --------------- | ----------- |
| [`GaussianModel`](./torchebm/core/base_model/classes/GaussianModel.md) | Multivariate Gaussian energy function |
| [`DoubleWellModel`](./torchebm/core/base_model/classes/DoubleWellModel.md) | Double well potential energy function |
| [`RastriginModel`](./torchebm/core/base_model/classes/RastriginModel.md) | Rastrigin function for testing optimization algorithms |
| [`RosenbrockModel`](./torchebm/core/base_model/classes/RosenbrockModel.md) | Rosenbrock function (banana function) |
| [`AckleyModel`](./torchebm/core/base_model/classes/AckleyModel.md) | Ackley function, a multimodal test function |
| [`HarmonicModel`](./torchebm/core/base_model/classes/HarmonicModel.md) | Harmonic oscillator energy function |

### Samplers

Available sampling algorithms:

| Sampler | Description |
| ------- | ----------- |
| [`LangevinDynamics`](./torchebm/samplers/langevin_dynamics/classes/LangevinDynamics.md) | Langevin dynamics sampling algorithm |
| [`HamiltonianMonteCarlo`](./torchebm/samplers/mcmc/classes/HamiltonianMonteCarlo.md) | Hamiltonian Monte Carlo sampling |

### BaseLoss Functions

TorchEBM implements several loss functions for training EBMs:

| BaseLoss Function | Description |
| ------------- | ----------- |
| [`ContrastiveDivergence`](./torchebm/losses/contrastive_divergence/classes/ContrastiveDivergence.md) | Standard contrastive divergence (CD-k) |
| [`PersistentContrastiveDivergence`](./torchebm/losses/contrastive_divergence/classes/PersistentContrastiveDivergence.md) | Persistent contrastive divergence |
| [`ParallelTemperingCD`](./torchebm/losses/contrastive_divergence/classes/ParallelTemperingCD.md) | Parallel tempering contrastive divergence |

## Module Details

For detailed information about each module, follow the links below:

- [Core Module](./torchebm/core)
- [Samplers](./torchebm/samplers)
- [Losses](./torchebm/losses)
- [Utils](./torchebm/utils)
- [CUDA](./torchebm/cuda)

