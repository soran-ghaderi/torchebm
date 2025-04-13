---
title: API Reference
description: Detailed API reference for TorchEBM
icon: material/api
---

# TorchEBM API Reference

Welcome to the TorchEBM API reference documentation. This section provides detailed information about the classes and functions available in TorchEBM.

## Package Structure

TorchEBM is organized into several modules:

<div class="grid cards" markdown>

-   :material-cube-outline:{ .lg .middle } __Core__

    ---

    Base classes and core functionality for energy functions, samplers, and trainers.

    [:octicons-arrow-right-24: Core Module](./torchebm/core)

-   :material-dice-multiple-outline:{ .lg .middle } __Samplers__

    ---

    Sampling algorithms for energy-based models including Langevin Dynamics and MCMC.

    [:octicons-arrow-right-24: Samplers](./torchebm/samplers)

-   :material-function-variant:{ .lg .middle } __Losses__

    ---

    BaseLoss functions for training energy-based models.

    [:octicons-arrow-right-24: Losses](./torchebm/losses)

-   :material-tools:{ .lg .middle } __Utils__

    ---

    Utility functions for working with energy-based models.

    [:octicons-arrow-right-24: Utils](./torchebm/utils)
    
-   :material-gpu:{ .lg .middle } __CUDA__

    ---

    CUDA-accelerated implementations for faster computation.

    [:octicons-arrow-right-24: CUDA](./torchebm/cuda)

</div>

## Getting Started with the API

If you're new to TorchEBM, we recommend starting with the following classes:

- [`BaseEnergyFunction`](./torchebm/core/energy_function/classes/EnergyFunction): Base class for all energy functions
- [`BaseSampler`](./torchebm/core/basesampler/classes/BaseSampler): Base class for all sampling algorithms
- [`LangevinDynamics`](./torchebm/samplers/langevin_dynamics/classes/LangevinDynamics): Implementation of Langevin dynamics sampling

## Core Components

### Energy Functions

TorchEBM provides various built-in energy functions:

| Energy Function | Description |
| --------------- | ----------- |
| [`GaussianEnergy`](./torchebm/core/energy_function/classes/GaussianEnergy.md) | Multivariate Gaussian energy function |
| [`DoubleWellEnergy`](./torchebm/core/energy_function/classes/DoubleWellEnergy.md) | Double well potential energy function |
| [`RastriginEnergy`](./torchebm/core/energy_function/classes/RastriginEnergy.md) | Rastrigin function for testing optimization algorithms |
| [`RosenbrockEnergy`](./torchebm/core/energy_function/classes/RosenbrockEnergy.md) | Rosenbrock function (banana function) |
| [`AckleyEnergy`](./torchebm/core/energy_function/classes/AckleyEnergy.md) | Ackley function, a multimodal test function |
| [`HarmonicEnergy`](./torchebm/core/energy_function/classes/HarmonicEnergy.md) | Harmonic oscillator energy function |

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
- [Models](./torchebm/models)
- [Utils](./torchebm/utils)
- [CUDA](./torchebm/cuda)

