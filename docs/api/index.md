---
title: API Reference
description: Detailed API reference for TorchEBM
icon: material/file-document
---

# API Reference

## Package Structure

<div class="grid cards" markdown>

-   [:material-cube-outline:{ .lg .middle } **Core**](./torchebm/core)

    Base classes, energy models, schedulers, and device management.

-   [:material-chart-scatter-plot:{ .lg .middle } **Samplers**](./torchebm/samplers)

    MCMC, optimization, and flow/diffusion sampling.

-   [:material-calculator-variant:{ .lg .middle } **Losses**](./torchebm/losses)

    Training objectives for energy-based and flow-based models.

-   [:material-sine-wave:{ .lg .middle } **Interpolants**](./torchebm/interpolants)

    Noise-to-data interpolation schemes.

-   [:material-math-integral:{ .lg .middle } **Integrators**](./torchebm/integrators)

    Numerical integrators for SDE, ODE, and Hamiltonian dynamics.

-   [:material-brain:{ .lg .middle } **Models**](./torchebm/models)

    Neural network architectures.

-   [:material-database-search:{ .lg .middle } **Datasets**](./torchebm/datasets)

    Synthetic data generators.

-   [:octicons-tools-24:{ .lg .middle } **Utils**](./torchebm/utils)

    Visualization and training utilities.

-   [:material-rocket-launch:{ .lg .middle } **CUDA**](./torchebm/cuda)

    Accelerated implementations.

</div>

Each module page lists every public class and function with full signatures, parameters, and usage notes. Start from the module that matches your task, or browse the base classes in [Core](./torchebm/core) to understand the shared interfaces.

