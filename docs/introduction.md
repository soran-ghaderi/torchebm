---
sidebar_position: 1
title: Introduction
description: Introduction to Energy-Based Models and TorchEBM
---

# Introduction to TorchEBM

TorchEBM is a CUDA-accelerated library for Energy-Based Models (EBMs) built on PyTorch. It provides efficient implementations of sampling, inference, and learning algorithms for EBMs, with a focus on scalability and performance.

## What are Energy-Based Models?

Energy-Based Models (EBMs) are a class of machine learning models that define a probability distribution through an energy function. The energy function assigns a scalar energy value to each configuration of the variables of interest, with lower energy values indicating more probable configurations.

The probability density of a configuration x is proportional to the negative exponential of its energy:

$$p(x) = \frac{e^{-E(x)}}{Z}$$

where:

- $E(x)$ is the energy function
- $Z = \int e^{-E(x)} dx$ is the normalizing constant (partition function)

EBMs are powerful because they can model complex dependencies between variables and capture multimodal distributions. They are applicable to a wide range of tasks including generative modeling, density estimation, and representation learning.

## Why TorchEBM?

While Energy-Based Models are powerful, they present several challenges:

- The partition function Z is often intractable to compute directly
- Sampling from EBMs requires advanced Markov Chain Monte Carlo (MCMC) methods
- Training can be computationally intensive

TorchEBM addresses these challenges by providing:

- **Efficient samplers**: CUDA-accelerated implementations of MCMC samplers like Langevin Dynamics and Hamiltonian Monte Carlo
- **Training methods**: Implementations of contrastive divergence and other specialized loss functions
- **Integration with PyTorch**: Seamless compatibility with the PyTorch ecosystem

## Key Concepts

### Energy Functions

Energy functions are the core component of EBMs. TorchEBM provides implementations of common energy functions like Gaussian, Double Well, and Rosenbrock, as well as a base class for creating custom energy functions.

### Samplers

Sampling from EBMs typically involves MCMC methods. TorchEBM implements several sampling algorithms:

- **Langevin Dynamics**: Updates samples using gradient information plus noise
- **Hamiltonian Monte Carlo**: Uses Hamiltonian dynamics for efficient exploration
- **Other samplers**: Various specialized samplers for different applications

### BaseLoss Functions

Training EBMs requires specialized methods to estimate and minimize the difference between the model distribution and the data distribution. TorchEBM implements several loss functions including contrastive divergence and score matching techniques.

## Applications

Energy-Based Models and TorchEBM can be applied to various tasks:

- Generative modeling
- Density estimation
- Unsupervised representation learning
- Out-of-distribution detection
- Structured prediction

## Next Steps

- Follow the [Getting Started](./getting_started.md) guide to install TorchEBM and run your first examples
- Check the [Guides](./guides/index.md) for more detailed information on specific components
- Explore the [Examples](./examples/index.md) for practical applications 