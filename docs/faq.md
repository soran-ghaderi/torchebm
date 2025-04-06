---
sidebar_position: 6
title: Frequently Asked Questions
description: Answers to common questions about TorchEBM
---

# Frequently Asked Questions

## General Questions

### What is TorchEBM?

TorchEBM is a PyTorch-based library for Energy-Based Models (EBMs). It provides efficient implementations of sampling, inference, and learning algorithms for EBMs, with a focus on scalability and performance through CUDA acceleration.

### How does TorchEBM differ from other generative modeling libraries?

TorchEBM specifically focuses on energy-based models, which can model complex distributions without assuming a specific functional form. Unlike libraries for GANs or VAEs, TorchEBM emphasizes the energy function formulation and MCMC sampling techniques.

### What can I use TorchEBM for?

TorchEBM can be used for:

- Generative modeling
- Density estimation
- Unsupervised representation learning
- Outlier detection
- Exploration of complex energy landscapes
- Scientific simulation and statistical physics applications

## Installation & Setup

### What are the system requirements for TorchEBM?

TorchEBM requires:

- Python 3.8 or newer
- PyTorch 1.10.0 or newer
- CUDA (optional, but recommended for performance)

### Does TorchEBM work on CPU-only machines?

Yes, TorchEBM works on CPU-only machines, though many operations will be significantly slower than on GPU.

### How do I install TorchEBM with CUDA support?

Make sure you have PyTorch with CUDA support installed first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install torchebm
```

## Technical Questions

### How do I choose the right energy function for my task?

The choice of energy function depends on:

- The complexity of the distribution you want to model
- Domain knowledge about the structure of your data
- Computational constraints

For complex data like images, neural network-based energy functions are typically used. For simpler problems, analytical energy functions may be sufficient.

### What sampling algorithm should I use?

Common considerations:

- **Langevin Dynamics**: Good for general-purpose sampling, especially in high dimensions
- **Hamiltonian Monte Carlo**: Better for complex energy landscapes, but more computationally expensive
- **Metropolis-Hastings**: Simple to implement, but may mix slowly in high dimensions

### How do I diagnose problems with sampling?

Common issues and solutions:

- **Poor mixing**: Increase step size or try a different sampler
- **Numerical instability**: Decrease step size or check energy function implementation
- **Slow convergence**: Use more iterations or try a more efficient sampler
- **Mode collapse**: Check your energy function or use samplers with better exploration capabilities

### How do I train an energy-based model on my own data?

Basic steps:

1. Define an energy function (e.g., a neural network)
2. Choose a loss function (e.g., contrastive divergence)
3. Set up a sampler for generating negative samples
4. Train using gradient descent
5. Evaluate the learned model

See the [training examples](./examples/training_neural_ebm.md) for more details.

## Performance

### How can I speed up sampling?

To improve sampling performance:

- Use GPU acceleration
- Reduce the dimensionality of your problem
- Parallelize sampling across multiple chains
- Optimize step sizes and other hyperparameters
- Use more efficient sampling algorithms for your specific energy landscape

### Does TorchEBM support distributed training?

Currently, TorchEBM focuses on single-machine GPU acceleration. Distributed training across multiple GPUs or machines is on our roadmap.

### How does TorchEBM's performance compare to other libraries?

TorchEBM is optimized for performance on GPU hardware, particularly for sampling operations. Our benchmarks show significant speedups compared to non-specialized implementations, especially for large-scale sampling tasks.

## Contributing

### How can I contribute to TorchEBM?

We welcome contributions! Check out:

- [GitHub Issues](https://github.com/soran-ghaderi/torchebm/issues) for current tasks
- [Contributing Guidelines](./developer_guide/contributing.md) for code style and contribution workflow

### I found a bug, how do I report it?

Please open an issue on our [GitHub repository](https://github.com/soran-ghaderi/torchebm/issues) with:

- A clear description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- Version information (TorchEBM, PyTorch, Python, CUDA)

### Can I add my own sampler or energy function to TorchEBM?

Absolutely! TorchEBM is designed to be extensible. See:

- [Custom Energy Functions](./guides/energy_functions.md#creating-custom-energy-functions)
- [Implementing Custom Samplers](./guides/samplers.md#implementing-custom-samplers)

## Future Development

### What features are planned for future releases?

See our [Roadmap](./index.md#roadmap--features) for planned features, including:

- Additional samplers and energy functions
- More loss functions for training
- Improved visualization tools
- Advanced neural network architectures
- Better integration with the PyTorch ecosystem

### How stable is the TorchEBM API?

TorchEBM is currently in early development, so the API may change between versions. We'll do our best to document breaking changes and provide migration guidance. 