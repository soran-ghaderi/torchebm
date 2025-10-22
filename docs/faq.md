---
sidebar_position: 6
title: Frequently Asked Questions
description: Answers to common questions about TorchEBM.
icon: material/help-circle-outline
---

# Frequently Asked Questions

This page provides answers to frequently asked questions about TorchEBM. If you have a question that is not answered here, please feel free to [open an issue](https://github.com/soran-ghaderi/torchebm/issues) on our GitHub repository.

## General

??? info "What is TorchEBM?"
    TorchEBM is a PyTorch-based library for Energy-Based Models (EBMs). It provides efficient, scalable, and CUDA-accelerated implementations of sampling, inference, and learning algorithms for EBMs.

??? abstract "How does TorchEBM differ from other generative modeling libraries?"
    TorchEBM specializes in energy-based models, which offer flexibility in modeling complex data distributions without requiring a normalized probability function. Unlike libraries for GANs or VAEs, TorchEBM is built around the energy function formulation and leverages MCMC-based sampling techniques.

??? question "What can I use TorchEBM for?"
    TorchEBM is suitable for a wide range of tasks, including:
    
    * Generative modeling and density estimation
    * Unsupervised representation learning
    * Outlier and anomaly detection
    * Exploring complex, high-dimensional energy landscapes
    * Applications in scientific simulation and statistical physics

## Installation & Setup

??? tip "What are the system requirements for TorchEBM?"
    * Python 3.8 or newer
    * PyTorch 1.10.0 or newer
    * CUDA (optional, but highly recommended for performance)

??? info "Does TorchEBM work on CPU-only machines?"
    Yes, TorchEBM is fully functional on CPU-only machines. However, for optimal performance, especially with large models and datasets, a GPU with CUDA support is recommended.

??? note "How do I install TorchEBM with CUDA support?"
    First, ensure you have installed CUDA drivers and a version of PyTorch installed that supports your CUDA toolkit. Then, you can install TorchEBM via pip:

    ```bash
    pip install torchebm
    ```

## Technical

??? warning "How do I diagnose sampling problems?"
    Common issues and potential solutions:

    * **Poor Mixing**: Try increasing the step size, using more sampling steps, or switching to a more advanced sampler.
    * **Numerical Instability**: Decrease the step size or check for numerical issues in your energy function.
    * **Mode Collapse**: Your energy function may be too simple, or you might need a sampler with better exploration capabilities.

??? tip "How do I train an energy-based model?"
    The basic training loop for an EBM involves these steps:

    1. **Define an Energy Function**: Typically a neural network that maps inputs to a scalar energy value.
    2. **Choose a Loss Function**: Such as contrastive divergence or maximum likelihood estimation.
    3. **Set Up a Sampler**: To generate negative samples from the model's distribution.
    4. **Train**: Use gradient descent to minimize the loss function.
    5. **Evaluate**: Assess the model's performance on a validation set.

    For a practical guide, see the [training examples](../examples/training/).

## Performance

??? tip "How can I speed up sampling?"
    To improve sampling performance:

    * **Use a GPU**: This is the most effective way to accelerate sampling.
    * **Parallelize**: Run multiple sampling chains in parallel.
    * **Tune Hyperparameters**: Optimize sampler-specific parameters like step size.
    * **Choose the Right Algorithm**: Some samplers are better suited for specific energy landscapes.

??? info "Does TorchEBM support distributed training?"
    Currently, TorchEBM is optimized for single-machine, multi-GPU training. Full distributed training support across multiple machines is on our roadmap.

## Contributing

??? note "How can I contribute to TorchEBM?"
    We welcome contributions! You can:

    * **Report Bugs**: Open an issue on our [GitHub repository](https://github.com/soran-ghaderi/torchebm/issues).
    * **Suggest Features**: Let us know what you'd like to see in future versions.
    * **Contribute Code**: Check out our [contributing guidelines](../developer_guide/getting_started/) to get started.

??? bug "I found a bug, how do I report it?"
    Please open an issue on our [GitHub repository](https://github.com/soran-ghaderi/torchebm/issues) and provide:

    * A clear description of the problem.
    * Steps to reproduce the issue.
    * The expected versus actual behavior.
    * Your TorchEBM, PyTorch, Python, and CUDA versions.

??? tip "Can I add my own sampler or energy function?"
    Absolutely! TorchEBM is designed to be extensible. See our guides on:

    * [Custom Energy Models](./api/torchebm/core/base_model/classes/BaseModel.md)
    * [Implementing Custom Samplers](./tutorials/samplers.md#implementing-custom-samplers)

## Future Development

??? info "What features are planned for future releases?"
    See our [Roadmap](./index.md#roadmap--features) for planned features. We're always working on adding:

    * Additional samplers and energy functions
    * More loss functions for training
    * Improved visualization tools
    * Advanced neural network architectures

??? warning "How stable is the TorchEBM API?"
    TorchEBM is in active development, so the API may evolve. We adhere to semantic versioning and will document any breaking changes in the release notes. 