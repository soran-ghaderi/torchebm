---
title: Guides
description: In-depth guides for using TorchEBM
---

# Getting Started with TorchEBM

Welcome to the TorchEBM guides section! These comprehensive guides will help you understand how to use TorchEBM effectively for your energy-based modeling tasks.

## Core Concepts

<div class="grid cards" markdown>

-   :material-function-variant:{ .lg .middle } __Energy Functions__

    ---

    Learn about the foundation of energy-based models and how to work with energy functions in TorchEBM.

    [:octicons-arrow-right-24: Energy Functions Guide](energy_functions.md)

-   :material-chart-scatter-plot:{ .lg .middle } __Samplers__

    ---

    Discover how to generate samples from energy landscapes using various sampling algorithms.

    [:octicons-arrow-right-24: Samplers Guide](samplers.md)

-   :material-calculator-variant:{ .lg .middle } __Loss Functions__

    ---

    Explore different loss functions for training energy-based models.

    [:octicons-arrow-right-24: BaseLoss Functions Guide](loss_functions.md)

-   :material-cube-outline:{ .lg .middle } __Custom Neural Networks__

    ---

    Learn how to create and use neural networks as energy functions.

    [:octicons-arrow-right-24: Custom Neural Networks Guide](custom_neural_networks.md)

-   :material-school:{ .lg .middle } __Training EBMs__

    ---

    Master the techniques for effectively training energy-based models.

    [:octicons-arrow-right-24: Training Guide](training.md)

-   :material-chart-bar:{ .lg .middle } __Visualization__

    ---

    Visualize energy landscapes and sampling results to gain insights.

    [:octicons-arrow-right-24: Visualization Guide](visualization.md)

</div>

## Quick Start

If you're new to energy-based models, we recommend the following learning path:

1. Start with the [Introduction](../introduction.md) to understand basic concepts
2. Follow the [Installation](../getting_started.md) guide to set up TorchEBM
3. Read the [Energy Functions](energy_functions.md) guide to understand the fundamentals
4. Explore the [Samplers](samplers.md) guide to learn how to generate samples
5. Study the [Training](training.md) guide to learn how to train your models

## Basic Example

Here's a simple example to get you started with TorchEBM:

```python
import torch
from torchebm.core import GaussianEnergy
from torchebm.samplers.langevin_dynamics import LangevinDynamics

# Create an energy function (2D Gaussian)
energy_fn = GaussianEnergy(
    mean=torch.zeros(2),
    cov=torch.eye(2)
)

# Create a sampler
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.01
)

# Generate samples
samples = sampler.sample_chain(
    dim=2, n_steps=100, n_samples=1000
)

# Print sample statistics
print(f"Sample mean: {samples.mean(0)}")
print(f"Sample std: {samples.std(0)}")
```

## Common Patterns

Here are some common patterns you'll encounter throughout the guides:

<div class="grid" markdown>
<div markdown>

### Energy Function Definition

```python
from torchebm.core import BaseEnergyFunction
import torch


class MyEnergyFunction(BaseEnergyFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(x ** 2, dim=-1)
```

</div>
<div markdown>

### Sampler Usage

```python
from torchebm.samplers.langevin_dynamics import LangevinDynamics

sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.01
)

samples = sampler.sample_chain(
    dim=2, n_steps=100, n_samples=1000
)
```

</div>
</div>

## Next Steps

Once you're familiar with the basics, you can:

- Explore detailed [Examples](../examples/index.md) that demonstrate TorchEBM in action
- Check the [API Reference](../api/index.md) for comprehensive documentation
- Learn how to contribute to TorchEBM in the [Developer Guide](../developer_guide/index.md)

Remember that all examples in these guides are tested with the latest version of TorchEBM, and you can run them in your own environment to gain hands-on experience. 