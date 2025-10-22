---
title: Tutorials
description: In-depth tutorials for using TorchEBM
icon: material/school
---

# Getting Started with TorchEBM

Welcome to the TorchEBM tutorials section! These comprehensive tutorials will help you understand how to use TorchEBM effectively for your energy-based modeling tasks.

## Core Concepts

<div class="grid cards" markdown>

-   [:material-function-variant:{ .lg .middle } Energy Models](../api/torchebm/core/base_model/)
-   [:material-chart-scatter-plot:{ .lg .middle } Samplers](samplers.md)
-   [:material-calculator-variant:{ .lg .middle } Loss Functions](loss_functions.md)
-   [:material-school:{ .lg .middle } Training](training.md)
-   [:material-chart-bar:{ .lg .middle } Visualization](visualization.md)

</div>

## Quick Start

If you're new to energy-based models, we recommend the following learning path:

1. Follow the [Installation/Introduction](getting_started.md) guide to set up TorchEBM and understand basic concepts
2. Read the [Energy Models API](../api/torchebm/core/base_model/) to understand model implementations
3. Explore the [Samplers](samplers.md) guide to learn how to generate samples
4. Study the [Training](training.md) guide to learn how to train your models

## Basic Example

Here's a simple example to get you started with TorchEBM:

```python
import torch
from torchebm.core import GaussianModel
from torchebm.samplers.langevin_dynamics import LangevinDynamics

# Create an energy model (2D Gaussian)
energy_fn = GaussianModel(
    mean=torch.zeros(2),
    cov=torch.eye(2)
)

# Create a sampler
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.01
)

# Generate samples
samples = sampler.sample(
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

### Energy Model Definition

```python
from torchebm.core import BaseModel
import torch


class MyEnergyModel(BaseModel):
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

samples = sampler.sample(
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