---
title: Models and Energy Functions
description: Learn how to define and use energy-based models in TorchEBM.
icon: material/function-variant
---

# Models and Energy Functions

At the core of any energy-based model (EBM) is the **energy function**, \( E_{\theta}(x) \), which assigns a scalar energy value to each data point \( x \). This function is used to define a probability distribution \( p_{\theta}(x) = \frac{e^{-E_{\theta}(x)}}{Z(\theta)} \), where regions of low energy correspond to high probability.

In TorchEBM, all energy functions are implemented as `torch.nn.Module` subclasses that inherit from the `torchebm.core.BaseModel` class.

## Defining a Custom Model

You can create a custom energy function by subclassing `BaseModel` and implementing the `forward()` method. Here is an example of a simple energy function based on a Multi-Layer Perceptron (MLP).

```python
import torch
import torch.nn as nn
from torchebm.core import BaseModel

class MLPModel(BaseModel):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPModel(input_dim=2).to(device)
print(model)
```

## Built-in Analytical Models

TorchEBM also provides several pre-built analytical models for common distributions and testing scenarios. These are useful for research and for understanding the behavior of samplers and training algorithms.

### GaussianModel

This model implements the energy function for a multivariate Gaussian distribution.

\[ E(x) = \frac{1}{2} (x - \mu)^{\top} \Sigma^{-1} (x - \mu) \]

```python
import torch
from torchebm.core import GaussianModel

mean = torch.tensor([0.0, 0.0])
covariance = torch.eye(2)
gaussian_model = GaussianModel(mean, covariance)
```

### DoubleWellModel

This model creates a double-well potential, which is useful for testing a sampler's ability to cross energy barriers.

\[ E(x) = h \sum_{i=1}^{n} (x_i^2 - b^2)^2 \]

```python
import torch
from torchebm.core import DoubleWellModel

double_well_model = DoubleWellModel(barrier_height=2.0)
```

## Visualizing Energy Landscapes

Understanding the shape of the energy landscape is crucial. Here's how you can visualize the 2D landscape of the `DoubleWellModel`.

```python
import numpy as np
import matplotlib.pyplot as plt

model = DoubleWellModel(barrier_height=2.0)

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)

with torch.no_grad():
    energy_values = model(grid_points).numpy().reshape(X.shape)

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, energy_values, levels=50, cmap='viridis')
plt.colorbar(label='Energy')
plt.title('Energy Landscape of DoubleWellModel')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()
```

<figure markdown>
  ![Double Well Energy Function](../../assets/images/e_functions/double_well.png){ width="500" }
  <figcaption>The `DoubleWellModel` has two low-energy regions (wells) separated by a high-energy barrier.</figcaption>
</figure>

TorchEBM includes a variety of other analytical models such as `RosenbrockModel`, `AckleyModel`, and `RastriginModel` which are commonly used for benchmarking optimization and sampling algorithms. You can visualize them using the same technique.