---
sidebar_position: 2
title: Energy Functions
description: Understanding and using energy functions in TorchEBM
---

# Energy Functions

Energy functions are the core component of Energy-Based Models. In TorchEBM, energy functions define the probability distribution from which we sample and learn.

## Built-in Energy Functions

TorchEBM provides several built-in energy functions for common use cases:

### Gaussian Energy

The multivariate Gaussian energy function defines a normal distribution:

```python
from torchebm.core import GaussianEnergy
import torch

# Standard Gaussian
gaussian = GaussianEnergy(
    mean=torch.zeros(2),
    cov=torch.eye(2)
)

# Custom mean and covariance
custom_mean = torch.tensor([1.0, -1.0])
custom_cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])
custom_gaussian = GaussianEnergy(
    mean=custom_mean,
    cov=custom_cov
)
```

### Double Well Energy

The double well potential has two local minima separated by a barrier:

```python
from torchebm.core import DoubleWellEnergy

# Default barrier height = 2.0
double_well = DoubleWellEnergy()

# Custom barrier height
custom_double_well = DoubleWellEnergy(barrier_height=5.0)
```

### Rosenbrock Energy

The Rosenbrock function has a narrow, curved valley with a global minimum:

```python
from torchebm.core import RosenbrockEnergy

# Default parameters a=1.0, b=100.0
rosenbrock = RosenbrockEnergy()

# Custom parameters
custom_rosenbrock = RosenbrockEnergy(a=2.0, b=50.0)
```

### Rastrigin Energy

The Rastrigin function has many local minima arranged in a regular pattern:

```python
from torchebm.core import RastriginEnergy

rastrigin = RastriginEnergy()
```

### Ackley Energy

The Ackley function has many local minima with a single global minimum:

```python
from torchebm.core import AckleyEnergy

ackley = AckleyEnergy()
```

## Using Energy Functions

Energy functions in TorchEBM implement these key methods:

### Energy Calculation

Calculate the energy of a batch of samples:

```python
# x shape: [batch_size, dimension]
energy_values = energy_function(x)  # returns [batch_size]
```

### Gradient Calculation

Calculate the gradient of the energy with respect to the input:

```python
# Requires grad enabled
x.requires_grad_(True)
energy_values = energy_function(x)


# Calculate gradients
gradients = torch.autograd.grad(
    energy_values.sum(), x, create_graph=True
)[0]  # shape: [batch_size, dimension]
```

### Device Management

Energy functions can be moved between devices:

```python
# Move to GPU
energy_function = energy_function.to("cuda")

# Move to CPU
energy_function = energy_function.to("cpu")
```

## Creating Custom Energy Functions

You can create custom energy functions by subclassing the `BaseEnergyFunction` base class:

```python
from torchebm.core import BaseEnergyFunction
import torch


class MyCustomEnergy(BaseEnergyFunction):
    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def forward(self, x):
        # Implement your energy function here
        # x shape: [batch_size, dimension]
        # Return shape: [batch_size]
        return torch.sum(self.param1 * x ** 2 + self.param2 * torch.sin(x), dim=-1)
```

## Neural Network Energy Functions

For more complex energy functions, you can use neural networks:

```python
import torch.nn as nn
from torchebm.core import BaseEnergyFunction


class NeuralNetworkEnergy(BaseEnergyFunction):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, input_dim]
        return self.network(x).squeeze(-1)  # Return shape: [batch_size]
```

## Best Practices

1. **Numerical Stability**: Avoid energy functions that can produce NaN or Inf values
2. **Scaling**: Keep energy values within a reasonable range to avoid numerical issues
3. **Conditioning**: Well-conditioned energy functions are easier to sample from
4. **Gradients**: Ensure your energy function has well-behaved gradients
5. **Batching**: Implement energy functions to efficiently handle batched inputs 