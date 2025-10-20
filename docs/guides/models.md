---
sidebar_position: 2
title: Models
description: Understanding and using models in TorchEBM
---

# Models

Models are the core component of Energy-Based Models. In TorchEBM, models define the probability distribution from which we sample and learn.

## Built-in Models

TorchEBM provides several built-in models for common use cases:

### Gaussian Model

The multivariate Gaussian model defines a normal distribution:

```python
from torchebm.core import GaussianModel
import torch

# Standard Gaussian
gaussian = GaussianModel(
    mean=torch.zeros(2),
    cov=torch.eye(2)
)

# Custom mean and covariance
custom_mean = torch.tensor([1.0, -1.0])
custom_cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])
custom_gaussian = GaussianModel(
    mean=custom_mean,
    cov=custom_cov
)
```

### Double Well Model

The double well potential has two local minima separated by a barrier:

```python
from torchebm.core import DoubleWellModel

# Default barrier height = 2.0
double_well = DoubleWellModel()

# Custom barrier height
custom_double_well = DoubleWellModel(barrier_height=5.0)
```

### Rosenbrock Model

The Rosenbrock function has a narrow, curved valley with a global minimum:

```python
from torchebm.core import RosenbrockModel

# Default parameters a=1.0, b=100.0
rosenbrock = RosenbrockModel()

# Custom parameters
custom_rosenbrock = RosenbrockModel(a=2.0, b=50.0)
```

### Rastrigin Model

The Rastrigin function has many local minima arranged in a regular pattern:

```python
from torchebm.core import RastriginModel

rastrigin = RastriginModel()
```

### Ackley Model

The Ackley function has many local minima with a single global minimum:

```python
from torchebm.core import AckleyModel

ackley = AckleyModel()
```

## Using Models

Models in TorchEBM implement these key methods:

### Energy Calculation

Calculate the energy of a batch of samples:

```python
# x batch_shape: [batch_size, dimension]
energy_values = model(x)  # returns [batch_size]
```

### Gradient Calculation

Calculate the gradient of the energy with respect to the input:

```python
# Requires grad enabled
x.requires_grad_(True)
energy_values = model(x)


# Calculate gradients
gradients = torch.autograd.grad(
    energy_values.sum(), x, create_graph=True
)[0]  # batch_shape: [batch_size, dimension]
```

### Device Management

Models can be moved between devices:

```python
# Move to GPU
model = model.to("cuda")

# Move to CPU
model = model.to("cpu")
```

## Creating Custom Models

You can create custom models by subclassing the `BaseModel` base class:

```python
from torchebm.core import BaseModel
import torch


class MyCustomModel(BaseModel):
    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def forward(self, x):
        # Implement your model here
        # x batch_shape: [batch_size, dimension]
        # Return batch_shape: [batch_size]
        return torch.sum(self.param1 * x ** 2 + self.param2 * torch.sin(x), dim=-1)
```

## Neural Network Models

For more complex models, you can use neural networks:

```python
import torch.nn as nn
from torchebm.core import BaseModel


class NeuralNetworkModel(BaseModel):
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
        # x batch_shape: [batch_size, input_dim]
        return self.network(x).squeeze(-1)  # Return batch_shape: [batch_size]
```

## Best Practices

1. **Numerical Stability**: Avoid models that can produce NaN or Inf values
2. **Scaling**: Keep energy values within a reasonable range to avoid numerical issues
3. **Conditioning**: Well-conditioned models are easier to sample from
4. **Gradients**: Ensure your model has well-behaved gradients
5. **Batching**: Implement models to efficiently handle batched inputs 