---
sidebar_position: 1
title: Getting Started
description: A hands-on guide to installing TorchEBM and training your first Energy-Based Model.
---

# Getting Started with TorchEBM

This guide provides a hands-on introduction to TorchEBM. You'll learn how to install the library, understand its core components, and train your first Energy-Based Model (EBM) on a synthetic dataset.

## Installation

TorchEBM can be installed from PyPI. Ensure you have PyTorch installed first.

```bash
pip install torchebm
```

!!! tip "Prerequisites"

    - Python 3.8+
    - PyTorch 1.10.0+
    - CUDA is optional but highly recommended for performance.

## Training Your First EBM

Let's train an EBM with a neural network to learn the distribution of a synthetic "two moons" dataset.

### Step 1: Create a Dataset

First, we'll generate a `TwoMoonsDataset` and create a `DataLoader` to iterate through it in batches.

```python
import torch
from torch.utils.data import DataLoader
from torchebm.datasets import TwoMoonsDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = TwoMoonsDataset(n_samples=5000)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
```

### Step 2: Define the Model

Next, we'll create a simple Multi-Layer Perceptron (MLP) to serve as our model. This network will take 2D points as input and output a single energy value for each. In TorchEBM, models inherit from `BaseModel`.

```python
import torch.nn as nn
from torchebm.core import BaseModel

class NeuralEnergyModel(BaseModel):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

model = NeuralEnergyModel().to(device)
```

### Step 3: Set up the Training Components

To train the EBM, we need three things: a loss function, a sampler, and an optimizer.

1.  **A Sampler**: We'll use `LangevinDynamics` to generate negative samples required by the loss function.
2.  **A Loss Function**: We'll use `ContrastiveDivergence`, a standard loss function for EBMs. It works by pushing down the energy of real data ("positive" samples) and pushing up the energy of generated data ("negative" samples).
3.  **An Optimizer**: A standard PyTorch optimizer like `Adam`.

```python
from torchebm.losses import ContrastiveDivergence
from torchebm.samplers import LangevinDynamics
from torch.optim import Adam

sampler = LangevinDynamics(
    model=model,
    step_size=10.0,
    noise_scale=0.1,
    n_steps=60
)

cd_loss = ContrastiveDivergence(sampler=sampler)
optimizer = Adam(model.parameters(), lr=1e-4)
```

### Step 4: The Training Loop

Now we'll write a standard PyTorch training loop. For each batch of real data, we calculate the contrastive divergence loss and update the model's weights.

```python
for epoch in range(100):
    for batch_data in dataloader:
        real_samples = batch_data.to(device)

        optimizer.zero_grad()
        loss, _ = cd_loss(real_samples)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Training finished!")
```
This loop adjusts the weights of our neural network so that its energy landscape matches the "two moons" data distribution.

## Next Steps

Congratulations on training your first Energy-Based Model with TorchEBM!

-   Learn to create [Custom Neural Networks](custom_neural_networks.md) for more complex energy functions.
-   Explore the different [Samplers](samplers.md) available.
-   Discover other [Loss Functions](loss_functions.md) for training EBMs.
-   Dive into the complete [Training](training.md) process.
-   Check out the [Visualization](visualization.md) guide to see how you can plot your energy landscapes and samples.
-   For advanced use-cases, see the guide on [Parallel Sampling](parallel_sampling.md).

