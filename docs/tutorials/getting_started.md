---
sidebar_position: 2
title: Getting Started
description: A hands-on guide to installing TorchEBM and training your first Energy-Based Model.
icon: material/rocket-launch
---

# Getting Started with TorchEBM

This guide provides a hands-on introduction to TorchEBM. You'll learn how to install the library, understand its core components, and train your first Energy-Based Model (EBM) on a synthetic dataset.

## 1. Installation

TorchEBM can be installed from PyPI. Ensure you have PyTorch installed first.

```bash
pip install torchebm
```

!!! tip "Prerequisites"

    - Python 3.8+
    - PyTorch 1.10.0+
    - CUDA is optional but highly recommended for performance.

## 2. The Core Concepts

An Energy-Based Model defines a probability distribution over data \(x\) through an **energy function** \(E(x)\). The probability is defined as \(p(x) = \frac{e^{-E(x)}}{Z}\), where lower energy corresponds to higher probability.

TorchEBM is built around two key components:

1.  **Energy Functions**: These are learnable functions (often neural networks) that map input data to a scalar energy value.
2.  **Samplers**: These are algorithms, typically based on Markov Chain Monte Carlo (MCMC), used to draw samples from the probability distribution defined by the energy function.

Let's explore these concepts with code.

### Concept 1: The Energy Function

An energy function is a `torch.nn.Module` that takes a tensor `x` of shape `(batch_size, *dims)` and returns a tensor of energy values of shape `(batch_size,)`.

TorchEBM provides several pre-built energy functions for testing and experimentation. Here's how to use the `GaussianEnergy` function, which models a multivariate normal distribution.

```python
import torch
from torchebm.core import GaussianEnergy

device = "cuda" if torch.cuda.is_available() else "cpu"

energy_fn = GaussianEnergy(
    mean=torch.zeros(2, device=device),
    cov=torch.eye(2, device=device)
).to(device)

x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], device=device)

energy = energy_fn(x)

print(f"Input shape: {x.shape}")
print(f"Energy shape: {energy.shape}")
print(f"Energies:\n{energy}")
```

The point `[0.0, 0.0]` is the mean of the distribution and thus has the lowest energy. As points move away from the mean, their energy increases.

### Concept 2: The Sampler

Samplers generate data points from the distribution defined by an energy function. They typically work by starting from random initial points and iteratively refining them to have lower energy (higher probability).

Let's use the `LangevinDynamics` sampler to draw samples from our `GaussianEnergy` distribution.

```python
from torchebm.samplers import LangevinDynamics

sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.1,
    noise_scale=1.0
).to(device)

samples = sampler.sample(
    dim=2,
    n_samples=1000,
    n_steps=100
)

print(f"Generated samples shape: {samples.shape}")
```

You have now sampled from your first energy-based model! These samples approximate a 2D Gaussian distribution.

## 3. Training Your First EBM

Now let's put everything together and train an EBM with a neural network as the energy function. The goal is to train the model to represent a synthetic "two moons" dataset.

### Step 1: Create a Dataset

First, we'll generate a `TwoMoonsDataset` and create a `DataLoader` to iterate through it in batches.

```python
import torch
from torch.utils.data import DataLoader
from torchebm.datasets import TwoMoonsDataset

dataset = TwoMoonsDataset(n_samples=5000)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
```

### Step 2: Define a Neural Energy Function

Next, we'll create a simple Multi-Layer Perceptron (MLP) to serve as our energy function. This network will take 2D points as input and output a single energy value for each.

```python
import torch.nn as nn
from torchebm.core import BaseEnergyFunction

class NeuralEnergy(BaseEnergyFunction):
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

neural_energy_fn = NeuralEnergy().to(device)
```

### Step 3: Set up the Training Components

To train the EBM, we need three things:

1.  **A Loss Function**: We'll use `ContrastiveDivergence`, a standard loss function for EBMs. It works by pushing down the energy of real data ("positive" samples) and pushing up the energy of generated data ("negative" samples).
2.  **A Sampler**: The loss function needs a sampler to generate the negative samples. We'll use `LangevinDynamics` again.
3.  **An Optimizer**: A standard PyTorch optimizer like `Adam`.

```python
from torchebm.losses import ContrastiveDivergence
from torchebm.samplers import LangevinDynamics
from torch.optim import Adam

sampler = LangevinDynamics(
    energy_function=neural_energy_fn,
    step_size=10.0,
    noise_scale=0.1,
    n_steps=60
)

cd_loss = ContrastiveDivergence(sampler=sampler)

optimizer = Adam(neural_energy_fn.parameters(), lr=1e-4)
```

### Step 4: The Training Loop

Now we'll write a standard PyTorch training loop. For each batch of real data, we calculate the contrastive divergence loss and update the model's weights.

```python
for epoch in range(100):
    for batch_data in dataloader:
        real_samples = batch_data.to(device)

        optimizer.zero_grad()

        loss = cd_loss(real_samples, neural_energy_fn)

        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Training finished!")
```
This loop adjusts the weights of our neural network so that its energy landscape matches the "two moons" data distribution.

## Next Steps

Congratulations on training your first Energy-Based Model with TorchEBM!

*   Learn more about the different [Samplers](samplers.md) available.
*   Explore other [Loss Functions](loss_functions.md) for training EBMs.
*   See how to create [Custom Neural Networks](custom_neural_networks.md) for more complex energy functions.
*   Check out the [Visualization](visualization.md) guide to see how you can plot your energy landscapes and samples.

