---
sidebar_position: 6
title: Custom Neural Networks
description: Creating custom neural network-based energy functions
---

# Custom Neural Network Energy Functions

Energy-based models (EBMs) are extremely flexible, and one of their key advantages is that the energy function can be parameterized using neural networks. This guide explains how to create and use neural network-based energy functions in TorchEBM.

## Overview

Neural networks provide a powerful way to represent complex energy landscapes that can't be easily defined analytically. By using neural networks as energy functions:

- You can capture complex, high-dimensional distributions
- The energy function can be learned from data
- You gain the expressivity of modern deep learning architectures

## Basic Neural Network Energy Function

To create a neural network-based energy function in TorchEBM, you need to subclass the `BaseEnergyFunction` base class and implement the `forward` method:

```python
import torch
import torch.nn as nn
from torchebm.core import BaseEnergyFunction


class NeuralNetEnergyFunction(BaseEnergyFunction):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()

        # Define neural network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x has shape (batch_size, input_dim)
        # Output should have shape (batch_size,)
        return self.network(x).squeeze(-1)
```

## Design Considerations

When designing neural network energy functions, consider the following:

### Network Architecture

The choice of architecture depends on the data type and complexity:

- **MLPs**: Good for generic, low-dimensional data
- **CNNs**: Effective for images and data with spatial structure
- **Transformers**: Useful for sequential data or when attention mechanisms are beneficial
- **Graph Neural Networks**: For data with graph structure

### Output Requirements

Remember the following key points:

1. The energy function should output a scalar value for each sample in the batch
2. Lower energy values should correspond to higher probability density
3. The neural network must be differentiable for gradient-based sampling methods to work

### Scale and Normalization

Energy values should be properly scaled to avoid numerical issues:

- Very large energy values can cause instability in sampling
- Energy functions that grow too quickly may cause sampling algorithms to fail

## Example: MLP Energy Function for 2D Data

Here's a complete example with a simple MLP energy function:

```python
import torch
import torch.nn as nn
from torchebm.core import BaseEnergyFunction
from torchebm.samplers.langevin_dynamics import LangevinDynamics
import matplotlib.pyplot as plt
import numpy as np


class MLPEnergyFunction(BaseEnergyFunction):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()

        # Define neural network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize with small weights
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Ensure x is batched
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # Forward pass through network
        return self.network(x).squeeze(-1)


# Create the energy function
energy_fn = MLPEnergyFunction()


# Define parameters we want the network to learn
# Let's create a "four peaks" energy landscape
def target_energy(x, y):
    return -2.0 * torch.exp(-0.2 * ((x - 2) ** 2 + (y - 2) ** 2))
        - 3.0 * torch.exp(-0.2 * ((x + 2) ** 2 + (y - 2) ** 2))
        - 1.0 * torch.exp(-0.3 * ((x - 2) ** 2 + (y + 2) ** 2))
        - 4.0 * torch.exp(-0.2 * ((x + 2) ** 2 + (y + 2) ** 2))
        + 0.1 * (x ** 2 + y ** 2)


# Generate training data from the target distribution
def generate_training_data(n_samples=10000):
    # Sample uniformly from a grid
    x = torch.linspace(-4, 4, 100)
    y = torch.linspace(-4, 4, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    positions = torch.stack([X.flatten(), Y.flatten()], dim=1)

    # Calculate target energy
    energies = target_energy(positions[:, 0], positions[:, 1])

    # Convert energies to probabilities (unnormalized)
    probs = torch.exp(-energies)

    # Normalize to create a distribution
    probs = probs / probs.sum()

    # Sample indices based on probability
    indices = torch.multinomial(probs, n_samples, replacement=True)

    # Return sampled positions
    return positions[indices]


# Generate training data
train_data = generate_training_data(10000)

# Set up optimizer
optimizer = torch.optim.Adam(energy_fn.parameters(), lr=0.001)

# Training loop
n_epochs = 1000
batch_size = 128

for epoch in range(n_epochs):
    # Generate random noise samples for contrastive divergence
    noise_samples = torch.randn_like(train_data)

    # Shuffle data
    indices = torch.randperm(train_data.shape[0])

    # Mini-batch training
    for start_idx in range(0, train_data.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, train_data.shape[0])
        batch_indices = indices[start_idx:end_idx]

        data_batch = train_data[batch_indices]
        noise_batch = noise_samples[batch_indices]

        # Zero gradients
        optimizer.zero_grad()

        # Calculate energy for data and noise samples
        data_energy = energy_fn(data_batch)
        noise_energy = energy_fn(noise_batch)

        # Contrastive divergence loss: make data energy lower, noise energy higher
        loss = data_energy.mean() - noise_energy.mean()

        # Backpropagation
        loss.backward()
        optimizer.step()

    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{n_epochs}, BaseLoss: {loss.item():.4f}')


# Visualize learned energy function
def visualize_energy_function(energy_fn, title="Learned Energy Function"):
    x = torch.linspace(-4, 4, 100)
    y = torch.linspace(-4, 4, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    positions = torch.stack([X.flatten(), Y.flatten()], dim=1)

    # Calculate energies
    with torch.no_grad():
        energies = energy_fn(positions).reshape(100, 100)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(X.numpy(), Y.numpy(), energies.numpy(), 50, cmap='viridis')
    plt.colorbar(label='Energy')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()


# Visualize the learned energy function
visualize_energy_function(energy_fn)

# Sample from the learned energy function using Langevin dynamics
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.01
)

samples = sampler.sample_chain(
    dim=2,
    n_steps=1000,
    n_samples=2000,
    burn_in=200
)

# Visualize samples
plt.figure(figsize=(10, 8))
plt.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=1, alpha=0.5)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.title('Samples from Learned Energy Function')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
```

## Example: Convolutional Energy Function for Images

For image data, convolutional architectures are more appropriate:

```python
import torch
import torch.nn as nn
from torchebm.core import BaseEnergyFunction


class ConvolutionalEnergyFunction(BaseEnergyFunction):
    def __init__(self, channels=1, width=28, height=28):
        super().__init__()

        # Convolutional feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 4x4
            nn.LeakyReLU(0.2),
        )

        # Calculate the size of the flattened features
        feature_size = 128 * (width // 8) * (height // 8)

        # Final energy output
        self.energy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Ensure x is batched and has correct channel dimension
        if x.ndim == 3:  # Single image with channels
            x = x.unsqueeze(0)
        elif x.ndim == 2:  # Single grayscale image
            x = x.unsqueeze(0).unsqueeze(0)

        # Extract features and compute energy
        features = self.feature_extractor(x)
        energy = self.energy_head(features).squeeze(-1)

        return energy
```

## Advanced Pattern: Hybrid Energy Functions

You can combine analytical energy functions with neural networks for best of both worlds:

```python
import torch
import torch.nn as nn
from torchebm.core import BaseEnergyFunction, GaussianEnergy


class HybridEnergyFunction(BaseEnergyFunction):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()

        # Analytical component: Gaussian energy
        self.analytical_component = GaussianEnergy(
            mean=torch.zeros(input_dim),
            cov=torch.eye(input_dim)
        )

        # Neural network component
        self.neural_component = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # Weight for combining components
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # Analytical energy
        analytical_energy = self.analytical_component(x)

        # Neural network energy
        neural_energy = self.neural_component(x).squeeze(-1)

        # Combine using learned weight
        # Use sigmoid to keep alpha between 0 and 1
        alpha = torch.sigmoid(self.alpha)
        combined_energy = alpha * analytical_energy + (1 - alpha) * neural_energy

        return combined_energy
```

## Training Strategies

Training neural network energy functions requires special techniques:

### Contrastive Divergence

A common approach is contrastive divergence, which minimizes the energy of data samples while maximizing the energy of samples from the model:

```python
def train_step_contrastive_divergence(energy_fn, optimizer, data_batch, sampler, n_sampling_steps=10):
    # Zero gradients
    optimizer.zero_grad()
    
    # Data energy
    data_energy = energy_fn(data_batch)
    
    # Generate negative samples (model samples)
    with torch.no_grad():
        # Start from random noise
        model_samples = torch.randn_like(data_batch)
        
        # Run MCMC for a few steps
        model_samples = sampler.sample_chain(
            initial_points=model_samples,
            n_steps=n_sampling_steps,
            return_final=True
        )
    
    # Model energy
    model_energy = energy_fn(model_samples)
    
    # BaseLoss: make data energy lower, model energy higher
    loss = data_energy.mean() - model_energy.mean()
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

### Score Matching

Score matching is another approach that avoids the need for MCMC sampling:

```python
def score_matching_loss(energy_fn, data_batch, noise_scale=0.01):
    # Add noise to data
    data_batch.requires_grad_(True)
    
    # Compute energy
    energy = energy_fn(data_batch)
    
    # Compute gradients w.r.t. inputs
    grad_energy = torch.autograd.grad(
        outputs=energy.sum(),
        inputs=data_batch,
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Compute score matching loss
    loss = 0.5 * (grad_energy ** 2).sum(dim=1).mean()
    
    # Add regularization term
    noise_data = data_batch + noise_scale * torch.randn_like(data_batch)
    noise_energy = energy_fn(noise_data)
    reg_loss = ((noise_energy - energy) ** 2).mean()
    
    return loss + 0.1 * reg_loss
```

## Tips for Neural Network Energy Functions

1. **Start Simple**: Begin with a simple architecture and gradually increase complexity
2. **Regularization**: Use weight decay or spectral normalization to prevent extreme energy values
3. **Gradient Clipping**: Apply gradient clipping during training to prevent instability
4. **Initialization**: Careful initialization of weights can help convergence
5. **Monitoring**: Track energy values during training to ensure they stay in a reasonable range
6. **Batch Normalization**: Use with caution as it can affect the shape of the energy landscape
7. **Residual Connections**: Can help with gradient flow in deeper networks

## Conclusion

Neural network energy functions provide a powerful way to model complex distributions in energy-based models. By leveraging the flexibility of deep learning architectures, you can create expressive energy functions that capture intricate patterns in your data.

Remember to carefully design your architecture, choose appropriate training methods, and monitor the behavior of your energy function during training and sampling. 