---
sidebar_position: 7
title: Training EBMs
description: Methods and best practices for training energy-based models
---

# Training Energy-Based Models

This guide covers the fundamental techniques for training energy-based models (EBMs) using TorchEBM. We'll explore various training methods, loss functions, and optimization strategies to help you effectively train your models.

## Overview

Training energy-based models involves estimating the parameters of an energy function such that the corresponding probability distribution matches a target data distribution. Unlike in traditional supervised learning, this is often an unsupervised task where the goal is to learn the underlying structure of the data.

The training process typically involves:

1. Defining an energy function (parameterized by a neural network or analytical form)
2. Choosing a training method and loss function
3. Optimizing the energy function parameters
4. Evaluating the model using sampling and visualization techniques

## Methods for Training EBMs

TorchEBM supports several methods for training EBMs, each with their own advantages and trade-offs:

### Contrastive Divergence (CD)

Contrastive Divergence is one of the most widely used methods for training EBMs. It approximates the gradient of the log-likelihood by comparing data samples to model samples obtained through short MCMC runs.

```python
import torch
from torchebm.core import BaseEnergyFunction, MLPEnergyFunction
from torchebm.samplers.langevin_dynamics import LangevinDynamics

# Create energy function
energy_fn = MLPEnergyFunction(input_dim=2, hidden_dim=64)

# Create sampler for generating negative samples
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.01
)

# Optimizer
optimizer = torch.optim.Adam(energy_fn.parameters(), lr=0.001)


# Contrastive Divergence training step
def train_step_cd(data_batch, k_steps=10):
    optimizer.zero_grad()

    # Positive phase: compute energy of real data
    pos_energy = energy_fn(data_batch)

    # Negative phase: generate samples from current model
    # Start from random noise
    neg_samples = torch.randn_like(data_batch)

    # Sample for n_steps steps
    neg_samples = sampler.sample_chain(
        initial_points=neg_samples,
        n_steps=k_steps,
        return_final=True
    )

    # Compute energy of generated samples
    neg_energy = energy_fn(neg_samples)

    # Compute loss
    # Minimize energy of real data, maximize energy of generated samples
    loss = pos_energy.mean() - neg_energy.mean()

    # Backpropagation
    loss.backward()
    optimizer.step()

    return loss.item(), pos_energy.mean().item(), neg_energy.mean().item()
```

### Persistent Contrastive Divergence (PCD)

PCD improves on standard CD by maintaining a persistent chain of samples that are used across training iterations:

```python
# Initialize persistent samples
persistent_samples = torch.randn(1000, 2)  # Shape: [n_persistent, data_dim]

# PCD training step
def train_step_pcd(data_batch, k_steps=10):
    global persistent_samples
    optimizer.zero_grad()
    
    # Positive phase: compute energy of real data
    pos_energy = energy_fn(data_batch)
    
    # Negative phase: continue sampling from persistent chains
    # Start from existing persistent samples (select a random subset)
    indices = torch.randperm(persistent_samples.shape[0])[:data_batch.shape[0]]
    initial_samples = persistent_samples[indices].clone()
    
    # Sample for n_steps steps
    neg_samples = sampler.sample_chain(
        initial_points=initial_samples,
        n_steps=k_steps,
        return_final=True
    )
    
    # Update persistent samples
    persistent_samples[indices] = neg_samples.detach()
    
    # Compute energy of generated samples
    neg_energy = energy_fn(neg_samples)
    
    # Compute loss
    loss = pos_energy.mean() - neg_energy.mean()
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    return loss.item(), pos_energy.mean().item(), neg_energy.mean().item()
```

### Score Matching

Score Matching avoids the need for MCMC sampling altogether, focusing on matching the gradient of the log-density (score function) instead:

```python
def train_step_score_matching(data_batch, noise_scale=0.01):
    optimizer.zero_grad()
    
    # Ensure data requires gradients
    data_batch.requires_grad_(True)
    
    # Compute energy
    energy = energy_fn(data_batch)
    
    # Compute gradients with respect to inputs
    grad_energy = torch.autograd.grad(
        outputs=energy.sum(),
        inputs=data_batch,
        create_graph=True
    )[0]
    
    # Compute score matching loss
    loss = 0.5 * (grad_energy.pow(2)).sum(dim=1).mean()
    
    # Add regularization (optional)
    noise_data = data_batch + noise_scale * torch.randn_like(data_batch)
    noise_energy = energy_fn(noise_data)
    reg_loss = ((noise_energy - energy) ** 2).mean()
    
    total_loss = loss + 0.1 * reg_loss
    
    # Backpropagation
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()
```

### Denoising Score Matching

Denoising Score Matching is a variant that adds noise to data points and trains the model to predict the score of the noised data:

```python
def train_step_denoising_score_matching(data_batch, sigma=0.1):
    optimizer.zero_grad()
    
    # Add noise to data
    noise = sigma * torch.randn_like(data_batch)
    noised_data = data_batch + noise
    noised_data.requires_grad_(True)
    
    # Compute energy of noised data
    energy = energy_fn(noised_data)
    
    # Compute gradients with respect to noised inputs
    grad_energy = torch.autograd.grad(
        outputs=energy.sum(),
        inputs=noised_data,
        create_graph=True
    )[0]
    
    # Ground truth score is -noise/sigma^2 for Gaussian noise
    target_score = -noise / (sigma**2)
    
    # Compute loss as MSE between predicted and target scores
    loss = 0.5 * ((grad_energy - target_score) ** 2).sum(dim=1).mean()
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

## Full Training Loop Example

Here's a complete example showing how to train an EBM using contrastive divergence:

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchebm.core import BaseEnergyFunction
from torchebm.samplers.langevin_dynamics import LangevinDynamics


# Define a neural network energy function
class MLPEnergyFunction(BaseEnergyFunction):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


# Create a synthetic dataset (mixture of Gaussians)
def generate_data(n_samples=10000):
    # Create a mixture of 5 Gaussians
    centers = [
        (0, 0),
        (2, 2),
        (-2, 2),
        (2, -2),
        (-2, -2)
    ]
    std = 0.3

    # Equal number of samples per component
    n_per_component = n_samples // len(centers)
    data = []

    for center in centers:
        component_data = torch.randn(n_per_component, 2) * std
        component_data[:, 0] += center[0]
        component_data[:, 1] += center[1]
        data.append(component_data)

    # Combine all components
    data = torch.cat(data, dim=0)

    # Shuffle
    indices = torch.randperm(data.shape[0])
    data = data[indices]

    return data


# Generate dataset
dataset = generate_data(10000)

# Create energy function and sampler
energy_fn = MLPEnergyFunction(input_dim=2, hidden_dim=128)
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.01
)

# Optimizer
optimizer = torch.optim.Adam(energy_fn.parameters(), lr=0.0001)

# Training hyperparameters
batch_size = 128
n_epochs = 200
k_steps = 10  # Number of MCMC steps for negative samples
log_interval = 20

# Persistent samples
persistent_samples = torch.randn(1000, 2)

# Training loop
for epoch in range(n_epochs):
    # Shuffle dataset
    indices = torch.randperm(dataset.shape[0])
    dataset = dataset[indices]

    total_loss = 0
    n_batches = 0

    for i in range(0, dataset.shape[0], batch_size):
        # Get batch
        batch_indices = indices[i:i + batch_size]
        batch = dataset[batch_indices]

        # Training step using PCD
        optimizer.zero_grad()

        # Positive phase
        pos_energy = energy_fn(batch)

        # Negative phase with persistent samples
        pcd_indices = torch.randperm(persistent_samples.shape[0])[:batch_size]
        neg_samples_init = persistent_samples[pcd_indices].clone()

        neg_samples = sampler.sample_chain(
            initial_points=neg_samples_init,
            n_steps=k_steps,
            return_final=True
        )

        # Update persistent samples
        persistent_samples[pcd_indices] = neg_samples.detach()

        # Compute energy of negative samples
        neg_energy = energy_fn(neg_samples)

        # Compute loss
        loss = pos_energy.mean() - neg_energy.mean()

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    # Log progress
    avg_loss = total_loss / n_batches

    if (epoch + 1) % log_interval == 0:
        print(f'Epoch {epoch + 1}/{n_epochs}, BaseLoss: {avg_loss:.4f}')

        # Visualize current energy function
        with torch.no_grad():
            # Create grid
            x = torch.linspace(-4, 4, 100)
            y = torch.linspace(-4, 4, 100)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)

            # Compute energy values
            energies = energy_fn(grid_points).reshape(100, 100)

            # Visualize
            plt.figure(figsize=(10, 8))
            plt.contourf(X.numpy(), Y.numpy(), energies.numpy(), 50, cmap='viridis')

            # Plot data points
            plt.scatter(dataset[:500, 0], dataset[:500, 1], s=1, color='red', alpha=0.5)

            # Plot samples from model
            sampled = sampler.sample_chain(
                dim=2,
                n_samples=500,
                n_steps=500,
                burn_in=100
            )
            plt.scatter(sampled[:, 0], sampled[:, 1], s=1, color='white', alpha=0.5)

            plt.title(f'Energy Landscape (Epoch {epoch + 1})')
            plt.xlim(-4, 4)
            plt.ylim(-4, 4)
            plt.colorbar(label='Energy')
            plt.tight_layout()
            plt.show()

# Final evaluation
print("Training complete. Generating samples...")

# Generate final samples
final_samples = sampler.sample_chain(
    dim=2,
    n_samples=5000,
    n_steps=1000,
    burn_in=200
)

# Visualize final distribution
plt.figure(figsize=(12, 5))

# Plot data distribution
plt.subplot(1, 2, 1)
plt.hist2d(dataset[:, 0].numpy(), dataset[:, 1].numpy(), bins=50, cmap='Blues')
plt.title('Data Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Count')

# Plot model distribution
plt.subplot(1, 2, 2)
plt.hist2d(final_samples[:, 0].numpy(), final_samples[:, 1].numpy(), bins=50, cmap='Reds')
plt.title('Model Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Count')

plt.tight_layout()
plt.show()
```

## Training with Regularization

Adding regularization can help stabilize training and prevent the energy function from assigning extremely low values to certain regions:

```python
def train_step_with_regularization(data_batch, k_steps=10, l2_reg=0.001):
    optimizer.zero_grad()
    
    # Positive phase
    pos_energy = energy_fn(data_batch)
    
    # Negative phase
    neg_samples = torch.randn_like(data_batch)
    neg_samples = sampler.sample_chain(
        initial_points=neg_samples,
        n_steps=k_steps,
        return_final=True
    )
    neg_energy = energy_fn(neg_samples)
    
    # Contrastive divergence loss
    cd_loss = pos_energy.mean() - neg_energy.mean()
    
    # L2 regularization on parameters
    l2_norm = sum(p.pow(2).sum() for p in energy_fn.parameters())
    
    # Total loss
    loss = cd_loss + l2_reg * l2_norm
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

## Training with Gradient Clipping

Gradient clipping can prevent explosive gradients during training:

```python
def train_step_with_gradient_clipping(data_batch, k_steps=10, max_norm=1.0):
    optimizer.zero_grad()
    
    # Positive phase
    pos_energy = energy_fn(data_batch)
    
    # Negative phase
    neg_samples = torch.randn_like(data_batch)
    neg_samples = sampler.sample_chain(
        initial_points=neg_samples,
        n_steps=k_steps,
        return_final=True
    )
    neg_energy = energy_fn(neg_samples)
    
    # Contrastive divergence loss
    loss = pos_energy.mean() - neg_energy.mean()
    
    # Backpropagation
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(energy_fn.parameters(), max_norm)
    
    optimizer.step()
    
    return loss.item()
```

## Monitoring Training Progress

It's important to monitor various metrics during training to ensure the model is learning effectively:

```python
def visualize_training_progress(energy_fn, data, sampler, epoch):
    # Generate samples from current model
    samples = sampler.sample_chain(
        dim=data.shape[1],
        n_samples=1000,
        n_steps=500,
        burn_in=100
    )
    
    # Compute energy statistics
    with torch.no_grad():
        data_energy = energy_fn(data[:1000]).mean().item()
        sample_energy = energy_fn(samples).mean().item()
    
    print(f"Epoch {epoch}: Data Energy: {data_energy:.4f}, Sample Energy: {sample_energy:.4f}")
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot data
    plt.subplot(1, 3, 1)
    plt.scatter(data[:1000, 0].numpy(), data[:1000, 1].numpy(), s=2, alpha=0.5)
    plt.title('Data')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    
    # Plot samples
    plt.subplot(1, 3, 2)
    plt.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=2, alpha=0.5)
    plt.title('Samples')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    
    # Plot energy landscape
    plt.subplot(1, 3, 3)
    x = torch.linspace(-4, 4, 100)
    y = torch.linspace(-4, 4, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    with torch.no_grad():
        energies = energy_fn(grid_points).reshape(100, 100)
    
    plt.contourf(X.numpy(), Y.numpy(), energies.numpy(), 50, cmap='viridis')
    plt.colorbar(label='Energy')
    plt.title('Energy Landscape')
    
    plt.tight_layout()
    plt.savefig(f"training_progress_epoch_{epoch}.png")
    plt.close()
```

## Debugging Tips

Training EBMs can be challenging. Here are some tips for debugging:

1. **Start with simpler energy functions**: Begin with analytical energy functions before moving to neural networks
2. **Monitor energy values**: If energy values become extremely large or small, the model may be collapsing
3. **Visualize samples**: Regularly visualize samples during training to check if they match the data distribution
4. **Adjust learning rate**: Try different learning rates; EBMs often require smaller learning rates
5. **Increase MCMC steps**: More MCMC steps for negative samples can improve training stability
6. **Add noise regularization**: Adding small noise to data samples can help prevent overfitting
7. **Use gradient clipping**: Clip gradients to prevent instability during training
8. **Try different initializations**: Initial parameter values can significantly impact training dynamics

## Training on Different Data Types

The training approach may vary depending on the type of data:

### Images

For image data, convolutional architectures are recommended:

```python
class ImageEBM(BaseEnergyFunction):
    def __init__(self, input_channels=1, image_size=28):
        super().__init__()
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),  # Downsampling
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # Downsampling
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # Downsampling
            nn.LeakyReLU(0.2)
        )
        
        # Calculate final feature map size
        feature_size = 256 * (image_size // 8) * (image_size // 8)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 1)
        )
    
    def forward(self, x):
        # Ensure proper dimensions for convolutional layers
        if x.ndim == 3:
            x = x.unsqueeze(1)  # Add channel dimension for grayscale
        
        features = self.conv_net(x)
        energy = self.fc(features).squeeze(-1)
        return energy
```

### Sequential Data

For sequential data, recurrent or transformer architectures may be more appropriate:

```python
class SequentialEBM(BaseEnergyFunction):
    def __init__(self, input_dim=1, hidden_dim=128, seq_len=50):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        
        # Use final hidden state
        final_hidden = lstm_out[:, -1, :]
        
        # Compute energy
        energy = self.fc(final_hidden).squeeze(-1)
        return energy
```

## Conclusion

Training energy-based models is a challenging but rewarding process. By leveraging the techniques outlined in this guide, you can effectively train EBMs using TorchEBM for a variety of applications. Remember to monitor training progress, visualize results, and adjust your approach based on the specific characteristics of your data and modeling objectives.

Whether you're using contrastive divergence, score matching, or other methods, the key is to ensure that your energy function accurately captures the underlying structure of your data distribution. With practice and experimentation, you can master the art of training energy-based models for complex tasks in unsupervised learning. 