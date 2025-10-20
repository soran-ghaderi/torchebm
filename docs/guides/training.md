---
sidebar_position: 5
title: Training EBMs
description: A complete guide to training energy-based models with TorchEBM.
---

# Training Energy-Based Models

This guide provides a complete, end-to-end example of how to train an Energy-Based Model using TorchEBM. We will bring together the concepts of models, samplers, and loss functions into a full training pipeline.

## The Complete Training Pipeline

We will train a neural network to model a synthetic "two moons" dataset. The full process involves setting up the dataset, defining the model, configuring the sampler and loss function, running the training loop, and visualizing the results.

### Step 1: Imports and Setup

First, let's import the necessary libraries and set up our device.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

from torchebm.core import BaseModel
from torchebm.samplers import LangevinDynamics
from torchebm.losses import ContrastiveDivergence
from torchebm.datasets import TwoMoonsDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("training_plots", exist_ok=True)
```

### Step 2: Define the Model

We'll use a simple MLP, inheriting from `BaseModel`.

```python
class MLPModel(BaseModel):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
```

### Step 3: Configure Training Components

Next, we set up the dataset, dataloader, model, sampler, loss function, and optimizer.

```python
# Dataset and DataLoader
dataset = TwoMoonsDataset(n_samples=3000, noise=0.05, device=device)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

# Model
model = MLPModel().to(device)

# Sampler for Contrastive Divergence
sampler = LangevinDynamics(
    model=model,
    step_size=0.1,
    noise_scale=0.01
)

# Loss Function (Persistent Contrastive Divergence)
loss_fn = ContrastiveDivergence(
    model=model,
    sampler=sampler,
    k_steps=10,
    persistent=True,
    buffer_size=256
).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

### Step 4: The Training Loop

The training loop iterates over the data, computes the loss, and updates the model parameters. We will also periodically visualize the model's progress.

```python
epochs = 200
for epoch in range(epochs):
    model.train()
    for data_batch in dataloader:
        optimizer.zero_grad()
        loss, _ = loss_fn(data_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    if (epoch + 1) % 20 == 0:
        plot_energy_and_samples(model, dataset.get_data(), sampler, epoch + 1)

print("Training finished!")
```

### Step 5: Visualization Helper

Visualizing the energy surface and samples during training is crucial for debugging and understanding the model's behavior.

```python
@torch.no_grad()
def plot_energy_and_samples(model, real_samples, sampler, epoch):
    plt.figure(figsize=(8, 8))
    plot_range = 2.5
    grid_size = 100

    # Create grid for energy surface plot
    x_coords = torch.linspace(-plot_range, plot_range, grid_size, device=device)
    y_coords = torch.linspace(-plot_range, plot_range, grid_size, device=device)
    xv, yv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    grid = torch.stack([xv.flatten(), yv.flatten()], dim=1)

    # Calculate and plot energy surface
    model.eval()
    energy_values = model(grid).cpu().numpy().reshape(grid_size, grid_size)
    prob_density = np.exp(-energy_values - np.max(-energy_values))
    plt.contourf(xv.cpu().numpy(), yv.cpu().numpy(), prob_density, levels=50, cmap="viridis")
    
    # Generate and plot model samples
    vis_start_noise = torch.randn(500, real_samples.shape[1], device=device)
    model_samples = sampler.sample(x=vis_start_noise, n_steps=200).cpu().numpy()

    # Plot real and model samples
    plt.scatter(real_samples.cpu()[:, 0], real_samples.cpu()[:, 1], s=10, alpha=0.5, label="Real Data", c="white", edgecolors="k")
    plt.scatter(model_samples[:, 0], model_samples[:, 1], s=10, alpha=0.5, label="Model Samples", c="red", edgecolors="darkred")

    plt.xlim(-plot_range, plot_range)
    plt.ylim(-plot_range, plot_range)
    plt.title(f"Epoch {epoch}")
    plt.legend()
    plt.savefig(f"training_plots/ebm_training_epoch_{epoch}.png")
    plt.close()
```

## Tips for Stable Training

Training EBMs can sometimes be unstable. Here are some tips to improve convergence:

-   **Gradient Clipping**: Use `torch.nn.utils.clip_grad_norm_` to prevent exploding gradients, which is a common issue.
-   **Weight Decay**: Add weight decay to your optimizer (e.g., `optim.Adam(..., weight_decay=1e-4)`) to regularize the model and prevent energy values from growing too large.
-   **Learning Rate**: EBMs often require smaller learning rates than supervised models. Consider using a learning rate scheduler.
-   **Sampler Parameters**: The sampler's `step_size` and `n_steps` in the loss function are critical. Too large a step size can cause instability, while too few steps can lead to poor negative samples.
-   **Persistent CD**: For complex distributions, `persistent=True` in `ContrastiveDivergence` often yields better results by improving the quality of negative samples.
-   **Monitor Energy Values**: Keep an eye on the energy values for positive and negative samples. If they grow uncontrollably or collapse, your training is likely unstable. 