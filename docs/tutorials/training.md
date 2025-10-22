---
sidebar_position: 7
title: Training EBMs
description: Methods and best practices for training energy-based models
icon: material/school
---

# Training Energy-Based Models

This guide covers the fundamental techniques for training energy-based models (EBMs) using TorchEBM. We'll explore various training methods, loss functions, and optimization strategies to help you effectively train your models.

## Overview

Training energy-based models involves estimating the parameters of a model such that the corresponding probability distribution matches a target data distribution. Unlike in traditional supervised learning, this is often an unsupervised task where the goal is to learn the underlying structure of the data.

The training process typically involves:

1. Defining a model (parameterized by a neural network or analytical form)
2. Choosing a training method and loss function
3. Optimizing the model parameters
4. Evaluating the model using sampling and visualization techniques

## Defining a Model

In TorchEBM, you can create custom models by subclassing `BaseModel`:

```python
import torch
import torch.nn as nn
from torchebm.core import BaseModel

class MLPModel(BaseModel):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
```

## Training with Contrastive Divergence

Contrastive Divergence (CD) is one of the most common methods for training EBMs. Here's a complete example of training with CD using TorchEBM:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

from torchebm.core import BaseModel, CosineScheduler
from torchebm.samplers import LangevinDynamics
from torchebm.losses import ContrastiveDivergence
from torchebm.datasets import TwoMoonsDataset

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

os.makedirs("training_plots", exist_ok=True)

INPUT_DIM = 2
HIDDEN_DIM = 16
BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 1e-3

SAMPLER_STEP_SIZE = CosineScheduler(start_value=3e-2, end_value=5e-3, n_steps=100)
SAMPLER_NOISE_SCALE = CosineScheduler(start_value=3e-1, end_value=1e-2, n_steps=100)

CD_K = 10
USE_PCD = True
VISUALIZE_EVERY = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = TwoMoonsDataset(n_samples=3000, noise=0.05, seed=42, device=device)
real_data_for_plotting = dataset.get_data()
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
)

model = MLPModel(INPUT_DIM, HIDDEN_DIM).to(device)
sampler = LangevinDynamics(
    model=model,
    step_size=SAMPLER_STEP_SIZE,
    noise_scale=SAMPLER_NOISE_SCALE,
    device=device,
)
loss_fn = ContrastiveDivergence(
    model=model,
    sampler=sampler,
    k_steps=CD_K,
    persistent=USE_PCD,
    buffer_size=BATCH_SIZE,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

losses = []
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for i, data_batch in enumerate(dataloader):
        optimizer.zero_grad()

        loss, negative_samples = loss_fn(data_batch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(dataloader)
    losses.append(avg_epoch_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_epoch_loss:.4f}")

    if (epoch + 1) % VISUALIZE_EVERY == 0 or epoch == 0:
        print("Generating visualization...")
        plot_energy_and_samples(
            model=model,
            real_samples=real_data_for_plotting,
            sampler=sampler,
            epoch=epoch + 1,
            device=device,
            plot_range=2.5,
            k_sampling=200,
        )

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)
plt.savefig('docs/assets/images/training/cd_training_loss.png')
plt.show()
```

## Visualization During Training

It's important to visualize the model's progress during training. Here's a helper function to plot the energy landscape and samples:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchebm.core import BaseModel
from torchebm.samplers import LangevinDynamics

@torch.no_grad()
def plot_energy_and_samples(
    model: BaseModel,
    real_samples: torch.Tensor,
    sampler: LangevinDynamics,
    epoch: int,
    device: torch.device,
    grid_size: int = 100,
    plot_range: float = 3.0,
    k_sampling: int = 100,
):
    plt.figure(figsize=(8, 8))

    x_coords = torch.linspace(-plot_range, plot_range, grid_size, device=device)
    y_coords = torch.linspace(-plot_range, plot_range, grid_size, device=device)
    xv, yv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    grid = torch.stack([xv.flatten(), yv.flatten()], dim=1)

    model.eval()
    energy_values = model(grid).cpu().numpy().reshape(grid_size, grid_size)

    log_prob_values = -energy_values
    log_prob_values = log_prob_values - np.max(log_prob_values)
    prob_density = np.exp(log_prob_values)

    plt.contourf(
        xv.cpu().numpy(),
        yv.cpu().numpy(),
        prob_density,
        levels=50,
        cmap="viridis",
    )
    plt.colorbar(label="exp(-Energy) (unnormalized density)")

    vis_start_noise = torch.randn(
        500, real_samples.shape[1], device=device
    )
    model_samples_tensor = sampler.sample(x=vis_start_noise, n_steps=k_sampling)
    model_samples = model_samples_tensor.cpu().numpy()

    real_samples_np = real_samples.cpu().numpy()
    plt.scatter(
        real_samples_np[:, 0],
        real_samples_np[:, 1],
        s=10,
        alpha=0.5,
        label="Real Data",
        c="white",
        edgecolors="k",
        linewidths=0.5,
    )
    plt.scatter(
        model_samples[:, 0],
        model_samples[:, 1],
        s=10,
        alpha=0.5,
        label="Model Samples",
        c="red",
        edgecolors="darkred",
        linewidths=0.5,
    )

    plt.xlim(-plot_range, plot_range)
    plt.ylim(-plot_range, plot_range)
    plt.title(f"Epoch {epoch}")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"docs/assets/images/training/ebm_training_epoch_{epoch}.png")
    plt.close()
```

## Training with Score Matching

An alternative to Contrastive Divergence is Score Matching, which doesn't require MCMC sampling:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchebm.core import BaseModel
from torchebm.losses import ScoreMatching
from torchebm.datasets import GaussianMixtureDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLPModel(input_dim=2).to(device)
sm_loss_fn = ScoreMatching(
    model=model,
    hessian_method="hutchinson",
    hutchinson_samples=5,
    device=device,
)
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = GaussianMixtureDataset(
    n_samples=500, n_components=4, std=0.1, seed=123
).get_data()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

losses = []
for epoch in range(50):
    epoch_loss = 0.0
    for batch_data in dataloader:
        batch_data = batch_data.to(device)

        optimizer.zero_grad()
        loss = sm_loss_fn(batch_data)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/50, Loss: {avg_loss:.6f}")

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Score Matching Training Loss')
plt.grid(True, alpha=0.3)
plt.savefig('docs/assets/images/training/sm_training_loss.png')
plt.show()
```

## Comparing Training Methods

Here's how the major training methods for EBMs compare:

| Method                            | Pros                                                                                                                             | Cons                                                                                                                                         | Best For                                                                        |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| **Contrastive Divergence (CD)**   | - Simple to implement<br>- Computationally efficient<br>- Works well for simple distributions                                    | - May not converge to true gradient<br>- Limited mode exploration with short MCMC runs<br>- Can lead to poor samples                         | Restricted Boltzmann Machines, simpler energy-based models                      |
| **Persistent CD (PCD)**           | - Better mode exploration than CD<br>- More accurate gradient estimation<br>- Improved sample quality                            | - Requires maintaining persistent chains<br>- Can be unstable with high learning rates<br>- Chains can get stuck in metastable states        | Deep Boltzmann Machines, models with complex energy landscapes                  |
| **Score Matching**                | - Avoids MCMC sampling<br>- Consistent estimator<br>- Stable optimization                                                        | - Requires computing Hessian diagonals<br>- High computational cost in high dimensions<br>- Need for second derivatives                      | Continuous data, models with tractable derivatives                              |
| **Denoising Score Matching**      | - Avoids explicit Hessian computation<br>- More efficient than standard score matching<br>- Works well for high-dimensional data | - Performance depends on noise distribution<br>- Trade-off between noise level and estimation accuracy<br>- May smooth out important details | Image modeling, high-dimensional continuous distributions                       |
| **Sliced Score Matching**         | - Linear computational complexity<br>- No Hessian computation needed<br>- Scales well to high dimensions                         | - Approximation depends on number of projections<br>- Less accurate with too few random projections<br>- Still requires gradient computation | High-dimensional problems where other score matching variants are too expensive |


## Advanced Training Techniques

### Gradient Clipping

Gradient clipping is essential for stable EBM training:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Regularization Techniques

Adding regularization can help stabilize training:

```python
weight_decay = 1e-4
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)

from torch.nn.utils import spectral_norm

class RegularizedMLPModel(BaseModel):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, 1))
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)
```

## Tips for Successful Training

1. **Start Simple**: Begin with a simple model and dataset, then increase complexity
2. **Monitor Energy Values**: Watch for energy collapse (very negative values) which indicates instability
3. **Adjust Sampling Parameters**: Tune MCMC step size and noise scale for effective exploration
4. **Use Persistent CD**: For complex distributions, persistent CD often yields better results
5. **Visualize Frequently**: Regularly check the energy landscape and samples to track progress
6. **Gradient Clipping**: Always use gradient clipping to prevent explosive gradients
7. **Parameter Scheduling**: Use schedulers for learning rate, step size, and noise scale
8. **Batch Normalization**: Consider adding batch normalization in your energy network
9. **Ensemble Methods**: Train multiple models and ensemble their predictions for better results
10. **Patience**: EBM training can be challenging - be prepared to experiment with hyperparameters 