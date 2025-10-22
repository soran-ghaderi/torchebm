---
sidebar_position: 6
title: Custom Neural Networks
description: Creating custom neural network-based models
icon: material/brain
---

# Custom Neural Network Models

Energy-based models (EBMs) are highly flexible, and one of their key advantages is that the model can be parameterized using neural networks. This guide explains how to create and use neural network-based models in TorchEBM.

## Overview

Neural networks provide a powerful way to represent complex energy landscapes that can't be easily defined analytically. By using neural networks as models:

- You can capture complex, high-dimensional distributions
- The model can be learned from data
- You gain the expressivity of modern deep learning architectures

## Basic Neural Network Model

To create a neural network-based model in TorchEBM, you need to subclass the `BaseModel` base class and implement the `forward` method:

```python
import torch
import torch.nn as nn
from torchebm.core import BaseModel


class NeuralNetModel(BaseModel):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)
```

## Design Considerations

When designing neural network models, consider the following:

### Network Architecture

The choice of architecture depends on the data type and complexity:

- **MLPs**: Good for generic, low-dimensional data
- **CNNs**: Effective for images and data with spatial structure
- **Transformers**: Useful for sequential data or when attention mechanisms are beneficial
- **Graph Neural Networks**: For data with graph structure

### Output Requirements

Remember the following key points:

1. The model should output a scalar value for each sample in the batch
2. Lower energy values should correspond to higher probability density
3. The neural network must be differentiable for gradient-based sampling methods to work

### Scale and Normalization

Energy values should be properly scaled to avoid numerical issues:

- Very large energy values can cause instability in sampling
- Models that grow too quickly may cause sampling algorithms to fail

## Example: MLP Model for 2D Data

Here's a complete example with a simple MLP model:

```python
import torch
import torch.nn as nn
from torchebm.core import (
    BaseModel,
    CosineScheduler,
)
from torchebm.samplers import LangevinDynamics
from torchebm.losses import ContrastiveDivergence
from torchebm.datasets import GaussianMixtureDataset
from torch.utils.data import DataLoader

SEED = 42
torch.manual_seed(SEED)

class MLPModel(BaseModel):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = GaussianMixtureDataset(
    n_samples=1000,
    n_components=5,
    std=0.1,
    radius=1.5,
    device=device,
    seed=SEED,
)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

model = MLPModel(input_dim=2, hidden_dim=64).to(device)
SAMPLER_NOISE_SCALE = CosineScheduler(
    initial_value=2e-1, final_value=1e-2, total_steps=50
)

sampler = LangevinDynamics(
    model=model,
    step_size=0.01,
    device=device,
    noise_scale=SAMPLER_NOISE_SCALE,
)

loss_fn = ContrastiveDivergence(
    model=model,
    sampler=sampler,
    k_steps=10,
    persistent=False,
    device=device,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 200
for epoch in range(n_epochs):
    epoch_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()

        loss, neg_samples = loss_fn(batch)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

def generate_samples(model, n_samples=500):
    sampler = LangevinDynamics(model=model, step_size=0.005, device=device)

    initial_samples = torch.randn(n_samples, 2).to(device)

    with torch.no_grad():
        samples = sampler.sample(
            initial_state=initial_samples,
            dim=initial_samples.shape[-1],
            n_samples=n_samples,
            n_steps=1000,
        )

    return samples.cpu()

samples = generate_samples(model)
print(f"Generated {len(samples)} samples from the energy-based model")

```

## Example: Convolutional Model for Images

For image data, convolutional architectures are more appropriate:

```python
import torch
import torch.nn as nn
from torchebm.core import BaseModel


class ConvolutionalModel(BaseModel):
    def __init__(self, channels=1, width=28, height=28):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.SELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.SELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.SELU(),
        )

        feature_size = 128 * (width // 8) * (height // 8)

        self.energy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 128),
            nn.SELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)

        features = self.conv_net(x)
        energy = self.energy_head(features).squeeze(-1)

        return energy
```

## Advanced Pattern: Composed Models

You can combine multiple analytical models with multiple neural networks for best of both worlds:

```python
import torch
import torch.nn as nn
from torchebm.core import BaseModel, GaussianModel


class CompositionalModel(BaseModel):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()

        self.analytical_component = GaussianModel(
            mean=torch.zeros(input_dim),
            cov=torch.eye(input_dim)
        )

        self.neural_component = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 1)
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        analytical_energy = self.analytical_component(x)

        neural_energy = self.neural_component(x).squeeze(-1)

        alpha = torch.sigmoid(self.alpha)
        combined_energy = alpha * analytical_energy + (1 - alpha) * neural_energy

        return combined_energy
```

## Training Strategies

Training neural network models requires special techniques:

### Contrastive Divergence

A common approach is contrastive divergence, which minimizes the energy of data samples while maximizing the energy of samples from the model:

```python
loss_fn = ContrastiveDivergence(
    model=model,
    sampler=sampler,
    k_steps=10,  # Number of MCMC steps
    persistent=False,  # Set to True for Persistent Contrastive Divergence
    device=device,
)

def train_step_contrastive_divergence(data_batch):
    optimizer.zero_grad()

    loss, neg_samples = loss_fn(data_batch)

    loss.backward()

    optimizer.step()

    return loss.item()
```

### Score Matching

Score matching is another approach that avoids the need for MCMC sampling:

```python

sm_loss_fn = ScoreMatching(
    model=model,
    hessian_method="hutchinson",
    hutchinson_samples=5,
    device=device,
)

batch_loss = train_step_contrastive_divergence(data_batch)
```

## Tips for Neural Network Models

1. **Start Simple**: Begin with a simple architecture and gradually increase complexity
2. **Regularization**: Use weight decay or spectral normalization to prevent extreme energy values
3. **Gradient Clipping**: Apply gradient clipping during training to prevent instability
4. **Initialization**: Careful initialization of weights can help convergence
5. **Monitoring**: Track energy values during training to ensure they stay in a reasonable range
6. **Batch Normalization**: Use with caution as it can affect the shape of the energy landscape
7. **Residual Connections**: Can help with gradient flow in deeper networks

## Conclusion

Neural network models provide a powerful way to model complex distributions in energy-based models. By leveraging the flexibility of deep learning architectures, you can create expressive models that capture intricate patterns in your data.

Remember to carefully design your architecture, choose appropriate training methods, and monitor the behavior of your model during training and sampling. 