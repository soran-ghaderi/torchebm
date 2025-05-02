---
sidebar_position: 4
title: Loss Functions
description: Understanding and using loss functions for training energy-based models
---

# Loss Functions

Training energy-based models involves estimating and minimizing the difference between the model distribution and the data distribution. TorchEBM provides various loss functions to accomplish this.

## Contrastive Divergence

Contrastive Divergence (CD) is one of the most popular methods for training energy-based models. It uses MCMC sampling to generate negative examples from the current model.

### Basic Usage

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchebm.core import BaseEnergyFunction
from torchebm.losses import ContrastiveDivergence
from torchebm.samplers import LangevinDynamics

# Define a custom energy function
class MLPEnergy(BaseEnergyFunction):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

# Create energy model, sampler, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
energy_fn = MLPEnergy(input_dim=2, hidden_dim=64).to(device)

# Set up sampler for negative samples
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.1,
    device=device
)

# Create Contrastive Divergence loss
loss_fn = ContrastiveDivergence(
    energy_function=energy_fn,
    sampler=sampler,
    k_steps=10,  # Number of MCMC steps
    persistent=False,  # Standard CD (non-persistent)
)

# Define optimizer
optimizer = optim.Adam(energy_fn.parameters(), lr=0.001)

# During training:
data_batch = torch.randn(128, 2).to(device)  # Your real data batch
optimizer.zero_grad()
loss, negative_samples = loss_fn(data_batch)
loss.backward()
optimizer.step()
```

### Advanced Options

The `ContrastiveDivergence` loss function in TorchEBM supports several advanced options:

#### Persistent Contrastive Divergence (PCD)

PCD maintains a buffer of negative samples across training iterations, which can lead to better mixing:

```python
# Create Persistent Contrastive Divergence loss
loss_fn = ContrastiveDivergence(
    energy_function=energy_fn,
    sampler=sampler,
    k_steps=10,
    persistent=True,  # Enable PCD
    buffer_size=1024,  # Size of the persistent buffer
    buffer_init='rand'  # How to initialize the buffer ('rand' or 'data')
)
```

#### Using Schedulers for Sampling Parameters

You can use schedulers to dynamically adjust the sampler's step size or noise scale during training:

```python
from torchebm.core import CosineScheduler, ExponentialDecayScheduler, LinearScheduler

# Define schedulers for step size and noise scale
step_size_scheduler = CosineScheduler(
    start_value=3e-2,
    end_value=5e-3,
    n_steps=100
)

noise_scheduler = CosineScheduler(
    start_value=3e-1,
    end_value=1e-2,
    n_steps=100
)

# Create sampler with schedulers
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=step_size_scheduler,
    noise_scale=noise_scheduler,
    device=device
)

# Create CD loss with this sampler
loss_fn = ContrastiveDivergence(
    energy_function=energy_fn,
    sampler=sampler,
    k_steps=10,
    persistent=True
)
```

## Score Matching

Score Matching is another approach for training EBMs that avoids the need for MCMC sampling. It directly optimizes the score function (gradient of log-density).

### Basic Usage

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchebm.core import BaseEnergyFunction
from torchebm.losses import ScoreMatching
from torchebm.datasets import GaussianMixtureDataset

# Define a custom energy function
class MLPEnergy(BaseEnergyFunction):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # a scalar value

# Setup model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
energy_fn = MLPEnergy(input_dim=2).to(device)

# Create score matching loss
sm_loss_fn = ScoreMatching(
    energy_function=energy_fn,
    hessian_method="hutchinson",  # More efficient for higher dimensions
    hutchinson_samples=5,
    device=device
)

# Setup optimizer
optimizer = optim.Adam(energy_fn.parameters(), lr=0.001)

# Setup data
dataset = GaussianMixtureDataset(
    n_samples=500, n_components=4, std=0.1, seed=123
).get_data()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(10):
    epoch_loss = 0.0
    for batch_data in dataloader:
        batch_data = batch_data.to(device)

        optimizer.zero_grad()
        loss = sm_loss_fn(batch_data)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.6f}")
```

### Variants of Score Matching

TorchEBM supports different variants of score matching:

#### Explicit Score Matching

This is the standard form of score matching, which requires computing the Hessian of the energy function:

```python
sm_loss_fn = ScoreMatching(
    energy_function=energy_fn,
    hessian_method="exact",  # Explicitly compute Hessian (slow for high dimensions)
    device=device
)
```

#### Hutchinson's Trick

To make score matching more efficient, we can use Hutchinson's trick to estimate the trace of the Hessian:

```python
sm_loss_fn = ScoreMatching(
    energy_function=energy_fn,
    hessian_method="hutchinson",  # Use Hutchinson's trick
    hutchinson_samples=5,  # Number of noise samples to use
    device=device
)
```

#### Denoising Score Matching

Denoising score matching adds noise to data points and tries to learn the score of the noised distribution:

```python
from torchebm.losses import DenoisingScoreMatching

dsm_loss_fn = DenoisingScoreMatching(
    energy_function=energy_fn,
    sigma=0.1,  # Noise level
    device=device
)

# During training:
optimizer.zero_grad()
loss = dsm_loss_fn(data_batch)
loss.backward()
optimizer.step()
```

## Noise Contrastive Estimation (NCE)

NCE is another alternative for training EBMs that uses a noise distribution to avoid computing the partition function:

```python
from torchebm.losses import NoiseContrastiveEstimation
import torch.distributions as D

# Define a noise distribution
noise_dist = D.Normal(0, 1)

# Create NCE loss
nce_loss_fn = NoiseContrastiveEstimation(
    energy_function=energy_fn,
    noise_distribution=noise_dist,
    noise_samples_per_data=10,
    device=device
)

# During training:
optimizer.zero_grad()
loss = nce_loss_fn(data_batch)
loss.backward()
optimizer.step()
```

## Complete Training Example with Loss Function

Here's a complete example showing how to train an EBM using Contrastive Divergence loss:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from torchebm.core import BaseEnergyFunction
from torchebm.samplers import LangevinDynamics
from torchebm.losses import ContrastiveDivergence
from torchebm.datasets import TwoMoonsDataset

# Define energy function
class MLPEnergy(BaseEnergyFunction):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
INPUT_DIM = 2
HIDDEN_DIM = 16
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 1e-3
CD_K = 10  # MCMC steps for Contrastive Divergence
USE_PCD = True  # Use Persistent Contrastive Divergence

# Setup data
dataset = TwoMoonsDataset(n_samples=3000, noise=0.05, seed=42, device=device)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Create model, sampler, and loss function
energy_model = MLPEnergy(INPUT_DIM, HIDDEN_DIM).to(device)
sampler = LangevinDynamics(
    energy_function=energy_model,
    step_size=0.1,
    device=device,
)
loss_fn = ContrastiveDivergence(
    energy_function=energy_model,
    sampler=sampler,
    k_steps=CD_K,
    persistent=USE_PCD,
    buffer_size=BATCH_SIZE,
).to(device)

# Optimizer
optimizer = optim.Adam(energy_model.parameters(), lr=LEARNING_RATE)

# Training loop
losses = []
print("Starting training...")
for epoch in range(EPOCHS):
    energy_model.train()
    epoch_loss = 0.0
    
    for i, data_batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Calculate Contrastive Divergence loss
        loss, negative_samples = loss_fn(data_batch)
        
        # Backpropagate and optimize
        loss.backward()
        
        # Optional: Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(energy_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_epoch_loss = epoch_loss / len(dataloader)
    losses.append(avg_epoch_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_epoch_loss:.4f}")

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('docs/assets/images/loss_functions/cd_training_loss.png')
plt.show()
```

![CD Training Loss](../assets/images/loss_functions/cd_training_loss.png)

## Choosing the Right Loss Function

Different loss functions are suitable for different scenarios:

- **Contrastive Divergence**: Good general-purpose method, especially with complex energy landscapes
- **Persistent CD**: Better mixing properties than standard CD, but requires more memory
- **Score Matching**: Avoids sampling but can be numerically unstable in high dimensions
- **Denoising Score Matching**: More stable than standard score matching, good for high dimensions
- **NCE**: Works well with complex distributions where sampling is difficult

## Tips for Stable Training

1. **Regularization**: Add L2 regularization to prevent the energy from collapsing
2. **Gradient Clipping**: Use `torch.nn.utils.clip_grad_norm_` to prevent unstable updates
3. **Learning Rate**: Use a small learning rate, especially at the beginning
4. **Sampling Steps**: Increase the number of sampling steps k for better negative samples
5. **Batch Size**: Use larger batch sizes for more stable gradient estimates 
6. **Parameter Schedulers**: Use schedulers for sampler parameters to improve mixing
7. **Monitor Energy Values**: Ensure the energy values don't collapse to very large negative values 