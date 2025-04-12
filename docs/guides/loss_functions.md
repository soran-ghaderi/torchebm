---
sidebar_position: 4
title: BaseLoss Functions
description: Understanding and using loss functions for training energy-based models
---

# BaseLoss Functions

Training energy-based models involves estimating and minimizing the difference between the model distribution and the data distribution. TorchEBM provides various loss functions to accomplish this.

## Contrastive Divergence Methods

Contrastive Divergence (CD) is a family of methods used to train energy-based models by comparing data samples with model samples.

### Contrastive Divergence (CD-k)

CD-k uses k steps of MCMC to generate model samples:

```python
from torchebm.losses import ContrastiveDivergenceLoss
from torchebm.samplers.langevin_dynamics import LangevinDynamics
from torchebm.core import GaussianEnergy
import torch

# Set up an energy model (could be a neural network)
model = GaussianEnergy(
    mean=torch.zeros(2),
    cov=torch.eye(2)
)

# Define a sampler for negative samples
sampler = LangevinDynamics(
    energy_function=model,
    step_size=0.01
)

# Create CD loss
cd_loss = ContrastiveDivergenceLoss(sampler, k=10)

# During training:
data_samples = torch.randn(100, 2)  # Your real data
loss = cd_loss(model, data_samples)
```

### Persistent Contrastive Divergence (PCD)

PCD maintains a persistent chain of samples across training iterations:

```python
from torchebm.losses import PersistentContrastiveDivergenceLoss

# Create PCD loss
pcd_loss = PersistentContrastiveDivergenceLoss(
    sampler,
    n_persistent_chains=1000,
    k=10
)

# During training:
for epoch in range(n_epochs):
    data_samples = get_batch()  # Your data batch
    loss = pcd_loss(model, data_samples)
    # Optimizer step...
```

## Score Matching Techniques

Score matching aims to match the gradient of the log-density rather than the density itself, avoiding the need to compute the partition function.

### Standard Score Matching

```python
from torchebm.losses import ScoreMatchingLoss

# Create score matching loss
sm_loss = ScoreMatchingLoss()

# During training:
data_samples = torch.randn(100, 2)  # Your real data
loss = sm_loss(model, data_samples)
```

### Denoising Score Matching

Denoising score matching adds noise to data samples and tries to predict the score of the noisy distribution:

```python
from torchebm.losses import DenoisingScoreMatchingLoss

# Create denoising score matching loss with noise scale sigma
dsm_loss = DenoisingScoreMatchingLoss(sigma=0.1)

# During training:
data_samples = torch.randn(100, 2)  # Your real data
loss = dsm_loss(model, data_samples)
```

## Other BaseLoss Functions

### Maximum Likelihood Estimation (MLE)

For models where the partition function can be computed:

```python
from torchebm.losses import MaximumLikelihoodLoss

# Only suitable for certain energy functions where Z is known
mle_loss = MaximumLikelihoodLoss()

# During training:
data_samples = torch.randn(100, 2)  # Your real data
loss = mle_loss(model, data_samples)
```

### Noise Contrastive Estimation (NCE)

NCE uses a noise distribution to avoid computing the partition function:

```python
from torchebm.losses import NoiseContrastiveEstimationLoss
import torch.distributions as D

# Define a noise distribution
noise_dist = D.Normal(0, 1)

# Create NCE loss
nce_loss = NoiseContrastiveEstimationLoss(
    noise_distribution=noise_dist,
    noise_samples_per_data=10
)

# During training:
data_samples = torch.randn(100, 2)  # Your real data
loss = nce_loss(model, data_samples)
```

## Training with BaseLoss Functions

Here's a general training loop for energy-based models:

```python
import torch
import torch.optim as optim
from torchebm.core import BaseEnergyFunction
import torch.nn as nn


# Define a neural network energy function
class NeuralNetEBM(BaseEnergyFunction):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# Create model, loss, and optimizer
model = NeuralNetEBM(input_dim=10, hidden_dim=64)
sampler = LangevinDynamics(energy_function=model, step_size=0.01)
loss_fn = ContrastiveDivergenceLoss(sampler, k=10)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    # Get a batch of real data
    real_data = get_data_batch()

    # Compute loss
    loss = loss_fn(model, real_data)

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, BaseLoss: {loss.item()}")
```

## Choosing BaseLoss Functions

Different loss functions are suitable for different scenarios:

- **Contrastive Divergence**: Good general-purpose method, especially with complex energy landscapes
- **Persistent CD**: Better mixing properties than standard CD, but requires more memory
- **Score Matching**: Avoids sampling but can be numerically unstable in high dimensions
- **Denoising Score Matching**: More stable than standard score matching, good for high dimensions
- **NCE**: Works well with complex distributions where sampling is difficult

## BaseLoss Function Implementation Details

Each loss function in TorchEBM follows a standard pattern:

1. Compute energy of data samples
2. Generate or obtain model samples
3. Compute energy of model samples
4. Calculate the loss based on these energies
5. Return the loss value for backpropagation

## Tips for Stable Training

1. **Regularization**: Add L2 regularization to prevent the energy from collapsing
2. **Gradient Clipping**: Use gradient clipping to prevent unstable updates
3. **Learning Rate**: Use a small learning rate, especially at the beginning
4. **Sampling Steps**: Increase the number of sampling steps k for better negative samples
5. **Batch Size**: Use larger batch sizes for more stable gradient estimates 