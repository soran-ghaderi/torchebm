---
sidebar_position: 2
title: Custom Neural Networks
description: Creating custom neural network-based models in TorchEBM.
---

# Custom Neural Network Models

A key advantage of Energy-Based Models (EBMs) is their flexibility; the model can be parameterized using any standard neural network. This guide explains how to create and use your own neural network-based models in TorchEBM.

## Creating a Custom Model

Any `torch.nn.Module` that outputs a single scalar energy value for each input sample can be used as an EBM. To ensure compatibility with the TorchEBM ecosystem, your custom model should inherit from `torchebm.core.BaseModel`.

The only requirement is to implement the `forward` method, which should take a batch of data `x` with shape `(batch_size, *dims)` and return a tensor of energy values with shape `(batch_size,)`.

### Example: MLP Model for 2D Data

Here's how to define a simple Multi-Layer Perceptron (MLP) as an EBM for 2D data.

```python
import torch
import torch.nn as nn
from torchebm.core import BaseModel

class MLPModel(BaseModel):
    def __init__(self, input_dim=2, hidden_dim=128):
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

# You can now use this model with TorchEBM samplers and losses.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLPModel().to(device)
```

### Example: Convolutional Model for Images

For image data, a convolutional architecture is more appropriate. The principle is the same: the network must output a single energy value per image.

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
            nn.SELU()
        )

        feature_size = 64 * (width // 4) * (height // 4)

        self.energy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 128),
            nn.SELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.conv_net(x)
        energy = self.energy_head(features).squeeze(-1)
        return energy

# Example with a batch of 28x28 grayscale images
image_model = ConvolutionalModel(channels=1, width=28, height=28)
dummy_images = torch.randn(64, 1, 28, 28) # (batch, channels, height, width)
energies = image_model(dummy_images)
print(energies.shape) # torch.Size([64])
```

## Design Considerations

-   **Architecture Choice**: The network architecture should match your data modality. Use MLPs for tabular data, CNNs for images, Transformers for sequences, etc.
-   **Output Shape**: The `forward` method *must* return a 1D tensor of shape `(batch_size,)`. Using `.squeeze(-1)` is a common way to achieve this.
-   **Differentiability**: The model must be differentiable for gradient-based samplers and training methods to work.
-   **Numerical Stability**: Energy values can grow very large or small during training. Consider using techniques like weight decay, spectral normalization, or activation function clipping (e.g., `nn.Tanh`) on the output to improve stability.

By following these simple guidelines, you can integrate any custom neural network into the TorchEBM framework. 