---
title: Models Implementation
description: Detailed implementation details of TorchEBM's models
icon: material/vector-square
---

# Models Implementation

!!! abstract "Implementation Details"
    This guide explains the implementation of neural network models in TorchEBM, including architecture designs, training workflows, and integration with energy functions.

## Base Model Architecture

The `BaseModel` class provides the foundation for all neural networks in TorchEBM:

```python
import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Any, Optional, Union

class BaseModel(nn.Module):
    """Base class for all neural network models."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], activation: Optional[nn.Module] = None):
        """Initialize base model.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden dimensions
            activation: Activation function
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation or nn.ReLU()
        
        # Build network architecture
        self._build_network()
    
    def _build_network(self):
        """Build the neural network architecture."""
        layers = []
        
        # Input layer
        prev_dim = self.input_dim
        
        # Hidden layers
        for dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(self.activation)
            prev_dim = dim
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor
        """
        return self.network(x)
```

## MLP Energy Model

```python
class MLPEnergyModel(BaseModel):
    """Multi-layer perceptron energy model."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        activation: Optional[nn.Module] = None,
        use_spectral_norm: bool = False
    ):
        """Initialize MLP energy model.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden dimensions
            activation: Activation function
            use_spectral_norm: Whether to use spectral normalization
        """
        super().__init__(input_dim, hidden_dims, activation)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # Apply spectral normalization if requested
        if use_spectral_norm:
            self._apply_spectral_norm()
    
    def _apply_spectral_norm(self):
        """Apply spectral normalization to all linear layers."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                setattr(
                    self, 
                    name, 
                    nn.utils.spectral_norm(module)
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute energy.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Energy values of shape (batch_size,)
        """
        features = super().forward(x)
        energy = self.output_layer(features)
        return energy.squeeze(-1)
```

## Convolutional Energy Models

For image data, convolutional architectures are more appropriate:

```python
class ConvEnergyModel(nn.Module):
    """Convolutional energy model for image data."""
    
    def __init__(
        self,
        input_channels: int,
        image_size: int,
        channels: List[int] = [32, 64, 128, 256],
        kernel_size: int = 3,
        activation: Optional[nn.Module] = None
    ):
        """Initialize convolutional energy model.
        
        Args:
            input_channels: Number of input channels
            image_size: Size of input images (assumed square)
            channels: List of channel dimensions for conv layers
            kernel_size: Size of convolutional kernel
            activation: Activation function
        """
        super().__init__()
        self.input_channels = input_channels
        self.image_size = image_size
        self.activation = activation or nn.LeakyReLU(0.2)
        
        # Build convolutional layers
        layers = []
        in_channels = input_channels
        
        for out_channels in channels:
            layers.append(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2
                )
            )
            layers.append(self.activation)
            in_channels = out_channels
        
        self.conv_net = nn.Sequential(*layers)
        
        # Calculate feature size after convolutions
        feature_size = image_size // (2 ** len(channels))
        
        # Final layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * feature_size * feature_size, 128),
            self.activation,
            nn.Linear(128, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute energy.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Energy values of shape (batch_size,)
        """
        # Ensure correct input shape
        if len(x.shape) == 2:
            x = x.view(-1, self.input_channels, self.image_size, self.image_size)
        
        features = self.conv_net(x)
        energy = self.fc(features)
        return energy.squeeze(-1)
```

## Neural Energy Functions

Neural networks can be used to create energy functions:

```python
from torchebm.core import BaseEnergyFunction


class NeuralEnergyFunction(BaseEnergyFunction):
    """Energy function implemented using a neural network."""

    def __init__(self, model: nn.Module):
        """Initialize neural energy function.
        
        Args:
            model: Neural network model
        """
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy values for inputs.
        
        Args:
            x: Input tensor
            
        Returns:
            Energy values
        """
        return self.model(x)
```

## Advanced Architectures

### Residual Blocks

```python
class ResidualBlock(nn.Module):
    """Residual block for energy models."""
    
    def __init__(self, dim: int, activation: nn.Module = nn.ReLU()):
        """Initialize residual block.
        
        Args:
            dim: Feature dimension
            activation: Activation function
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            activation,
            nn.Linear(dim, dim)
        )
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return x + self.block(x)

class ResNetEnergyModel(nn.Module):
    """ResNet-style energy model."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_blocks: int = 4,
        activation: nn.Module = nn.ReLU()
    ):
        """Initialize ResNet energy model.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            n_blocks: Number of residual blocks
            activation: Activation function
        """
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, activation)
            for _ in range(n_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet energy model.
        
        Args:
            x: Input tensor
            
        Returns:
            Energy values
        """
        h = self.input_proj(x)
        
        for block in self.blocks:
            h = block(h)
        
        energy = self.output_proj(h)
        return energy.squeeze(-1)
```

## Integration with Trainers

Models are integrated with trainer classes:

```python
class EBMTrainer:
    """Trainer for energy-based models."""
    
    def __init__(
        self,
        energy_function: BaseEnergyFunction,
        sampler: "Sampler",
        optimizer: torch.optim.Optimizer,
        loss_fn: "BaseLoss"
    ):
        """Initialize EBM trainer.
        
        Args:
            energy_function: Energy function to train
            sampler: Sampler for negative samples
            optimizer: Optimizer for model parameters
            loss_fn: BaseLoss function
        """
        self.energy_function = energy_function
        self.sampler = sampler
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
    def train_step(
        self,
        pos_samples: torch.Tensor,
        neg_samples: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Perform one training step.
        
        Args:
            pos_samples: Positive samples from data
            neg_samples: Optional negative samples
            
        Returns:
            Dictionary of metrics
        """
        # Generate negative samples if not provided
        if neg_samples is None:
            with torch.no_grad():
                neg_samples = self.sampler.sample(
                    n_samples=pos_samples.shape[0],
                    dim=pos_samples.shape[1]
                )
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute loss
        loss, metrics = self.loss_fn(pos_samples, neg_samples)
        
        # Backward and optimize
        loss.backward()
        self.optimizer.step()
        
        return metrics
```

## Performance Optimizations

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionEBMTrainer(EBMTrainer):
    """Trainer with mixed precision for faster training."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = GradScaler()
    
    def train_step(
        self,
        pos_samples: torch.Tensor,
        neg_samples: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Perform one training step with mixed precision."""
        # Generate negative samples if not provided
        if neg_samples is None:
            with torch.no_grad():
                neg_samples = self.sampler.sample(
                    n_samples=pos_samples.shape[0],
                    dim=pos_samples.shape[1]
                )
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            loss, metrics = self.loss_fn(pos_samples, neg_samples)
        
        # Backward and optimize with gradient scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return metrics
```

## Best Practices for Custom Models

When implementing custom models, follow these best practices:

<div class="grid" markdown>

<div markdown>
### Do

* Subclass appropriate base classes
* Handle device placement correctly
* Use proper initialization
* Consider normalization techniques
* Document architecture clearly
</div>

<div markdown>
### Don't

* Create overly complex architectures
* Ignore numerical stability
* Forget to validate inputs
* Mix different PyTorch versions
* Ignore gradient flow issues
</div>

</div>

!!! example "Custom Model Example"
    ```python
    class CustomEnergyModel(nn.Module):
        def __init__(self, input_dim, hidden_dim=128):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),  # SiLU/Swish activation
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            )
            
            # Initialize weights properly
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    nn.init.zeros_(m.bias)
        
        def forward(self, x):
            return self.network(x).squeeze(-1)
    ```

## Resources

<div class="grid cards" markdown>

-   :material-function:{ .lg .middle } __Core Components__

    ---

    Learn about core components and their interactions.

    [:octicons-arrow-right-24: Core Components](core_components.md)

-   :material-code-json:{ .lg .middle } __Energy Functions__

    ---

    Explore energy function implementation details.

    [:octicons-arrow-right-24: Energy Functions](implementation_energy.md)

-   :material-function-variant:{ .lg .middle } __Samplers__

    ---

    Understand sampler implementation details.

    [:octicons-arrow-right-24: Implementation Samplers](implementation_samplers.md)

</div> 