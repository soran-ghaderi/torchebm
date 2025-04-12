---
title: Energy Functions
description: Examples of various energy functions and their visualization
---

# Energy Function Examples

This section demonstrates the various energy functions available in TorchEBM and how to visualize them.

## Basic Energy Landscapes

The `landscape_2d.py` example shows how to create and visualize basic energy functions:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchebm.core import DoubleWellEnergy

# Create the energy function
energy_fn = DoubleWellEnergy(barrier_height=2.0)

# Create a grid for visualization
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Compute energy values
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32).unsqueeze(0)
        Z[i, j] = energy_fn(point).item()

# Create 3D surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
```

## Multimodal Energy Functions

The `multimodal.py` example demonstrates more complex energy functions with multiple local minima:

```python
class MultimodalEnergy:
    """
    A 2D energy function with multiple local minima to demonstrate sampling behavior.
    """
    def __init__(self, device=None, dtype=torch.float32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Define centers and weights for multiple Gaussian components
        self.centers = torch.tensor(
            [[-1.0, -1.0], [1.0, 1.0], [-0.5, 1.0], [1.0, -0.5]],
            device=self.device,
            dtype=self.dtype,
        )

        self.weights = torch.tensor(
            [1.0, 0.8, 0.6, 0.7], device=self.device, dtype=self.dtype
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate energy as negative log of mixture of Gaussians
        dists = torch.cdist(x, self.centers)
        energy = -torch.log(
            torch.sum(self.weights * torch.exp(-0.5 * dists.pow(2)), dim=-1)
        )
        return energy
```

## Parametric Energy Functions

The `parametric.py` example shows how to create energy functions with adjustable parameters:

```python
# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

# Calculate energy landscapes for different barrier heights
barrier_heights = [0.5, 1.0, 2.0, 4.0]

for i, barrier_height in enumerate(barrier_heights):
    # Create energy function with the specified barrier height
    energy_fn = DoubleWellEnergy(barrier_height=barrier_height)
    
    # Compute energy values
    # ...
    
    # Create contour plot
    contour = axes[i].contourf(X, Y, Z, 50, cmap="viridis")
    fig.colorbar(contour, ax=axes[i], label="Energy")
    axes[i].set_title(f"Double Well Energy (Barrier Height = {barrier_height})")
```

## Running the Examples

To run these examples:

```bash
# List available energy function examples
python examples/main.py --list

# Run a specific example
python examples/main.py core/energy_functions/landscape_2d
python examples/main.py core/energy_functions/multimodal
python examples/main.py core/energy_functions/parametric
```

## Additional Resources

For more information on energy functions, see:

- [API Reference: Energy Functions](../../api/torchebm/core/energy_function/index.md)
- [Guide: Energy Functions](../../guides/energy_functions.md) 