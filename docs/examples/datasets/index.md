---
title: Dataset Classes
description: Using TorchEBM's dataset classes for synthetic 2D distributions
#icon: material/database
---

# Dataset Classes

The `torchebm` library provides a variety of 2D synthetic datasets through the [`torchebm.datasets`][datasets_module] module. These datasets are implemented as PyTorch `Dataset` classes for easy integration with DataLoaders. This walkthrough explores each dataset class with examples and visualizations.

[datasets_module]: ../../api/torchebm/datasets/generators/index.md

## Setup

First, let's import the necessary packages:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchebm.datasets import (
    GaussianMixtureDataset, EightGaussiansDataset, TwoMoonsDataset,
    SwissRollDataset, CircleDataset, CheckerboardDataset,
    PinwheelDataset, GridDataset
)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Helper function to visualize a dataset
def visualize_dataset(data, title, figsize=(5, 5)):
    plt.figure(figsize=figsize)
    plt.scatter(data[:, 0], data[:, 1], s=5, alpha=0.6)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
```

## Dataset Types

### 1. Gaussian Mixture

!!! info "Gaussian Mixture"

    Generate points from a mixture of Gaussian distributions arranged in a circle.
    
    This dataset generator is useful for testing mode-seeking behavior in energy-based models.

    **Parameters**:
    
    - `n_samples`: Number of samples to generate
    - `n_components`: Number of Gaussian components (modes)
    - `std`: Standard deviation of each Gaussian
    - `radius`: Radius of the circle on which centers are placed

```python
# Generate 1000 samples from a 6-component Gaussian mixture
gmm_dataset = GaussianMixtureDataset(
    n_samples=1000, 
    n_components=6, 
    std=0.05, 
    radius=1.0, 
    seed=42
)
gmm_data = gmm_dataset.get_data()
visualize_dataset(gmm_data, "Gaussian Mixture (6 components)")
```

<figure markdown>
  ![Gaussian Mixture](../../assets/images/datasets/gaussian_mixture.png){ width="400" }
  <figcaption>Gaussian Mixture with 6 components</figcaption>
</figure>

### 2. Eight Gaussians

!!! info "Eight Gaussians"

    A specific case of Gaussian mixture with 8 components arranged at compass and diagonal points.
    
    This is a common benchmark distribution in energy-based modeling literature.

    **Parameters**:
    
    - `n_samples`: Number of samples to generate
    - `std`: Standard deviation of each component
    - `scale`: Scaling factor for the centers

```python
# Generate 1000 samples from the 8 Gaussians distribution
eight_gauss_dataset = EightGaussiansDataset(
    n_samples=1000, 
    std=0.02, 
    scale=2.0, 
    seed=42
)
eight_gauss_data = eight_gauss_dataset.get_data()
visualize_dataset(eight_gauss_data, "Eight Gaussians")
```

<figure markdown>
  ![Eight Gaussians](../../assets/images/datasets/eight_gaussians.png){ width="400" }
  <figcaption>Eight Gaussians distribution</figcaption>
</figure>

### 3. Two Moons

!!! info "Two Moons"

    Generate the classic "two moons" dataset with two interleaving half-circles.
    
    This dataset is excellent for testing classification, clustering, and density estimation algorithms 
    due to its non-linear separation boundary.

    **Parameters**:
    
    - `n_samples`: Number of samples to generate
    - `noise`: Standard deviation of Gaussian noise added

```python
# Generate 1000 samples from the Two Moons distribution
moons_dataset = TwoMoonsDataset(
    n_samples=1000, 
    noise=0.05, 
    seed=42
)
moons_data = moons_dataset.get_data()
visualize_dataset(moons_data, "Two Moons")
```

<figure markdown>
  ![Two Moons](../../assets/images/datasets/two_moons.png){ width="400" }
  <figcaption>Two Moons distribution</figcaption>
</figure>

### 4. Swiss Roll

!!! info "Swiss Roll"

    Generate the 2D Swiss roll dataset with a spiral structure.
    
    The Swiss roll is a classic example of a nonlinear manifold.

    **Parameters**:
    
    - `n_samples`: Number of samples to generate
    - `noise`: Standard deviation of Gaussian noise added
    - `arclength`: Controls how many rolls (pi*arclength)

```python
# Generate 1000 samples from the Swiss Roll distribution
swiss_roll_dataset = SwissRollDataset(
    n_samples=1000, 
    noise=0.05, 
    arclength=3.0, 
    seed=42
)
swiss_roll_data = swiss_roll_dataset.get_data()
visualize_dataset(swiss_roll_data, "Swiss Roll")
```

<figure markdown>
  ![Swiss Roll](../../assets/images/datasets/swiss_roll.png){ width="400" }
  <figcaption>Swiss Roll distribution</figcaption>
</figure>

### 5. Circle

!!! info "Circle"

    Generate points uniformly distributed on a circle with optional noise.
    
    This simple distribution is useful for testing density estimation on a 1D manifold embedded in 2D space.

    **Parameters**:
    
    - `n_samples`: Number of samples to generate
    - `noise`: Standard deviation of Gaussian noise added
    - `radius`: Radius of the circle

```python
# Generate 1000 samples from a Circle distribution
circle_dataset = CircleDataset(
    n_samples=1000, 
    noise=0.05, 
    radius=1.0, 
    seed=42
)
circle_data = circle_dataset.get_data()
visualize_dataset(circle_data, "Circle")
```

<figure markdown>
  ![Circle](../../assets/images/datasets/circle.png){ width="400" }
  <figcaption>Circle distribution</figcaption>
</figure>

### 6. Checkerboard

!!! info "Checkerboard"

    Generate points in a 2D checkerboard pattern with alternating high and low density regions.
    
    The checkerboard pattern creates multiple modes in a regular structure, challenging an EBM's ability 
    to capture complex multimodal distributions.

    **Parameters**:
    
    - `n_samples`: Target number of samples
    - `range_limit`: Defines the square region [-lim, lim] x [-lim, lim]
    - `noise`: Small Gaussian noise added to points

```python
# Generate 1000 samples from a Checkerboard distribution
checkerboard_dataset = CheckerboardDataset(
    n_samples=1000, 
    range_limit=4.0, 
    noise=0.01, 
    seed=42
)
checkerboard_data = checkerboard_dataset.get_data()
visualize_dataset(checkerboard_data, "Checkerboard")
```

<figure markdown>
  ![Checkerboard](../../assets/images/datasets/checkerboard.png){ width="400" }
  <figcaption>Checkerboard distribution</figcaption>
</figure>

### 7. Pinwheel

!!! info "Pinwheel"

    Generate the pinwheel dataset with curved blades spiraling outward.
    
    The pinwheel dataset is highly configurable:
    
    - Adjust the number of blades with `n_classes`
    - Control blade length with `radial_scale`
    - Control blade thickness with `angular_scale`
    - Control how tightly the blades spiral with `spiral_scale`

    **Parameters**:
    
    - `n_samples`: Number of samples to generate
    - `n_classes`: Number of 'blades' in the pinwheel
    - `noise`: Standard deviation of Gaussian noise
    - `radial_scale`: Scales the maximum radius of the points
    - `angular_scale`: Controls blade thickness
    - `spiral_scale`: Controls how tightly blades spiral

```python
# Generate 1000 samples from a Pinwheel distribution with 5 blades
pinwheel_dataset = PinwheelDataset(
    n_samples=1000, 
    n_classes=5, 
    noise=0.05, 
    radial_scale=2.0,
    angular_scale=0.1,
    spiral_scale=5.0,
    seed=42
)
pinwheel_data = pinwheel_dataset.get_data()
visualize_dataset(pinwheel_data, "Pinwheel (5 blades)")
```

<figure markdown>
  ![Pinwheel](../../assets/images/datasets/pinwheel.png){ width="400" }
  <figcaption>Pinwheel distribution with 5 blades</figcaption>
</figure>

### 8. 2D Grid

!!! info "2D Grid"

    Generate points on a regular 2D grid with optional noise.
    
    This is useful for creating test points to evaluate model predictions across a regular spatial arrangement.

    **Parameters**:
    
    - `n_samples_per_dim`: Number of points along each dimension
    - `range_limit`: Defines the square region [-lim, lim] x [-lim, lim]
    - `noise`: Standard deviation of Gaussian noise added

```python
# Generate a 20x20 grid of points
grid_dataset = GridDataset(
    n_samples_per_dim=20, 
    range_limit=1.0, 
    noise=0.01, 
    seed=42
)
grid_data = grid_dataset.get_data()
visualize_dataset(grid_data, "2D Grid (20x20)")
```

<figure markdown>
  ![2D Grid](../../assets/images/datasets/grid.png){ width="400" }
  <figcaption>2D Grid with 20x20 points</figcaption>
</figure>

## Usage Examples

### Using with DataLoader

One of the key advantages of the dataset classes is their compatibility with PyTorch's DataLoader for efficient batch processing:

```python
from torch.utils.data import DataLoader

# Create a dataset
dataset = GaussianMixtureDataset(n_samples=2000, n_components=8, std=0.1, seed=42)

# Create a DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True
)

# Iterate through batches
for batch in dataloader:
    # Each batch is a tensor of batch_shape [batch_size, 2]
    print(f"Batch batch_shape: {batch.shape}")
    # Process the batch...
    break  # Just showing the first batch
```

### Comparing Multiple Datasets

You can easily generate and compare multiple datasets:

=== "Code"

    ```python
    # Create a figure with multiple datasets
    plt.figure(figsize=(15, 10))
    
    # Generate datasets
    datasets = [
        (GaussianMixtureDataset(1000, 8, 0.05, seed=42).get_data(), "Gaussian Mixture"),
        (TwoMoonsDataset(1000, 0.05, seed=42).get_data(), "Two Moons"),
        (SwissRollDataset(1000, 0.05, seed=42).get_data(), "Swiss Roll"),
        (CircleDataset(1000, 0.05, seed=42).get_data(), "Circle"),
        (CheckerboardDataset(1000, 4.0, 0.01, seed=42).get_data(), "Checkerboard"),
        (PinwheelDataset(1000, 5, 0.05, seed=42).get_data(), "Pinwheel")
    ]
    
    # Plot each dataset
    for i, (data, title) in enumerate(datasets):
        plt.subplot(2, 3, i+1)
        plt.scatter(data[:, 0], data[:, 1], s=3, alpha=0.6)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    ```

=== "Output"
    
    <figure markdown>
      <div class="grid cards" markdown>
      - ![Gaussian Mixture](../../assets/images/datasets/gaussian_mixture.png){ width="200" }
      - ![Two Moons](../../assets/images/datasets/two_moons.png){ width="200" }
      - ![Swiss Roll](../../assets/images/datasets/swiss_roll.png){ width="200" }
      - ![Circle](../../assets/images/datasets/circle.png){ width="200" }
      - ![Checkerboard](../../assets/images/datasets/checkerboard.png){ width="200" }
      - ![Pinwheel](../../assets/images/datasets/pinwheel.png){ width="200" }
      </div>
    </figure>

### Device Support

All dataset classes support placing tensors directly on specific devices:

```python
# Generate data on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_dataset = GaussianMixtureDataset(1000, 4, 0.1, device=device, seed=42)
gpu_data = gpu_dataset.get_data()
print(f"Data is on: {gpu_data.device}")
```

## Training Example

Here's a simplified example of using these datasets for training an energy-based model, similar to what's shown in the [mlp_cd_training.py](../../examples/training_models/mlp_cd_training.md) example:

=== "Complete Example"

    ```python
    # Imports
    from torchebm.core import BaseEnergyFunction
    from torchebm.samplers import LangevinDynamics
    from torchebm.losses import ContrastiveDivergence
    import torch.nn as nn
    import torch.optim as optim
    
    # Define an energy function
    class MLPEnergy(BaseEnergyFunction):
        def __init__(self, input_dim=2, hidden_dim=64):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        
        def forward(self, x):
            return self.network(x).squeeze(-1)
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset directly with device specification
    dataset = TwoMoonsDataset(n_samples=3000, noise=0.05, seed=42, device=device)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
    
    # Model components
    energy_model = MLPEnergy(input_dim=2, hidden_dim=16).to(device)
    sampler = LangevinDynamics(
        energy_function=energy_model,
        step_size=0.1, 
        noise_scale=0.1,
        device=device
    )
    loss_fn = ContrastiveDivergence(
        energy_function=energy_model,
        sampler=sampler,
        n_steps=10
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(energy_model.parameters(), lr=1e-3)
    
    # Training loop (simplified)
    for epoch in range(5):  # Just a few epochs for demonstration
        for data_batch in dataloader:
            optimizer.zero_grad()
            loss, _ = loss_fn(data_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    ```

=== "Key Components"

    - **Dataset**: `TwoMoonsDataset` placed directly on device
    - **Energy Function**: Simple MLP implementing `BaseEnergyFunction`
    - **Sampler**: `LangevinDynamics` for generating samples
    - **Loss**: `ContrastiveDivergence` for EBM training
    - **Training Loop**: Standard PyTorch pattern with DataLoader

For more detailed examples, see [Training Energy Models](../../training_guide/index.md).

## Summary

!!! success "Key Features"

    - **Dataset Variety**: 8 distinct 2D distributions for different testing scenarios
    - **PyTorch Integration**: Built as `torch.utils.data.Dataset` subclasses
    - **Device Support**: Create datasets directly on CPU or GPU
    - **Configurability**: Extensive parameterization for all distributions
    - **Reproducibility**: Seed support for deterministic generation

These dataset classes provide diverse 2D distributions for testing energy-based models. Each distribution has different characteristics that can challenge different aspects of model learning:

| Dataset | Testing Focus |
| ------- | ------------- |
| Gaussian Mixtures | Mode-seeking behavior |  
| Two Moons | Non-linear decision boundaries |
| Swiss Roll & Circle | Manifold learning capabilities |
| Checkerboard | Multiple modes in regular patterns |
| Pinwheel | Complex spiral structure with varying density |

The class-based implementation provides seamless integration with PyTorch's DataLoader system, making it easy to incorporate these datasets into your training pipeline.

## See Also

- [Energy Function Implementations](../../api/core/energy_functions.md)
- [Sampler Options](../../api/samplers/index.md)
- [Training Guide](../../training_guide/index.md)
- [EBM Applications](../../examples/applications/index.md) 