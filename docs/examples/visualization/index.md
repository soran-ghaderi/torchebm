---
title: Visualization Tools
description: Examples of visualization tools for energy functions and sampling
---

# Visualization Tools

This section demonstrates various visualization tools and techniques available in TorchEBM for visualizing energy functions and sampling processes.

## Basic Visualizations

### Contour Plots

The `contour_plots.py` example demonstrates basic contour plots for energy functions:

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

# Create contour plot
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, 50, cmap="viridis")
plt.colorbar(label="Energy")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Double Well Energy Landscape")
```

### Distribution Comparison

The `distribution_comparison.py` example compares sampled distributions to their ground truth:

```python
# Create figure with multiple plots
fig = plt.figure(figsize=(15, 5))

# Ground truth contour
ax1 = fig.add_subplot(131)
contour = ax1.contourf(X, Y, Z, 50, cmap="Blues")
fig.colorbar(contour, ax=ax1, label="Density")
ax1.set_title("Ground Truth Density")

# Sample density (using kernel density estimation)
ax2 = fig.add_subplot(132)
h = ax2.hist2d(samples_np[:, 0], samples_np[:, 1], bins=50, cmap="Reds", density=True)
fig.colorbar(h[3], ax=ax2, label="Density")
ax2.set_title("Sampled Distribution")

# Scatter plot of samples
ax3 = fig.add_subplot(133)
ax3.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5, s=3)
ax3.set_title("Sample Points")
```

## Advanced Visualizations

### Trajectory Animation

The `trajectory_animation.py` example visualizes sampling trajectories on energy landscapes:

```python
# Extract trajectory coordinates
traj_x = trajectory[0, :, 0].numpy()
traj_y = trajectory[0, :, 1].numpy()

# Plot trajectory with colormap based on step number
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, 50, cmap="viridis", alpha=0.7)  # Energy landscape
points = plt.scatter(
    traj_x, traj_y, c=np.arange(len(traj_x)), cmap="plasma", s=5, alpha=0.7
)
plt.colorbar(points, label="Sampling Step")

# Plot arrows to show direction of trajectory
step = 50  # Plot an arrow every 50 steps
plt.quiver(
    traj_x[:-1:step],
    traj_y[:-1:step],
    traj_x[1::step] - traj_x[:-1:step],
    traj_y[1::step] - traj_y[:-1:step],
    scale_units="xy",
    angles="xy",
    scale=1,
    color="red",
    alpha=0.7,
)
```

### Parallel Chains

The `parallel_chains.py` example visualizes multiple sampling chains:

```python
# Plot contour
plt.figure(figsize=(12, 10))
contour = plt.contourf(X, Y, Z, 50, cmap="viridis", alpha=0.7)
plt.colorbar(label="Energy")

# Plot each trajectory with a different color
colors = ["red", "blue", "green", "orange", "purple"]
for i in range(num_chains):
    traj_x = trajectories[i, :, 0].numpy()
    traj_y = trajectories[i, :, 1].numpy()

    plt.plot(traj_x, traj_y, alpha=0.7, linewidth=1, c=colors[i], label=f"Chain {i+1}")

    # Mark start and end points
    plt.scatter(traj_x[0], traj_y[0], c="black", s=50, marker="o")
    plt.scatter(traj_x[-1], traj_y[-1], c=colors[i], s=100, marker="*")
```

### Energy Over Time

The `energy_over_time.py` example tracks energy values during sampling:

```python
# Track the trajectory and energy manually
trajectory = torch.zeros((1, n_steps, dim))
energy_values = torch.zeros(n_steps)
current_sample = initial_point.clone()

# Run the sampling steps and store each position and energy
for i in range(n_steps):
    current_sample = sampler.langevin_step(
        current_sample, torch.randn_like(current_sample)
    )
    trajectory[:, i, :] = current_sample.clone().detach()
    energy_values[i] = energy_fn(current_sample).item()

# Plot energy evolution
plt.figure(figsize=(10, 6))
plt.plot(energy_values.numpy())
plt.xlabel("Step")
plt.ylabel("Energy")
plt.title("Energy Evolution During Sampling")
plt.grid(True, alpha=0.3)
```

## Common Visualization Utilities

The `utils.py` file provides common visualization functions that can be reused across examples:

```python
def plot_2d_energy_landscape(
    energy_fn, 
    title=None, 
    x_range=(-3, 3), 
    y_range=(-3, 3), 
    resolution=100,
    device="cpu",
    save_path=None
):
    """
    Plot a 2D energy landscape as a contour plot.
    
    Args:
        energy_fn: The energy function to visualize
        title: Optional title for the plot
        x_range: Range for x-axis (min, max)
        y_range: Range for y-axis (min, max)
        resolution: Number of points along each axis
        device: Device for tensor calculations
        save_path: Optional path to save the figure
    
    Returns:
        The figure object
    """
    # Implementation details...
    
def plot_sample_trajectories(
    trajectories, 
    energy_fn=None, 
    title=None,
    device="cpu",
    save_path=None
):
    """
    Plot sample trajectories on an energy landscape.
    
    Args:
        trajectories: Tensor of shape (n_samples, n_steps, dim)
        energy_fn: Optional energy function for background
        title: Optional title for the plot
        device: Device for tensor calculations
        save_path: Optional path to save the figure
    
    Returns:
        The figure object
    """
    # Implementation details...
```

## Running the Examples

To run these examples:

```bash
# List available visualization examples
python examples/main.py --list

# Run basic visualization examples
python examples/main.py visualization/basic/contour_plots
python examples/main.py visualization/basic/distribution_comparison

# Run advanced visualization examples
python examples/main.py visualization/advanced/trajectory_animation
python examples/main.py visualization/advanced/parallel_chains
python examples/main.py visualization/advanced/energy_over_time
```

## Additional Resources

For more information on visualization tools, see:

- [Matplotlib Documentation](https://matplotlib.org/)