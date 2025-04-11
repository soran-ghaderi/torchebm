import numpy as np
import torch
import matplotlib.pyplot as plt
from torchebm.core import DoubleWellEnergy
from pathlib import Path

# Create output directory
output_dir = Path("../../../docs/assets/images/visualization")
output_dir.mkdir(parents=True, exist_ok=True)

# Since we can't show interactive plots in the documentation, we'll create multiple plots
# with different barrier heights and combine them into a single figure

# Create a grid for visualization
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Create barrier height values
barrier_heights = [0.5, 1.0, 2.0, 4.0]

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

# Calculate energy landscapes for different barrier heights
for i, barrier_height in enumerate(barrier_heights):
    # Create energy function with the specified barrier height
    energy_fn = DoubleWellEnergy(barrier_height=barrier_height)

    # Compute energy values
    for j in range(X.shape[0]):
        for k in range(X.shape[1]):
            point = torch.tensor([X[j, k], Y[j, k]], dtype=torch.float32).unsqueeze(0)
            Z[j, k] = energy_fn(point).item()

    # Create contour plot
    contour = axes[i].contourf(X, Y, Z, 50, cmap="viridis")
    fig.colorbar(contour, ax=axes[i], label="Energy")
    axes[i].set_xlabel("x")
    axes[i].set_ylabel("y")
    axes[i].set_title(f"Double Well Energy (Barrier Height = {barrier_height})")

plt.tight_layout()

# Save figure
plt.savefig(output_dir / "interactive_visualization.png", dpi=300, bbox_inches="tight")
print(f"Saved interactive_visualization.png")
plt.close()
