import torch
import numpy as np
import matplotlib.pyplot as plt
from torchebm.core import DoubleWellEnergy
from pathlib import Path

# Create output directory
output_dir = Path("../../../docs/assets/images/visualization")
output_dir.mkdir(parents=True, exist_ok=True)

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
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Energy")
ax.set_title("Double Well Energy Landscape")
cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
cbar.set_label("Energy")
plt.tight_layout()

# Save figure
plt.savefig(output_dir / "basic_energy_landscape.png", dpi=300, bbox_inches="tight")
print(f"Saved basic_energy_landscape.png")
plt.close()
