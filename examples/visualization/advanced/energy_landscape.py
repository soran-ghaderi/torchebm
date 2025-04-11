import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from torchebm.core import RastriginEnergy
from pathlib import Path

# Create output directory
output_dir = Path("../../../docs/assets/images/visualization")
output_dir.mkdir(parents=True, exist_ok=True)

# Create energy function
energy_fn = RastriginEnergy(a=10.0)

# Create a grid
resolution = 200
x = np.linspace(-5, 5, resolution)
y = np.linspace(-5, 5, resolution)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Compute energy values
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32).unsqueeze(0)
        Z[i, j] = energy_fn(point).item()

# Apply log-scaling for better visualization
Z_vis = np.log(Z - np.min(Z) + 1)

# Create figure with two subplots
fig = plt.figure(figsize=(16, 6))

# 3D surface plot
ax1 = fig.add_subplot(121, projection="3d")
surf = ax1.plot_surface(X, Y, Z_vis, cmap=cm.viridis, linewidth=0, antialiased=True)
ax1.set_title("Rastrigin Energy Function (3D)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("Log Energy")
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.15, label="Log Energy")

# 2D contour plot
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X, Y, Z_vis, 50, cmap=cm.viridis)
ax2.set_title("Rastrigin Energy Function (Contour)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
fig.colorbar(contour, ax=ax2, label="Log Energy")

plt.tight_layout()

# Save figure
plt.savefig(output_dir / "advanced_energy_landscape.png", dpi=300, bbox_inches="tight")
print(f"Saved advanced_energy_landscape.png")
plt.close()
