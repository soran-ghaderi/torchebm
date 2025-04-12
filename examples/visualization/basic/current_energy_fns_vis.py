import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from pathlib import Path

from torchebm.core.base_energy_function import (
    RosenbrockEnergy,
    AckleyEnergy,
    RastriginEnergy,
    DoubleWellEnergy,
    GaussianEnergy,
    HarmonicEnergy,
)

# Create output directory
output_dir = Path("../../../docs/assets/images/e_functions")
output_dir.mkdir(parents=True, exist_ok=True)


def plot_energy_function(energy_fn, x_range, y_range, title, filename):
    """
    Plot an energy function as both a 3D surface and a 2D contour plot

    Args:
        energy_fn: The energy function to visualize
        x_range: Range for x-axis (min, max)
        y_range: Range for y-axis (min, max)
        title: Title for the plot
        filename: Filename to save the plot (without extension)
    """
    # Create high-resolution grid
    resolution = 200
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Compute energy values
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32).unsqueeze(0)
            Z[i, j] = energy_fn(point).item()

    # Apply logarithmic scaling for better visualization (optional)
    # Shift to make all values positive
    if np.min(Z) < 0:
        Z_vis = Z - np.min(Z) + 1
    else:
        Z_vis = Z + 1

    # Optional: use log scale for better visualization
    Z_vis = np.log(Z_vis)

    # Create a figure with two subplots (3D surface and 2D contour)
    fig = plt.figure(figsize=(12, 5))

    # 3D surface plot
    ax1 = fig.add_subplot(121, projection="3d")
    surf = ax1.plot_surface(
        X, Y, Z_vis, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.8
    )
    ax1.set_title(f"{title} - 3D Surface")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("Log Energy")

    # 2D contour plot with more levels for better detail
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z_vis, 50, cmap=cm.viridis)
    ax2.set_title(f"{title} - Contour Plot")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig.colorbar(contour, ax=ax2, label="Log Energy")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300, bbox_inches="tight")
    print(f"Saved {filename}.png")
    plt.close()  # Close the figure to free memory


# Define energy functions with appropriate ranges and titles
energy_functions = [
    (RosenbrockEnergy(), [-2, 2], [-1, 3], "Rosenbrock Energy", "rosenbrock"),
    (AckleyEnergy(), [-5, 5], [-5, 5], "Ackley Energy", "ackley"),
    (RastriginEnergy(), [-5, 5], [-5, 5], "Rastrigin Energy", "rastrigin"),
    (DoubleWellEnergy(), [-2, 2], [-2, 2], "Double Well Energy", "double_well"),
    (
        GaussianEnergy(
            torch.tensor([0.0, 0.0]), torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        ),
        [-3, 3],
        [-3, 3],
        "Gaussian Energy",
        "gaussian",
    ),
    (HarmonicEnergy(), [-3, 3], [-3, 3], "Harmonic Energy", "harmonic"),
]

# Generate and save each energy function visualization
for energy_fn, x_range, y_range, title, filename in energy_functions:
    plot_energy_function(energy_fn, x_range, y_range, title, filename)

print("All energy function visualizations have been generated!")
