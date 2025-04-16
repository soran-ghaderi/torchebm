"""
Generate visualization images for all dataset classes in the torchebm.datasets.generators module.
These images are used in the documentation to illustrate each dataset.
"""

import os
import torch

# import numpy as np # No longer strictly needed for generation, only potentially for plotting if not handled by matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl

from pathlib import Path
import warnings  # To handle potential warnings during generation if needed

# Import the new dataset classes
from torchebm.datasets.generators import (
    GaussianMixtureDataset,
    EightGaussiansDataset,
    TwoMoonsDataset,
    SwissRollDataset,
    CircleDataset,
    CheckerboardDataset,
    PinwheelDataset,
    GridDataset,
    # BaseSyntheticDataset # Not needed for generation itself
)


# Define the seed for reproducibility (passed to each dataset)
SEED = 42

# Create output directory for images
# Assuming the script is run from e.g., scripts/visualizations/
# Adjust the relative path if needed
output_dir = (
    Path(__file__).parent.parent.parent / "docs" / "assets" / "images" / "datasets"
)
# Or use an absolute path if preferred
# output_dir = Path("/path/to/your/project/docs/assets/images/datasets")

output_dir.mkdir(parents=True, exist_ok=True)  # Use parents=True for safety


# Function to generate and save dataset visualization
def visualize_and_save(data: torch.Tensor, title: str, filename: str, figsize=(8, 6)):
    """Visualizes a 2D dataset tensor and saves it to a file."""
    plt.figure(figsize=figsize)

    # Ensure data is on CPU and converted to NumPy for plotting
    # Matplotlib typically handles this, but being explicit is safer
    if data.requires_grad:
        data_np = data.detach().cpu().numpy()
    else:
        data_np = data.cpu().numpy()

    plt.scatter(data_np[:, 0], data_np[:, 1], s=5, alpha=0.6)
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.axis("equal")  # Maintain aspect ratio
    plt.grid(True, alpha=0.3)  # Add grid with transparency
    plt.tight_layout()  # Adjust plot to prevent labels overlapping

    save_path = output_dir / filename
    try:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Error saving {save_path}: {e}")

    # plt.show() # Usually commented out in scripts generating many plots
    plt.close()  # Close the figure to free memory


# --- Generate and save visualizations for each dataset ---

print(f"Generating dataset visualizations in: {output_dir}")

# 1. Gaussian Mixture
try:
    dataset = GaussianMixtureDataset(n_samples=3000, n_components=4, std=0.1, seed=SEED)
    data = dataset.get_data()
    visualize_and_save(data, "Gaussian Mixture (4 components)", "gaussian_mixture.png")
except Exception as e:
    print(f"Error generating Gaussian Mixture: {e}")

# 2. 8 Gaussians
try:
    dataset = EightGaussiansDataset(n_samples=3000, std=0.05, seed=SEED)
    data = dataset.get_data()
    visualize_and_save(data, "8 Gaussians", "eight_gaussians.png")
except Exception as e:
    print(f"Error generating 8 Gaussians: {e}")

# 3. Two Moons
try:
    dataset = TwoMoonsDataset(n_samples=3000, noise=0.05, seed=SEED)
    data = dataset.get_data()
    visualize_and_save(data, "Two Moons", "two_moons.png")
except Exception as e:
    print(f"Error generating Two Moons: {e}")

# 4. Swiss Roll
try:
    dataset = SwissRollDataset(n_samples=3000, noise=0.05, arclength=3.0, seed=SEED)
    data = dataset.get_data()
    visualize_and_save(data, "Swiss Roll", "swiss_roll.png")
except Exception as e:
    print(f"Error generating Swiss Roll: {e}")

# 5. Circle
try:
    dataset = CircleDataset(n_samples=1000, noise=0.05, radius=1.0, seed=SEED)
    data = dataset.get_data()
    visualize_and_save(data, "Circle", "circle.png")
except Exception as e:
    print(f"Error generating Circle: {e}")

# 6. Checkerboard
try:
    # Increase samples slightly if needed, as rejection sampling might be less dense
    dataset = CheckerboardDataset(
        n_samples=10000, range_limit=3.0, noise=0.01, seed=SEED
    )
    data = dataset.get_data()
    visualize_and_save(data, "Checkerboard", "checkerboard.png")
except Exception as e:
    print(f"Error generating Checkerboard: {e}")

# 7. Pinwheel
try:
    dataset = PinwheelDataset(
        n_samples=3000,
        n_classes=5,
        noise=0.05,
        radial_scale=1.0,  # Adjusted to match original call
        angular_scale=0.1,  # Adjusted to match original call
        spiral_scale=1.2,  # Adjusted to match original call
        seed=SEED,
    )
    data = dataset.get_data()
    visualize_and_save(data, "Pinwheel (5 blades)", "pinwheel.png")
except Exception as e:
    print(f"Error generating Pinwheel: {e}")

# 8. 2D Grid
try:
    n_dim = 10
    dataset = GridDataset(
        n_samples_per_dim=n_dim, range_limit=1.0, noise=0.01, seed=SEED
    )
    data = dataset.get_data()
    # Update title to reflect actual dimensions
    visualize_and_save(data, f"2D Grid ({n_dim}x{n_dim})", "grid.png")
except Exception as e:
    print(f"Error generating 2D Grid: {e}")

print("-" * 30)
print("Dataset visualization generation complete.")
print(f"Images saved in: {output_dir}")
print("-" * 30)

# """
# Generate visualization images for all datasets in the torchebm.datasets.generators module.
# These images are used in the documentation to illustrate each dataset.
# """
#
# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
#
# from torchebm.datasets.generators import (
#     make_gaussian_mixture,
#     make_8gaussians,
#     make_two_moons,
#     make_swiss_roll,
#     make_circle,
#     make_checkerboard,
#     make_pinwheel,
#     make_2d_grid,
# )
#
# # Set seed for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)
#
# # Create output directory for images
# output_dir = Path("../../docs/assets/images/datasets")
# output_dir.mkdir(parents=False, exist_ok=True)
#
#
# # Function to generate and save dataset visualization
# def visualize_and_save(data, title, filename, figsize=(8, 6)):
#     plt.figure(figsize=figsize)
#     plt.scatter(data[:, 0], data[:, 1], s=5, alpha=0.6)
#     plt.title(title)
#     plt.axis("equal")
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#
#     plt.savefig(output_dir / filename, dpi=150)
#     plt.show()
#     plt.close()
#
#
# # Generate and save visualizations for each dataset
#
# # 1. Gaussian Mixture
# data = make_gaussian_mixture(n_samples=3000, n_components=4, std=0.1)
# visualize_and_save(data, "Gaussian Mixture (4 components)", "gaussian_mixture.png")
#
# # 2. 8 Gaussians
# data = make_8gaussians(n_samples=3000, std=0.05)
# visualize_and_save(data, "8 Gaussians", "eight_gaussians.png")
#
# # 3. Two Moons
# data = make_two_moons(n_samples=3000, noise=0.05)
# visualize_and_save(data, "Two Moons", "two_moons.png")
#
# # 4. Swiss Roll
# data = make_swiss_roll(n_samples=3000, noise=0.05, arclength=3.0)
# visualize_and_save(data, "Swiss Roll", "swiss_roll.png")
#
# # 5. Circle
# data = make_circle(n_samples=1000, noise=0.05, radius=1.0)
# visualize_and_save(data, "Circle", "circle.png")
#
# # 6. Checkerboard
# data = make_checkerboard(n_samples=10000, range_limit=3.0, noise=0.01)
# visualize_and_save(data, "Checkerboard", "checkerboard.png")
#
# # 7. Pinwheel
# data = make_pinwheel(
#     n_samples=3000,
#     n_classes=5,
#     noise=0.05,
#     radial_scale=1.0,
#     angular_scale=0.1,
#     spiral_scale=1.2,
# )
# visualize_and_save(data, "Pinwheel (5 blades)", "pinwheel.png")
#
# # 8. 2D Grid
# data = make_2d_grid(n_samples_per_dim=35, range_limit=1.0, noise=0.02)
# visualize_and_save(data, "2D Grid (15×15)", "grid.png")
#
# print(f"Generated dataset visualization images in {output_dir}")
