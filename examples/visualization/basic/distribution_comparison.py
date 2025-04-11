import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from torchebm.core import GaussianEnergy
from torchebm.samplers.langevin_dynamics import LangevinDynamics
from pathlib import Path

# Create output directory
output_dir = Path("../../../docs/assets/images/visualization")
output_dir.mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create a Gaussian energy function
mean = torch.tensor([1.0, -1.0])
cov = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
energy_fn = GaussianEnergy(mean=mean, cov=cov)

# Sample using Langevin dynamics
sampler = LangevinDynamics(energy_function=energy_fn, step_size=0.01)

# Generate samples
n_samples = 5000
dim = 2
n_steps = 1000
burn_in = 200

# Initialize random samples
x = torch.randn(n_samples, dim)

# Run sampling for burn-in period (discard these samples)
for i in range(burn_in):
    x = sampler.langevin_step(x, torch.randn_like(x))

# Run sampling for the desired number of steps
for i in range(n_steps):
    x = sampler.langevin_step(x, torch.randn_like(x))

# Final samples
samples = x

# Convert to numpy for visualization
samples_np = samples.numpy()
mean_np = mean.numpy()
cov_np = cov.numpy()

# Create a grid for the ground truth density
x = np.linspace(-3, 5, 100)
y = np.linspace(-5, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Calculate multivariate normal PDF
rv = stats.multivariate_normal(mean_np, cov_np)
Z = rv.pdf(pos)

# Create figure with multiple plots
fig = plt.figure(figsize=(15, 5))

# Ground truth contour
ax1 = fig.add_subplot(131)
contour = ax1.contourf(X, Y, Z, 50, cmap="Blues")
fig.colorbar(contour, ax=ax1, label="Density")
ax1.set_title("Ground Truth Density")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# Sample density (using kernel density estimation)
ax2 = fig.add_subplot(132)
h = ax2.hist2d(samples_np[:, 0], samples_np[:, 1], bins=50, cmap="Reds", density=True)
fig.colorbar(h[3], ax=ax2, label="Density")
ax2.set_title("Sampled Distribution")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

# Scatter plot of samples
ax3 = fig.add_subplot(133)
ax3.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5, s=3)
ax3.set_title("Sample Points")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_xlim(ax2.get_xlim())
ax3.set_ylim(ax2.get_ylim())

plt.tight_layout()

# Save figure
plt.savefig(output_dir / "distribution_comparison.png", dpi=300, bbox_inches="tight")
print(f"Saved distribution_comparison.png")
plt.close()
