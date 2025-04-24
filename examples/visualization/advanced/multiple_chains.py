import torch
import numpy as np
import matplotlib.pyplot as plt
from torchebm.core import RastriginEnergy
from torchebm.samplers.langevin_dynamics import LangevinDynamics
from pathlib import Path

# Create output directory
output_dir = Path("../../../docs/assets/images/visualization")
output_dir.mkdir(parents=True, exist_ok=True)

# Create energy function and sampler
energy_fn = RastriginEnergy(a=10.0)
sampler = LangevinDynamics(energy_function=energy_fn, step_size=0.01)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Parameters for sampling
dim = 2
n_steps = 1000
num_chains = 5

# Generate random starting points
initial_points = torch.randn(num_chains, dim) * 3

# Track the trajectories manually
trajectories = torch.zeros((num_chains, n_steps, dim))
current_samples = initial_points.clone()

# Run the sampling steps and store each position
for i in range(n_steps):
    current_samples = sampler.langevin_step(
        current_samples, torch.randn_like(current_samples)
    )
    trajectories[:, i, :] = current_samples.clone().detach()

# Create background contour
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32).unsqueeze(0)
        Z[i, j] = energy_fn(point).item()

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

plt.xlabel("x")
plt.ylabel("y")
plt.title("Multiple Langevin Dynamics Sampling Chains on Rastrigin Potential")
plt.legend()
plt.tight_layout()

# Save figure
plt.savefig(output_dir / "multiple_chains.png", dpi=300, bbox_inches="tight")
print(f"Saved multiple_chains.png")
plt.close()
