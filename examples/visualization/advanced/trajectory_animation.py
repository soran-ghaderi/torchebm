import torch
import numpy as np
import matplotlib.pyplot as plt
from torchebm.core import DoubleWellEnergy
from torchebm.samplers.langevin_dynamics import LangevinDynamics
from pathlib import Path

# Create output directory
output_dir = Path("../../../docs/assets/images/visualization")
output_dir.mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)

# Create energy function and sampler
energy_fn = DoubleWellEnergy(barrier_height=2.0)
sampler = LangevinDynamics(energy_function=energy_fn, step_size=0.01)

# We'll manually track the trajectory in a 2D space
dim = 2
n_steps = 1000
initial_point = torch.tensor([[-2.0, 0.0]], dtype=torch.float32)

# Track the trajectory manually
trajectory = torch.zeros((1, n_steps, dim))
current_sample = initial_point

# Run the sampling steps and store each position
for i in range(n_steps):
    current_sample = sampler.langevin_step(
        current_sample, torch.randn_like(current_sample)
    )
    trajectory[:, i, :] = current_sample.clone().detach()

# Prepare the background energy landscape
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32).unsqueeze(0)
        Z[i, j] = energy_fn(point).item()

# Plot contour with trajectory
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, 50, cmap="viridis", alpha=0.7)
cbar1 = plt.colorbar(label="Energy")

# Extract trajectory coordinates
traj_x = trajectory[0, :, 0].numpy()
traj_y = trajectory[0, :, 1].numpy()

# Plot trajectory with colormap based on step number
points = plt.scatter(
    traj_x, traj_y, c=np.arange(len(traj_x)), cmap="plasma", s=5, alpha=0.7
)
cbar2 = plt.colorbar(points, label="Sampling Step")

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

plt.xlabel("x")
plt.ylabel("y")
plt.title("Langevin Dynamics Sampling Trajectory on Double Well Potential")
plt.tight_layout()

# Save figure
plt.savefig(output_dir / "langevin_trajectory.png", dpi=300, bbox_inches="tight")
print(f"Saved langevin_trajectory.png")
plt.close()
