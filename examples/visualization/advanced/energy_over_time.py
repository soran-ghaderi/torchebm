import torch
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

# Parameters for sampling
dim = 2
n_steps = 1000
initial_point = torch.tensor([[-2.0, 0.0]], dtype=torch.float32)

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

# Convert to numpy for plotting
energy_values_np = energy_values.numpy()

# Plot energy evolution
plt.figure(figsize=(10, 6))
plt.plot(energy_values_np)
plt.xlabel("Step")
plt.ylabel("Energy")
plt.title("Energy Evolution During Langevin Dynamics Sampling")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save figure
plt.savefig(output_dir / "energy_evolution.png", dpi=300, bbox_inches="tight")
print(f"Saved energy_evolution.png")
plt.close()
