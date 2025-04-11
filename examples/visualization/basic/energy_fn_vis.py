#!/usr/bin/env python
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import torchebm
import sys

sys.path.append("../../..")

from torchebm.core import (
    GaussianEnergy,
    DoubleWellEnergy,
    RosenbrockEnergy,
    RastriginEnergy,
    AckleyEnergy,
)
from torchebm.utils.visualization import (
    plot_2d_energy_landscape,
    plot_3d_energy_landscape,
    plot_samples_on_energy,
    plot_sample_trajectories,
)
from torchebm.samplers.langevin_dynamics import LangevinDynamics

# Create output directory if it doesn't exist
os.makedirs("../../../docs/assets/images/e_functions", exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate 2D visualizations for all energy functions

# Gaussian Energy
gaussian_energy = GaussianEnergy(
    mean=torch.tensor([0.0, 0.0]), cov=torch.tensor([[1.0, 0.5], [0.5, 1.0]])
)
fig = plot_2d_energy_landscape(
    energy_fn=gaussian_energy,
    title="Gaussian Energy",
    device=device,
    save_path="../../../docs/assets/images/e_functions/gaussian.png",
)
plt.close(fig)

# Double Well Energy
double_well_energy = DoubleWellEnergy(barrier_height=2.0)
fig = plot_2d_energy_landscape(
    energy_fn=double_well_energy,
    title="Double Well Energy",
    device=device,
    save_path="../../../docs/assets/images/e_functions/double_well.png",
)
plt.close(fig)

# Rosenbrock Energy
rosenbrock_energy = RosenbrockEnergy(a=1.0, b=100.0)
fig = plot_2d_energy_landscape(
    energy_fn=rosenbrock_energy,
    x_range=(-2, 2),
    y_range=(-1, 3),
    title="Rosenbrock Energy",
    device=device,
    save_path="../../../docs/assets/images/e_functions/rosenbrock.png",
)
plt.close(fig)

# Rastrigin Energy
rastrigin_energy = RastriginEnergy(a=10.0)
fig = plot_2d_energy_landscape(
    energy_fn=rastrigin_energy,
    x_range=(-5.12, 5.12),
    y_range=(-5.12, 5.12),
    title="Rastrigin Energy",
    device=device,
    save_path="../../../docs/assets/images/e_functions/rastrigin.png",
)
plt.close(fig)

# Ackley Energy
ackley_energy = AckleyEnergy(a=20.0, b=0.2, c=2 * np.pi)
fig = plot_2d_energy_landscape(
    energy_fn=ackley_energy,
    x_range=(-5, 5),
    y_range=(-5, 5),
    title="Ackley Energy",
    device=device,
    save_path="../../../docs/assets/images/e_functions/ackley.png",
)
plt.close(fig)

# Generate sampling visualizations

# Gaussian sampling
sampler = LangevinDynamics(
    energy_function=gaussian_energy, step_size=0.01, noise_scale=1.0
)

# Initial states far from the mean
initial_states = torch.tensor(
    [[-3.0, -3.0], [3.0, 3.0], [-3.0, 3.0], [3.0, -3.0]], dtype=torch.float32
)

# Run sampling with trajectory tracking
samples, trajectories = sampler.sample_chain(
    x=initial_states, n_steps=200, return_trajectory=True
)

# Plot trajectories
fig = plot_sample_trajectories(
    trajectories=trajectories,
    energy_fn=gaussian_energy,
    title="Sampling Trajectories on Gaussian Energy",
    device=device,
    save_path="../docs/assets/images/e_functions/gaussian_trajectories.png",
)
plt.close(fig)

# Double well sampling
sampler = LangevinDynamics(
    energy_function=double_well_energy, step_size=0.01, noise_scale=1.0
)

# Initial states in the middle of the barrier
initial_states = torch.tensor(
    [[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5], [0.0, 0.5]], dtype=torch.float32
)

# Run sampling with trajectory tracking
samples, trajectories = sampler.sample_chain(
    x=initial_states, n_steps=300, return_trajectory=True
)

# Plot trajectories
fig = plot_sample_trajectories(
    trajectories=trajectories,
    energy_fn=double_well_energy,
    title="Sampling Trajectories on Double Well Energy",
    device=device,
    save_path="../docs/assets/images/e_functions/double_well_trajectories.png",
)
plt.close(fig)

# Rastrigin sampling
sampler = LangevinDynamics(
    energy_function=rastrigin_energy, step_size=0.005, noise_scale=0.5
)

# Initial states far from origin
initial_states = torch.tensor(
    [[4.0, 4.0], [-4.0, -4.0], [4.0, -4.0], [-4.0, 4.0]], dtype=torch.float32
)

# Run sampling with trajectory tracking
samples, trajectories = sampler.sample_chain(
    x=initial_states, n_steps=400, return_trajectory=True
)

# Plot trajectories
fig = plot_sample_trajectories(
    trajectories=trajectories,
    energy_fn=rastrigin_energy,
    title="Sampling Trajectories on Rastrigin Energy",
    x_range=(-5.12, 5.12),
    y_range=(-5.12, 5.12),
    device=device,
    save_path="../docs/assets/images/e_functions/rastrigin_trajectories.png",
)
plt.close(fig)

print("All visualizations generated successfully.")
