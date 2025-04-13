"""
Script to generate visualization images for examples and save them to docs/assets directory.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torchebm.core import GaussianEnergy, DoubleWellEnergy
from torchebm.samplers.langevin_dynamics import LangevinDynamics
from torchebm.samplers.hmc import HamiltonianMonteCarlo

# Create output directory
output_dir = Path("../../docs/assets/images/examples")
output_dir.mkdir(parents=True, exist_ok=True)


# Generate Langevin Dynamics sampling from Gaussian
def generate_langevin_gaussian():
    print("Generating Langevin Dynamics Gaussian sampling visualization...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 2  # dimension of the state space
    n_steps = 100  # steps between samples
    n_samples = 1000  # num of samples
    mean = torch.tensor([1.0, -1.0], device=device)
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]], device=device)
    energy_fn = GaussianEnergy(mean, cov)

    # Initialize sampler
    sampler = LangevinDynamics(
        energy_function=energy_fn,
        step_size=0.01,
        noise_scale=0.1,
    ).to(device)

    # Generate samples
    initial_state = torch.zeros(n_samples, dim, device=device)
    samples = sampler.sample_chain(
        x=initial_state,
        n_steps=n_steps,
    )

    # Plot results
    samples = samples.cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1)
    plt.title("Samples from 2D Gaussian using Langevin Dynamics")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.savefig(output_dir / "langevin_basic.png", dpi=300, bbox_inches="tight")
    print(f"Image saved to {output_dir}/langevin_basic.png")


# Generate HMC sampling from Gaussian
def generate_hmc_gaussian():
    print("Generating HMC Gaussian sampling visualization...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 2  # dimension of the state space
    n_steps = 100  # steps between samples
    n_samples = 1000  # num of samples
    mean = torch.tensor([1.0, -1.0], device=device)
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]], device=device)
    energy_fn = GaussianEnergy(mean, cov)

    # Initialize HMC sampler
    sampler = HamiltonianMonteCarlo(
        energy_function=energy_fn, step_size=0.1, n_leapfrog_steps=5, device=device
    )

    # Generate samples
    initial_state = torch.zeros(n_samples, dim, device=device)
    samples = sampler.sample_chain(x=initial_state, n_steps=n_steps)

    # Plot results
    samples = samples.cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1)
    plt.title("Samples from 2D Gaussian using HMC")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.savefig(output_dir / "hmc_basic.png", dpi=300, bbox_inches="tight")
    print(f"Image saved to {output_dir}/hmc_basic.png")


# Generate Double Well trajectory and energy visualization
def generate_double_well_trajectory():
    print("Generating Double Well trajectory visualization...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    barrier_height = 2.0  # Keep consistent barrier height
    energy_fn = DoubleWellEnergy(barrier_height=barrier_height)

    # Initialize sampler with better parameters for exploration
    sampler = LangevinDynamics(
        energy_function=energy_fn,
        step_size=0.1,  # Larger step size to help cross barriers
        noise_scale=0.3,  # More noise to help escape local minima
        device=device,
    )

    # Start from one of the wells to observe transitions
    initial_state = torch.tensor([-1.5], device=device).view(1, 1)

    # Run for more steps to ensure we observe transitions
    n_steps = 5000
    trajectory, diagnostics = sampler.sample_chain(
        x=initial_state,
        n_steps=n_steps,
        return_trajectory=True,
        return_diagnostics=True,
    )

    # Extract data for plotting
    traj_data = trajectory[0, :, 0].cpu().numpy()
    energy_data = diagnostics[:, 2, 0, 0].cpu().numpy()

    # Plot trajectory and energy evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: trajectory with double well energy function overlay
    ax1.plot(traj_data, label="Position")
    ax1.set_title("Double Well Sampling Trajectory")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Position")
    ax1.grid(True, alpha=0.3)

    # Plot underlying energy function on secondary y-axis
    ax1_twin = ax1.twinx()
    x_range = np.linspace(-2, 2, 1000)
    energy_values = barrier_height * ((x_range**2 - 1) ** 2)
    ax1_twin.plot(
        np.linspace(0, n_steps, 1000),
        energy_values,
        "r--",
        alpha=0.5,
        label="Energy Function",
    )
    ax1_twin.set_ylabel("Energy")
    ax1_twin.spines["right"].set_color("red")
    ax1_twin.tick_params(axis="y", colors="red")
    ax1_twin.yaxis.label.set_color("red")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    # Right plot: energy values during sampling
    ax2.plot(np.arange(len(energy_data)), energy_data, "r-", linewidth=1.0)
    ax2.set_title("Energy Evolution")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Energy")
    ax2.grid(True, alpha=0.3)

    # Highlight transitions between wells
    for i in range(1, len(traj_data)):
        if (traj_data[i - 1] < 0 and traj_data[i] > 0) or (
            traj_data[i - 1] > 0 and traj_data[i] < 0
        ):
            ax1.axvline(x=i, color="g", alpha=0.3, linestyle="--")
            ax2.axvline(x=i, color="g", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_dir / "double_well_trajectory.png", dpi=300, bbox_inches="tight")
    print(f"Image saved to {output_dir}/double_well_trajectory.png")


if __name__ == "__main__":
    # generate_langevin_gaussian()
    generate_hmc_gaussian()
    # generate_double_well_trajectory()
    print("All example visualizations have been generated!")
