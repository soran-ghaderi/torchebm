"""
Contrastive Divergence Example

This example demonstrates how to use the ContrastiveDivergence loss function to train
an energy-based model on a simple 2D distribution.
"""

from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torchebm.core import BaseEnergyFunction
from torchebm.samplers import LangevinDynamics
from torchebm.losses import ContrastiveDivergence

output_dir = Path("../../docs/assets/images/examples")


# 1. Define a simple energy function using MLP
class MLPEnergy(BaseEnergyFunction):
    """Simple MLP-based energy function for demonstration."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # Sigmoid Linear Unit (SiLU/Swish)
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """Compute energy values for input samples."""
        # Energy function should return scalar energy per sample
        # Don't move the network to the device of x during forward pass
        return self.net(x).squeeze(-1)


# 2. Generate synthetic data (mixture of 2D Gaussians)
def generate_mixture_data(n_samples=1000, centers=None, std=0.1):
    """Generate samples from a mixture of 2D Gaussians."""
    if centers is None:
        # Default: 4 Gaussians in a circle
        radius = 2.0
        centers = [
            [radius * np.cos(angle), radius * np.sin(angle)]
            for angle in [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        ]

    centers = torch.tensor(centers, dtype=torch.float32)
    n_components = len(centers)

    # Randomly pick components and generate samples
    data = []
    for _ in range(n_samples):
        idx = np.random.randint(0, n_components)
        sample = torch.randn(2) * std + centers[idx]
        data.append(sample)

    return torch.stack(data)


# 3. Utility function for visualization
def visualize_model(
    energy_fn, real_samples, model_samples=None, title="Energy Landscape", grid_size=100
):
    """Visualize the energy landscape and samples."""
    # Create a grid for evaluation
    x = torch.linspace(-4, 4, grid_size)
    y = torch.linspace(-4, 4, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Move grid points to the same device as energy_fn
    device = next(energy_fn.parameters()).device
    grid_points = grid_points.to(device)

    # Compute energy on the grid
    with torch.no_grad():
        energies = energy_fn(grid_points).reshape(grid_size, grid_size).cpu()

    # Plot energy as a colormap
    plt.figure(figsize=(10, 8))
    energy_plot = plt.contourf(
        xx.cpu(), yy.cpu(), torch.exp(-energies), 50, cmap="viridis"
    )
    plt.colorbar(energy_plot, label="exp(-Energy)")

    # Plot real samples
    plt.scatter(
        real_samples[:, 0],
        real_samples[:, 1],
        color="white",
        edgecolor="black",
        alpha=0.6,
        label="Real Data",
    )

    # Plot model samples if provided
    if model_samples is not None:
        plt.scatter(
            model_samples[:, 0],
            model_samples[:, 1],
            color="red",
            alpha=0.4,
            label="Model Samples",
        )

    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Ensure output directory exists
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate synthetic data
    real_data = generate_mixture_data(n_samples=1000)
    # Move data to the correct device
    real_data = real_data.to(device)

    print(f"Generated {len(real_data)} samples from mixture of Gaussians")

    # Define model components
    input_dim = 2
    hidden_dim = 64

    # Create the energy function
    energy_fn = MLPEnergy(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    # Create the sampler (Langevin dynamics)
    sampler = LangevinDynamics(
        energy_function=energy_fn,
        step_size=0.1,  # Step size for Langevin updates
        noise_scale=0.01,  # Noise scale for stochastic gradient Langevin dynamics
        device=device,
    )

    # Create the Contrastive Divergence loss
    cd_loss = ContrastiveDivergence(
        energy_function=energy_fn,
        sampler=sampler,
        n_steps=10,  # Run MCMC for 10 steps per iteration
        persistent=True,  # Use persistent chains for better mixing
        device=device,
    )

    # Create optimizer
    optimizer = optim.Adam(energy_fn.parameters(), lr=0.001)

    # Training parameters
    num_epochs = 100
    batch_size = 128

    # Create data loader
    dataset = torch.utils.data.TensorDataset(real_data)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for i, (batch_data,) in enumerate(dataloader):
            batch_data = batch_data.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Compute CD loss and get negative samples
            loss, neg_samples = cd_loss(batch_data)

            # Backpropagate and update parameters
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Print progress
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

        # Visualize intermediate results every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Generate samples from the model for visualization
            with torch.no_grad():
                init_noise = torch.randn(500, input_dim, device=device)
                model_samples = sampler.sample_chain(init_noise, n_steps=100).cpu()

            # Visualize
            plt = visualize_model(
                energy_fn,
                real_data.cpu(),
                model_samples.cpu(),
                title=f"Energy Landscape - Epoch {epoch+1}",
            )
            plt.savefig(output_dir / f"energy_landscape_epoch_{epoch+1}.png")
            plt.close()

    print("Training complete!")

    # Final visualization
    with torch.no_grad():
        init_noise = torch.randn(1000, input_dim, device=device)
        model_samples = sampler.sample_chain(init_noise, n_steps=500).cpu()

    plt = visualize_model(
        energy_fn, real_data.cpu(), model_samples.cpu(), title="Final Energy Landscape"
    )
    plt.savefig(output_dir / "energy_landscape_final.png")
    plt.show()


if __name__ == "__main__":
    # real_data = generate_mixture_data(1000)
    #
    # plt.figure(figsize=(6, 6))
    # plt.scatter(real_data[:, 0], real_data[:, 1], s=10, alpha=0.5)
    # plt.title("Target Distribution: 2D Gaussian Mixture")
    # plt.grid(True, alpha=0.3)
    # plt.savefig("../../docs/assets/images/examples/gaussian_mixture_target.png")
    #
    # plt.show()
    main()
