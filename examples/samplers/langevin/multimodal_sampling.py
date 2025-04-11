import torch
import numpy as np
import matplotlib.pyplot as plt
from torchebm.samplers.langevin_dynamics import LangevinDynamics
from pathlib import Path


class MultimodalEnergy:
    """
    A 2D energy function with multiple local minima to demonstrate sampling behavior.
    """

    def __init__(self, device=None, dtype=torch.float32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Define centers and weights for multiple Gaussian components
        self.centers = torch.tensor(
            [[-1.0, -1.0], [1.0, 1.0], [-0.5, 1.0], [1.0, -0.5]],
            device=self.device,
            dtype=self.dtype,
        )

        self.weights = torch.tensor(
            [1.0, 0.8, 0.6, 0.7], device=self.device, dtype=self.dtype
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has correct dtype and shape
        x = x.to(dtype=self.dtype)
        if x.dim() == 1:
            x = x.view(1, -1)

        # Calculate distance to each center
        dists = torch.cdist(x, self.centers)

        # Calculate energy as negative log of mixture of Gaussians
        energy = -torch.log(
            torch.sum(self.weights * torch.exp(-0.5 * dists.pow(2)), dim=-1)
        )

        return energy

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has correct dtype and shape
        x = x.to(dtype=self.dtype)
        if x.dim() == 1:
            x = x.view(1, -1)

        # Calculate distances and Gaussian components
        diff = x.unsqueeze(1) - self.centers
        exp_terms = torch.exp(-0.5 * torch.sum(diff.pow(2), dim=-1))
        weights_exp = self.weights * exp_terms

        # Calculate gradient
        normalizer = torch.sum(weights_exp, dim=-1, keepdim=True)
        gradient = torch.sum(
            weights_exp.unsqueeze(-1) * diff / normalizer.unsqueeze(-1), dim=1
        )

        return gradient

    def to(self, device):
        self.device = device
        self.centers = self.centers.to(device)
        self.weights = self.weights.to(device)
        return self


def visualize_energy_landscape_and_sampling():
    # Set up device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Create energy function
    energy_fn = MultimodalEnergy(device=device, dtype=dtype)

    # Initialize the standard Langevin dynamics sampler from the library
    sampler = LangevinDynamics(
        energy_function=energy_fn, step_size=0.01, noise_scale=0.1, device=device
    )

    # Create grid for energy landscape visualization
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # Calculate energy values
    grid_points = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1), device=device, dtype=dtype
    )
    energy_values = energy_fn(grid_points).cpu().numpy().reshape(X.shape)

    # Set up sampling parameters
    dim = 2  # 2D energy function
    n_steps = 200

    # Create figure
    plt.figure(figsize=(10, 8))

    # Plot energy landscape with clear contours
    contour = plt.contour(X, Y, energy_values, levels=20, cmap="viridis")
    plt.colorbar(contour, label="Energy")

    # Run multiple independent chains from different starting points
    n_chains = 5

    # Define distinct colors for the chains
    colors = plt.cm.tab10(np.linspace(0, 1, n_chains))

    # Generate seeds for random starting positions to make chains start in different areas
    seeds = [42, 123, 456, 789, 999]

    for i, seed in enumerate(seeds):
        # Set the seed for reproducibility
        torch.manual_seed(seed)

        # Run one chain using the standard API
        trajectory = sampler.sample_chain(
            dim=dim,  # 2D space
            n_samples=1,  # Single chain
            n_steps=n_steps,  # Number of steps
            return_trajectory=True,  # Return full trajectory
        )

        # Extract trajectory data
        traj_np = trajectory.cpu().numpy().squeeze(0)  # Remove n_samples dimension

        # Plot the trajectory
        plt.plot(
            traj_np[:, 0],
            traj_np[:, 1],
            "o-",
            color=colors[i],
            alpha=0.6,
            markersize=3,
            label=f"Chain {i+1}",
        )

        # Mark the start and end points
        plt.plot(traj_np[0, 0], traj_np[0, 1], "o", color=colors[i], markersize=8)
        plt.plot(traj_np[-1, 0], traj_np[-1, 1], "*", color=colors[i], markersize=10)

    # Add labels and title
    plt.title("Energy Landscape and Langevin Dynamics Sampling Trajectories")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Set axis limits to better focus on the interesting region
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    # Save the figure
    output_dir = Path("../docs/assets/images/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "langevin_trajectory.png", dpi=300, bbox_inches="tight")
    print(f"Image saved to {output_dir}/langevin_trajectory.png")


if __name__ == "__main__":
    print("Running energy landscape visualization...")
    visualize_energy_landscape_and_sampling()
