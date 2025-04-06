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

        # Define centers and weights for multiple Gaussian components with explicit dtype
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
        try:
            dists = torch.cdist(x, self.centers)
        except RuntimeError as e:
            print(
                f"Error in distance calculation. Input shape: {x.shape}, Centers shape: {self.centers.shape}"
            )
            raise e

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
        gradient = -torch.sum(
            weights_exp.unsqueeze(-1) * diff / normalizer.unsqueeze(-1), dim=1
        )

        return gradient.squeeze()  # Ensure consistent output shape

    def to(self, device):
        self.device = device
        self.centers = self.centers.to(device)
        self.weights = self.weights.to(device)
        return self


class ModifiedLangevinDynamics(LangevinDynamics):
    """
    Modified version of LangevinDynamics to ensure consistent tensor shapes
    """

    def sample(
        self, initial_state, n_steps, return_trajectory=False, return_diagnostics=False
    ):
        current_state = initial_state.clone()

        if return_trajectory:
            trajectory = [current_state.view(1, -1)]  # Ensure consistent shape

        diagnostics = {"energies": []} if return_diagnostics else None

        for _ in range(n_steps):
            # Calculate gradient
            grad = self.energy_function.gradient(current_state)

            # Add noise
            noise = torch.randn_like(current_state) * self.noise_scale

            # Update state
            current_state = current_state - self.step_size * grad + noise

            if return_trajectory:
                trajectory.append(current_state.view(1, -1))  # Ensure consistent shape

            if return_diagnostics:
                diagnostics["energies"].append(
                    self.energy_function(current_state).item()
                )

        if return_trajectory:
            result = torch.cat(trajectory, dim=0)  # Use cat instead of stack
        else:
            result = current_state

        if return_diagnostics:
            diagnostics["energies"] = torch.tensor(diagnostics["energies"])

        return result, diagnostics if return_diagnostics else None


def visualize_energy_landscape_and_sampling():
    # Set up device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Create energy function with explicit dtype
    energy_fn = MultimodalEnergy(device=device, dtype=dtype)

    # Create modified sampler
    sampler = ModifiedLangevinDynamics(
        energy_function=energy_fn, step_size=0.01, noise_scale=0.1, device=device
    )

    try:
        # Create grid for energy landscape visualization
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)

        # Calculate energy values with explicit dtype
        grid_points = torch.tensor(
            np.stack([X.flatten(), Y.flatten()], axis=1), device=device, dtype=dtype
        )

        energy_values = energy_fn(grid_points).cpu().numpy().reshape(X.shape)

        # Generate samples with trajectory tracking
        n_chains = 5
        initial_states = torch.randn(n_chains, 2, device=device, dtype=dtype) * 2

        trajectories = []
        for init_state in initial_states:
            trajectory, _ = sampler.sample(
                initial_state=init_state, n_steps=200, return_trajectory=True
            )
            trajectories.append(trajectory.cpu().numpy())

        # Plotting
        plt.figure(figsize=(12, 10))

        # Plot energy landscape
        contour = plt.contour(X, Y, energy_values, levels=20, cmap="viridis")
        plt.colorbar(contour, label="Energy")

        # Plot sampling trajectories
        colors = plt.cm.rainbow(np.linspace(0, 1, n_chains))
        for idx, (trajectory, color) in enumerate(zip(trajectories, colors)):
            plt.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                "o-",
                color=color,
                alpha=0.5,
                markersize=2,
                label=f"Chain {idx+1}",
            )
            plt.plot(trajectory[0, 0], trajectory[0, 1], "o", color=color, markersize=8)
            plt.plot(
                trajectory[-1, 0], trajectory[-1, 1], "*", color=color, markersize=12
            )

        plt.title("Energy Landscape and Langevin Dynamics Sampling Trajectories")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.grid(True)
        plt.legend()
        # Save the figure to the docs assets directory
        output_dir = Path("docs/assets/images/examples")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "langevin_trajectory.png", dpi=300, bbox_inches='tight')
        print(f"Image saved to {output_dir}/langevin_trajectory.png")
        # plt.show()

    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        print(f"Error details: {type(e).__name__}")
        raise


if __name__ == "__main__":
    print("Running energy landscape visualization...")
    visualize_energy_landscape_and_sampling()
