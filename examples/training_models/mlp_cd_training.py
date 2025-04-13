import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from torchebm.core import BaseEnergyFunction
from torchebm.samplers import LangevinDynamics
from torchebm.losses import ContrastiveDivergence


class MLPEnergy(BaseEnergyFunction):
    """A simple MLP to act as the energy function."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Output a single scalar energy value
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure energy is scalar per batch element
        return self.network(x).squeeze(-1)


# --- 2. Generate Target Data (2D Gaussian Mixture) ---
def generate_gaussian_mixture_data(
    n_samples: int, n_components: int = 4, std: float = 0.1
) -> torch.Tensor:
    """Generates data from a 2D Gaussian mixture centered on a circle."""
    centers = []
    angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
    radius = 1.5
    for angle in angles:
        centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    centers = torch.tensor(centers, dtype=torch.float32)

    data = []
    for _ in range(n_samples):
        comp_idx = np.random.randint(0, n_components)
        point = torch.randn(2) * std + centers[comp_idx]
        data.append(point)

    return torch.stack(data)


# --- 3. Visualization Function ---
@torch.no_grad()
def plot_energy_and_samples(
    energy_fn: BaseEnergyFunction,
    real_samples: torch.Tensor,
    sampler: LangevinDynamics,
    epoch: int,
    device: torch.device,
    grid_size: int = 100,
    plot_range: float = 3.0,
    k_sampling: int = 100,  # Number of steps to generate samples for visualization
):
    """Plots the energy surface, real data, and model samples."""
    plt.figure(figsize=(8, 8))

    # Create grid for energy surface plot
    x_coords = torch.linspace(-plot_range, plot_range, grid_size, device=device)
    y_coords = torch.linspace(-plot_range, plot_range, grid_size, device=device)
    xv, yv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    grid = torch.stack([xv.flatten(), yv.flatten()], dim=1)

    # Calculate energy on the grid
    energy_values = energy_fn(grid).cpu().numpy().reshape(grid_size, grid_size)

    # Plot energy surface
    plt.contourf(
        xv.cpu().numpy(),
        yv.cpu().numpy(),
        np.exp(energy_values),
        levels=50,
        cmap="viridis",
    )
    plt.colorbar(label="exp(Energy)")

    # Generate samples from the current model for visualization
    # Start from random noise for visualization samples
    vis_start_noise = torch.randn(
        500, real_samples.shape[1], device=device
    )  # 500 samples, dim=2
    model_samples = (
        sampler.sample_chain(x=vis_start_noise, n_steps=k_sampling).cpu().numpy()
    )

    # Plot real and model samples
    plt.scatter(
        real_samples[:, 0].cpu().numpy(),
        real_samples[:, 1].cpu().numpy(),
        s=10,
        alpha=0.5,
        label="Real Data",
        c="white",
    )
    plt.scatter(
        model_samples[:, 0],
        model_samples[:, 1],
        s=10,
        alpha=0.5,
        label="Model Samples",
        c="red",
    )

    plt.xlim(-plot_range, plot_range)
    plt.ylim(-plot_range, plot_range)
    plt.title(f"Epoch {epoch}")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.savefig(f"ebm_training_epoch_{epoch:04d}.png") # Optional: save figures
    plt.show()
    plt.close()


# --- 4. Training Setup ---
if __name__ == "__main__":
    # Hyperparameters
    N_SAMPLES = 500
    INPUT_DIM = 2
    HIDDEN_DIM = 16
    BATCH_SIZE = 256
    EPOCHS = 200
    LEARNING_RATE = 1e-2
    SAMPLER_STEP_SIZE = 0.1
    SAMPLER_NOISE_SCALE = 0.1  # Adjust carefully with step_size
    CD_K = 10  # Number of steps for CD sampler
    USE_PCD = False  # Set to True to use Persistent CD
    VISUALIZE_EVERY = 10  # How often to generate plots

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    real_data = generate_gaussian_mixture_data(N_SAMPLES)
    dataset = TensorDataset(real_data)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    print(f"Data shape: {real_data.shape}")

    # Model Components
    energy_model = MLPEnergy(INPUT_DIM, HIDDEN_DIM).to(device)
    sampler = LangevinDynamics(
        energy_function=energy_model,
        step_size=SAMPLER_STEP_SIZE,
        noise_scale=SAMPLER_NOISE_SCALE,  # Often related to sqrt(2*step_size)
        device=device,
    )
    loss_fn = ContrastiveDivergence(
        energy_function=energy_model, sampler=sampler, n_steps=CD_K, persistent=USE_PCD
    ).to(device)

    # Optimizer (Optimizes the parameters of the energy function)
    optimizer = optim.Adam(energy_model.parameters(), lr=LEARNING_RATE)

    # --- 5. Training Loop ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for i, (data_batch,) in enumerate(dataloader):
            data_batch = data_batch.to(device)

            # Zero gradients before calculation
            optimizer.zero_grad()

            # Calculate Contrastive Divergence loss
            # The loss_fn.forward() internally calls the sampler and energy_fn
            loss, negative_samples = loss_fn(data_batch)

            # Backpropagate the loss through the energy function parameters
            loss.backward()

            # Update the energy function parameters
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_epoch_loss:.4f}")

        # --- 6. Visualization ---
        if (epoch + 1) % VISUALIZE_EVERY == 0 or epoch == 0:
            print("Generating visualization...")
            plot_energy_and_samples(
                energy_fn=energy_model,
                real_samples=real_data,  # Plot all real data for context
                sampler=sampler,
                epoch=epoch + 1,
                device=device,
                k_sampling=200,  # Use more steps for better visualization samples
            )

    print("Training finished.")

    # Final visualization
    print("Generating final visualization...")
    plot_energy_and_samples(
        energy_fn=energy_model,
        real_samples=real_data,
        sampler=sampler,
        epoch=EPOCHS,
        device=device,
        k_sampling=500,
    )
