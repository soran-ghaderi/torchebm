import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

from torchebm.core import (
    BaseEnergyFunction,
    CosineScheduler,
    LinearScheduler,
    ExponentialDecayScheduler,
)
from torchebm.samplers import LangevinDynamics
from torchebm.losses import ContrastiveDivergence
from torchebm.datasets import TwoMoonsDataset

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Create output directory for plots
os.makedirs("ebm_training_plots", exist_ok=True)


class MLPEnergy(BaseEnergyFunction):
    """A simple MLP to act as the energy function."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            # nn.Linear(input_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.Tanh(),
            # nn.Linear(hidden_dim, 1),  # Output a single scalar energy value
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


@torch.no_grad()
def plot_energy_and_samples(
    energy_fn: BaseEnergyFunction,
    real_samples: torch.Tensor,  # Expects the full data tensor
    sampler: LangevinDynamics,
    epoch: int,
    device: torch.device,
    grid_size: int = 100,
    plot_range: float = 3.0,
    k_sampling: int = 100,
):
    """Plots the energy surface, real data, and model samples."""
    plt.figure(figsize=(8, 8))

    # Create grid for energy surface plot
    x_coords = torch.linspace(-plot_range, plot_range, grid_size, device=device)
    y_coords = torch.linspace(-plot_range, plot_range, grid_size, device=device)
    xv, yv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    grid = torch.stack([xv.flatten(), yv.flatten()], dim=1)

    # Calculate energy on the grid
    # Ensure energy_fn is in eval mode if it has dropout/batchnorm, although not strictly needed for this MLP
    energy_fn.eval()
    energy_values = energy_fn(grid).cpu().numpy().reshape(grid_size, grid_size)
    energy_fn.train()  # Set back to train mode after plotting

    # Plot energy surface (using probability density for better visualization)
    # Subtract max for numerical stability before exponentiating
    log_prob_values = -energy_values
    log_prob_values = log_prob_values - np.max(log_prob_values)
    prob_density = np.exp(log_prob_values)

    plt.contourf(
        xv.cpu().numpy(),
        yv.cpu().numpy(),
        prob_density,  # Plot probability density
        levels=50,
        cmap="viridis",
    )
    plt.colorbar(label="exp(-Energy) (unnormalized density)")

    # Generate samples from the current model for visualization
    # Start from random noise for visualization samples
    vis_start_noise = torch.randn(
        500, real_samples.shape[1], device=device  # 500 samples, dim matches real data
    )
    model_samples_tensor = sampler.sample(x=vis_start_noise, n_steps=k_sampling)
    model_samples = model_samples_tensor.cpu().numpy()

    # Plot real and model samples
    real_samples_np = (
        real_samples.cpu().numpy()
    )  # Ensure real samples are on CPU for plotting
    plt.scatter(
        real_samples_np[:, 0],
        real_samples_np[:, 1],
        s=10,
        alpha=0.5,
        label="Real Data",
        c="white",
        edgecolors="k",  # Add edge colors for better visibility
        linewidths=0.5,
    )
    plt.scatter(
        model_samples[:, 0],
        model_samples[:, 1],
        s=10,
        alpha=0.5,
        label="Model Samples",
        c="red",
        edgecolors="darkred",
        linewidths=0.5,
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


if __name__ == "__main__":
    # Hyperparameters
    N_SAMPLES = 500
    INPUT_DIM = 2
    HIDDEN_DIM = 16
    BATCH_SIZE = 256
    EPOCHS = 200
    LEARNING_RATE = 1e-3
    SAMPLER_STEP_SIZE = 0.1
    # SAMPLER_STEP_SIZE = ExponentialDecayScheduler(
    #     start_value=1e-2, decay_rate=0.99, min_value=5e-3
    # )
    SAMPLER_STEP_SIZE = CosineScheduler(start_value=3e-2, end_value=5e-3, n_steps=100)

    # SAMPLER_NOISE_SCALE = torch.sqrt(torch.Tensor([SAMPLER_STEP_SIZE])).numpy()[0]
    SAMPLER_NOISE_SCALE = 0.1
    # SAMPLER_NOISE_SCALE = LinearScheduler(start_value=1.0, end_value=0.01, n_steps=50)
    # SAMPLER_NOISE_SCALE = ExponentialDecayScheduler(
    #     start_value=1e-1, decay_rate=0.99, min_value=1e-2
    # )
    SAMPLER_NOISE_SCALE = CosineScheduler(start_value=3e-1, end_value=1e-2, n_steps=100)

    print(f"Sampler noise scale: {SAMPLER_NOISE_SCALE}")
    CD_K = 10
    USE_PCD = True
    VISUALIZE_EVERY = 20
    SEED = 42

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading using Dataset Class
    # Instantiate the dataset object directly
    # It handles generation and device placement internally
    # dataset = GaussianMixtureDataset(
    #     n_samples=N_SAMPLES,
    #     n_components=4,  # Specific parameters for this dataset
    #     std=0.1,
    #     radius=1.5,
    #     device=device,  # Tell dataset where to place the data
    #     seed=SEED,  # Pass the seed
    # )
    dataset = TwoMoonsDataset(n_samples=3000, noise=0.05, seed=SEED, device=device)

    # Get the full tensor ONLY for visualization purposes
    # The DataLoader will use the 'dataset' object directly
    real_data_for_plotting = dataset.get_data()
    print(f"Data shape: {real_data_for_plotting.shape}")

    # Create DataLoader using the Dataset instance directly
    dataloader = DataLoader(
        dataset,  # Use the GaussianMixtureDataset object
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,  # Good practice if batch sizes vary slightly
    )
    # -----------------------------------------

    # Model Components
    energy_model = MLPEnergy(INPUT_DIM, HIDDEN_DIM).to(device)
    sampler = LangevinDynamics(
        energy_function=energy_model,
        step_size=SAMPLER_STEP_SIZE,
        noise_scale=SAMPLER_NOISE_SCALE,
        device=device,
    )
    loss_fn = ContrastiveDivergence(
        energy_function=energy_model,
        sampler=sampler,
        k_steps=CD_K,
        persistent=USE_PCD,
        buffer_size=BATCH_SIZE,
    ).to(
        device
    )  # Loss function itself can be on device

    # Optimizer (Optimizes the parameters of the energy function)
    optimizer = optim.Adam(energy_model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(EPOCHS):
        energy_model.train()  # Ensure model is in training mode
        epoch_loss = 0.0
        for i, data_batch in enumerate(dataloader):
            # data_batch should already be on the correct device because
            # the 'dataset' object was created with device=device.
            # The .to(device) call below is slightly redundant but safe.
            # data_batch = data_batch.to(device)

            # Zero gradients before calculation
            optimizer.zero_grad()

            # Calculate Contrastive Divergence loss
            # The loss_fn.forward() internally calls the sampler and energy_fn
            loss, negative_samples = loss_fn(data_batch)

            # Backpropagate the loss through the energy function parameters
            loss.backward()

            # Optional: Gradient clipping can help stabilize training
            torch.nn.utils.clip_grad_norm_(energy_model.parameters(), max_norm=1.0)

            # Update the energy function parameters
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_epoch_loss:.4f}")

        if (epoch + 1) % VISUALIZE_EVERY == 0 or epoch == 0:
            print("Generating visualization...")
            energy_model.eval()  # Set model to evaluation mode for visualization
            plot_energy_and_samples(
                energy_fn=energy_model,
                real_samples=real_data_for_plotting,  # Use the full dataset tensor
                sampler=sampler,
                epoch=epoch + 1,
                device=device,
                plot_range=2.5,  # Adjusted plot range based on radius=1.5 + std
                k_sampling=200,  # Use more steps for better visualization samples
            )
            # No need to set back to train mode here, it's done at the start of the next epoch loop

    print("Training finished.")

    # Final visualization
    print("Generating final visualization...")
    energy_model.eval()
    plot_energy_and_samples(
        energy_fn=energy_model,
        real_samples=real_data_for_plotting,
        sampler=sampler,
        epoch=EPOCHS,
        device=device,
        plot_range=2.5,
        k_sampling=500,
    )
