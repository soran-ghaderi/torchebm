from typing import Tuple, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt

from torchebm.core import GaussianEnergy
from torchebm.samplers.mcmc import HamiltonianMonteCarlo


def test_hmc():
    """Test Hamiltonian Monte Carlo sampler."""
    torch.manual_seed(0)
    device = "cpu"
    energy_function = GaussianEnergy(
        mean=torch.zeros(2), cov=torch.eye(2), device=device
    )
    hmc = HamiltonianMonteCarlo(
        energy_function, step_size=0.1, n_leapfrog_steps=10, device=device
    )

    initial_state = torch.randn(10, 2).to(device=hmc.device)
    samples, diagnostics = hmc.sample(
        initial_state, n_steps=100, return_diagnostics=True
    )

    print('diagnostics["energies"]: ', diagnostics["energies"])
    assert samples.shape == (10, 2)
    assert diagnostics["energies"].shape == (100,)
    assert diagnostics["acceptance_rate"] > 0.0
    assert diagnostics["acceptance_rate"] < 1.0

    initial_states = torch.randn(10, 2).to(device=hmc.device)
    samples, diagnostics = hmc.sample_parallel(
        initial_states, n_steps=100, return_diagnostics=True
    )

    print('diagnostics["mean_energies"]: ', diagnostics["mean_energies"])
    assert samples.shape == (10, 2)
    assert diagnostics["mean_energies"].shape == (100,)
    assert diagnostics["acceptance_rates"].shape == (100,)
    assert diagnostics["acceptance_rates"].mean() > 0.0
    assert diagnostics["acceptance_rates"].mean() < 1.0


def visualize_sampling_trajectory(
    n_steps: int = 100,
    step_size: float = 0.1,
    n_leapfrog_steps: int = 10,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize HMC sampling trajectory with diagnostics.

    Args:
        n_steps: Number of sampling steps
        step_size: Step size for HMC
        n_leapfrog_steps: Number of leapfrog steps
        figsize: Figure size for the plot
        save_path: Optional path to save the figure
    """
    # Set style
    sns.set_theme(style="whitegrid")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    # Create energy function
    energy_function = GaussianEnergy(
        mean=torch.zeros(2, device=device),
        cov=torch.eye(2, device=device),
        device=device,
    )

    # Initialize HMC sampler
    hmc = HamiltonianMonteCarlo(
        energy_function=energy_function,
        step_size=step_size,
        n_leapfrog_steps=n_leapfrog_steps,
        device=device,
    )

    # Generate samples
    initial_state = torch.tensor([[-2.0, 0.0]], dtype=torch.float32, device=device)
    samples, diagnostics = hmc.sample(
        initial_state=initial_state, n_steps=n_steps, return_diagnostics=True
    )

    # Move samples to CPU for plotting
    samples = samples.detach().cpu().numpy()

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # plot 1: Sampling Trajectory
    scatter = ax1.scatter(
        samples[:, 0],
        samples[:, 1],
        c=np.arange(len(samples)),
        cmap="viridis",
        s=50,
        alpha=0.6,
    )
    ax1.plot(samples[:, 0], samples[:, 1], "b-", alpha=0.3)
    ax1.scatter(samples[0, 0], samples[0, 1], c="red", s=100, label="Start")
    ax1.scatter(samples[-1, 0], samples[-1, 1], c="green", s=100, label="End")
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.set_title("HMC Sampling Trajectory")
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label="Step")

    # plot 2: Energy Evolution
    energies = diagnostics["energies"].cpu().numpy()
    ax2.plot(energies, "b-", alpha=0.7)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Energy")
    ax2.set_title("Energy Evolution")

    # plot 3: sample Distribution
    sns.kdeplot(
        x=samples[:, 0], y=samples[:, 1], ax=ax3, fill=True, cmap="viridis", levels=10
    )
    ax3.set_xlabel("x₁")
    ax3.set_ylabel("x₂")
    ax3.set_title("sample Distribution")

    # Add acceptance rate as text
    acceptance_rate = diagnostics["acceptance_rate"].item()
    fig.suptitle(
        f"HMC Sampling Analysis\nAcceptance Rate: {acceptance_rate:.2%}", y=1.05
    )

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_hmc_diagnostics(
    samples: torch.Tensor,
    diagnostics: dict,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> None:
    """
    plot detailed diagnostics for HMC sampling.

    Args:
        samples: Tensor of samples
        diagnostics: Dictionary containing diagnostics
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Set style
    sns.set_theme(style="whitegrid")

    # Move data to CPU for plotting
    samples = samples.detach().cpu().numpy()
    energies = diagnostics["energies"].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # plot 1: Energy Trace
    axes[0].plot(energies, "b-", alpha=0.7)
    axes[0].set_title("Energy Trace")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Energy")

    # plot 2: Energy Distribution
    sns.histplot(energies, kde=True, ax=axes[1])
    axes[1].set_title("Energy Distribution")
    axes[1].set_xlabel("Energy")

    # plot 3: sample Autocorrelation
    from statsmodels.tsa.stattools import acf

    max_lag = min(50, len(samples) - 1)
    acf_values = acf(samples[:, 0], nlags=max_lag, fft=True)
    axes[2].plot(range(max_lag + 1), acf_values)
    axes[2].set_title("sample Autocorrelation")
    axes[2].set_xlabel("Lag")
    axes[2].set_ylabel("ACF")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# test_hmc()


import torch
import seaborn as sns

if __name__ == "__main__":
    # visualize_sampling_trajectory(n_steps=100, step_size=0.1, n_leapfrog_steps=10)

    visualize_sampling_trajectory(
        n_steps=200,
        step_size=0.05,
        n_leapfrog_steps=15,
        figsize=(18, 6),
        save_path="hmc_analysis.png",
    )
