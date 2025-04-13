from typing import Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from torchebm.core import GaussianEnergy, DoubleWellEnergy
from torchebm.samplers.hmc import HamiltonianMonteCarlo

import time

output_dir = Path("../../../docs/assets/images/examples")
output_dir.mkdir(parents=True, exist_ok=True)


def hmc_gaussian_sampling():
    energy_fn = GaussianEnergy(mean=torch.zeros(10), cov=torch.eye(10))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize HMC sampler
    hmc_sampler = HamiltonianMonteCarlo(
        energy_function=energy_fn, step_size=0.1, n_leapfrog_steps=10, device=device
    )

    # Sample 10,000 points in 10 dimensions
    ts = time.time()
    final_x = hmc_sampler.sample_chain(
        dim=10, n_steps=500, n_samples=10000, return_trajectory=False
    )
    print(final_x.shape)  # Output: (10000, 10)
    print("Time taken: ", time.time() - ts)

    # Sample with diagnostics and trajectory
    n_samples = 250
    n_steps = 500
    dim = 10
    final_samples, diagnostics = hmc_sampler.sample_chain(
        n_samples=n_samples,
        n_steps=n_steps,
        dim=dim,
        return_trajectory=True,
        return_diagnostics=True,
    )
    print(final_samples.shape)  # (250, 500, 10)
    print(diagnostics.shape)  # (500, 4, 250, 10)
    print(diagnostics[-1, 3].mean())  # Average acceptance rate

    # Sample from a custom initialization
    x_init = torch.randn(n_samples, dim, dtype=torch.float32, device=device)
    samples = hmc_sampler.sample_chain(x=x_init, n_steps=100)
    print(samples.shape)  # (250, 10)


def hmc_standard_gaussian():
    """
    Generate samples from a 2D Gaussian using standard HMC and visualize the results.
    Saves the visualization to ../docs/assets/images/examples/hmc_standard.png
    """
    print("Generating standard HMC Gaussian sampling visualization...")

    # Set up device and random seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    # Create energy function for a 2D Gaussian
    dim = 2  # dimension of the state space
    n_steps = 100  # steps between samples
    n_samples = 1000  # num of samples
    mean = torch.tensor([1.0, -1.0], device=device)
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]], device=device)
    energy_fn = GaussianEnergy(mean, cov)

    # Initialize HMC sampler
    hmc_sampler = HamiltonianMonteCarlo(
        energy_function=energy_fn, step_size=0.1, n_leapfrog_steps=5, device=device
    )

    # Generate samples
    initial_state = torch.zeros(n_samples, dim, device=device)
    samples = hmc_sampler.sample_chain(x=initial_state, n_steps=n_steps)

    # Plot results
    samples = samples.cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1)
    plt.title("Samples from 2D Gaussian using HMC")
    plt.xlabel("x₁")
    plt.ylabel("x₂")

    # Add mean point with a different color
    plt.scatter([mean[0].item()], [mean[1].item()], color="red", s=100, label="Mean")

    # Add ellipse to represent the covariance structure
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    def plot_cov_ellipse(cov, pos, ax=None, n_std=2.0, **kwargs):
        """
        Plot an ellipse representing the covariance matrix on the given axis.
        """
        if ax is None:
            ax = plt.gca()

        # Convert covariance matrix to numpy if it's a torch tensor
        if isinstance(cov, torch.Tensor):
            cov = cov.cpu().numpy()
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().numpy()

        # Compute eigenvalues and eigenvectors
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        # Width and height are "full" widths, not radii
        width, height = 2 * n_std * np.sqrt(vals)

        # Compute angle of rotation
        theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        # Create ellipse
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

        ax.add_patch(ellip)
        return ellip

    # Plot 2-sigma confidence ellipse
    plot_cov_ellipse(
        cov,
        mean,
        n_std=2.0,
        facecolor="none",
        edgecolor="red",
        linestyle="--",
        linewidth=2,
        label="2σ Confidence",
    )

    plt.legend()
    plt.grid(alpha=0.3)

    # Save figure
    plt.savefig(output_dir / "hmc_standard.png", dpi=300, bbox_inches="tight")
    print(f"Image saved to {output_dir}/hmc_standard.png")


def hmc_custom_mass_matrix():
    """
    Generate samples from a 2D Gaussian using HMC with a custom mass matrix
    and visualize the results. Saves the visualization to
    ../docs/assets/images/examples/hmc_custom_mass.png
    """
    print("Generating HMC with custom mass matrix sampling visualization...")

    # Set up device and random seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    # Create energy function for a 2D Gaussian
    dim = 2  # dimension of the state space
    n_steps = 100  # steps between samples
    n_samples = 1000  # num of samples
    mean = torch.tensor([1.0, -1.0], device=device)
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]], device=device)
    energy_fn = GaussianEnergy(mean, cov)

    # Create custom mass matrix (diagonal in this case)
    # Using 0.1 for first dimension and 1.0 for second dimension
    mass_matrix = torch.tensor([0.1, 1.0], device=device)

    # Initialize HMC sampler with custom mass matrix
    hmc_sampler = HamiltonianMonteCarlo(
        energy_function=energy_fn,
        step_size=0.1,
        n_leapfrog_steps=10,
        mass=mass_matrix,
        device=device,
    )

    # Generate samples
    initial_state = torch.zeros(n_samples, dim, device=device)
    samples = hmc_sampler.sample_chain(x=initial_state, n_steps=n_steps)

    # Plot results
    samples = samples.cpu().numpy()
    plt.figure(figsize=(10, 5))

    # Create a scatter plot with more interesting colors
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1, c="blue")
    plt.title("Samples from 2D Gaussian using HMC with Custom Mass Matrix")
    plt.xlabel("x₁")
    plt.ylabel("x₂")

    # Add mean point with a different color
    plt.scatter([mean[0].item()], [mean[1].item()], color="red", s=100, label="Mean")

    # Add ellipse to represent the covariance structure
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    def plot_cov_ellipse(cov, pos, ax=None, n_std=2.0, **kwargs):
        """
        Plot an ellipse representing the covariance matrix on the given axis.
        """
        if ax is None:
            ax = plt.gca()

        # Convert covariance matrix to numpy if it's a torch tensor
        if isinstance(cov, torch.Tensor):
            cov = cov.cpu().numpy()
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().numpy()

        # Compute eigenvalues and eigenvectors
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        # Width and height are "full" widths, not radii
        width, height = 2 * n_std * np.sqrt(vals)

        # Compute angle of rotation
        theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        # Create ellipse
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

        ax.add_patch(ellip)
        return ellip

    # Plot 2-sigma confidence ellipse
    plot_cov_ellipse(
        cov,
        mean,
        n_std=2.0,
        facecolor="none",
        edgecolor="red",
        linestyle="--",
        linewidth=2,
        label="2σ Confidence",
    )

    plt.legend()
    plt.grid(alpha=0.3)

    # Add text annotation about the mass matrix
    plt.annotate(
        "Mass Matrix = diag([0.1, 1.0])",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        horizontalalignment="left",
        verticalalignment="top",
    )

    # Save figure
    plt.savefig(output_dir / "hmc_custom_mass.png", dpi=300, bbox_inches="tight")
    print(f"Image saved to {output_dir}/hmc_custom_mass.png")


def compare_hmc_implementations():
    """
    Generate and compare samples from standard HMC and HMC with custom mass matrix.
    """
    print("Generating comparison between HMC implementations...")

    # Set up device and random seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    # Create energy function for a 2D Gaussian
    dim = 2  # dimension of the state space
    n_steps = 100  # steps between samples
    n_samples = 1000  # num of samples
    mean = torch.tensor([1.0, -1.0], device=device)
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]], device=device)
    energy_fn = GaussianEnergy(mean, cov)

    # Standard HMC sampler
    standard_hmc = HamiltonianMonteCarlo(
        energy_function=energy_fn, step_size=0.1, n_leapfrog_steps=5, device=device
    )

    # Custom mass matrix
    mass_matrix = torch.tensor([0.1, 1.0], device=device)

    # HMC with custom mass matrix
    custom_hmc = HamiltonianMonteCarlo(
        energy_function=energy_fn,
        step_size=0.1,
        n_leapfrog_steps=10,
        mass=mass_matrix,
        device=device,
    )

    # Generate samples
    initial_state = torch.zeros(n_samples, dim, device=device)

    standard_samples = standard_hmc.sample_chain(
        x=initial_state.clone(), n_steps=n_steps
    )

    custom_samples = custom_hmc.sample_chain(x=initial_state.clone(), n_steps=n_steps)

    # Convert to numpy for plotting
    standard_samples = standard_samples.cpu().numpy()
    custom_samples = custom_samples.cpu().numpy()

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot standard HMC
    ax1.scatter(standard_samples[:, 0], standard_samples[:, 1], alpha=0.1, c="blue")
    ax1.set_title("Standard HMC")
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.grid(alpha=0.3)

    # Plot custom mass matrix HMC
    ax2.scatter(custom_samples[:, 0], custom_samples[:, 1], alpha=0.1, c="green")
    ax2.set_title("HMC with Custom Mass Matrix")
    ax2.set_xlabel("x₁")
    ax2.set_ylabel("x₂")
    ax2.annotate(
        "Mass Matrix = diag([0.1, 1.0])",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        horizontalalignment="left",
        verticalalignment="top",
    )
    ax2.grid(alpha=0.3)

    # Add mean point and covariance ellipse to both plots
    from matplotlib.patches import Ellipse

    def plot_cov_ellipse(cov, pos, ax=None, n_std=2.0, **kwargs):
        if ax is None:
            ax = plt.gca()

        if isinstance(cov, torch.Tensor):
            cov = cov.cpu().numpy()
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().numpy()

        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        width, height = 2 * n_std * np.sqrt(vals)
        theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

        ax.add_patch(ellip)
        return ellip

    mean_np = mean.cpu().numpy()
    cov_np = cov.cpu().numpy()

    # Add mean and ellipse to first plot
    ax1.scatter([mean_np[0]], [mean_np[1]], color="red", s=100)
    plot_cov_ellipse(
        cov_np,
        mean_np,
        ax=ax1,
        n_std=2.0,
        facecolor="none",
        edgecolor="red",
        linestyle="--",
        linewidth=2,
    )

    # Add mean and ellipse to second plot
    ax2.scatter([mean_np[0]], [mean_np[1]], color="red", s=100)
    plot_cov_ellipse(
        cov_np,
        mean_np,
        ax=ax2,
        n_std=2.0,
        facecolor="none",
        edgecolor="red",
        linestyle="--",
        linewidth=2,
    )

    plt.tight_layout()

    # Save figure
    comparison_path = output_dir / "hmc_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    print(f"Comparison image saved to {comparison_path}")


if __name__ == "__main__":
    print("Running HMC examples and generating visualizations...")
    # hmc_gaussian_sampling()  # Original example
    hmc_standard_gaussian()  # Generate standard HMC visualization
    hmc_custom_mass_matrix()  # Generate custom mass matrix visualization
    compare_hmc_implementations()  # Generate comparison visualization
    print("All visualizations completed!")
