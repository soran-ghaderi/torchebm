import numpy as np
import torch
from matplotlib import pyplot as plt

from torchebm.core import GaussianEnergy
from torchebm.samplers.hmc import HamiltonianMonteCarlo


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
