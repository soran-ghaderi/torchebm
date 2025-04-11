"""
Examples for using the Langevin Dynamics sampler.
"""

import time

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from torch.xpu import device

import torch
from torchebm.core import GaussianEnergy, DoubleWellEnergy
from torchebm.samplers.langevin_dynamics import LangevinDynamics

# Define a 10D Gaussian energy function
energy_fn = GaussianEnergy(mean=torch.zeros(10), cov=torch.eye(10))
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Langevin dynamics sampler
langevin_sampler = LangevinDynamics(
    energy_function=energy_fn, step_size=5e-3, device=device
).to(device)

# Sample 10,000 points in 10 dimensions
final_samples = langevin_sampler.sample_chain(
    dim=10, n_steps=500, n_samples=10000, return_trajectory=False
)
print(final_samples.shape)  # Output: (10000, 10) -> (n_samples, dim)

# Sample with trajectory and diagnostics
n_samples = 250
n_steps = 500
dim = 10
samples, diagnostics = langevin_sampler.sample_chain(
    dim=dim,
    n_steps=n_steps,
    n_samples=n_samples,
    return_trajectory=True,
    return_diagnostics=True,
)
print(samples.shape)  # Output: (250, 500, 10) -> (samples, n_steps, dim)
# ===================== Example 1 =====================


def basic_example():
    """
    simple Langevin dynamics sampling from a 2D Gaussian distribution.
    """

    # Create energy function for a 2D Gaussian
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 2  # dimension of the state space
    n_steps = 100  # steps between samples
    n_samples = 1000  # num of samples
    mean = torch.tensor([1.0, -1.0])
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])
    energy_fn = GaussianEnergy(mean, cov, device=device)

    # Initialize sampler
    sampler = LangevinDynamics(
        energy_function=energy_fn,
        step_size=0.01,
        noise_scale=0.1,
        device=device,  # Make sure to pass the same device
    )

    # Generate samples
    initial_state = torch.zeros(n_samples, dim, device=device)
    samples = sampler.sample_chain(
        x=initial_state,
        n_steps=n_steps,  # steps between samples
        n_samples=n_samples,  # number of samples to collect
    )

    # Plot results
    samples = samples.cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1)
    plt.title("Samples from 2D Gaussian using Langevin Dynamics")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.show()


# ===================== Example 2 =====================


def advanced_example():
    """
    Advanced example showing:
    1. Custom energy functions
    2. Parallel sampling
    3. Diagnostics and trajectory tracking
    4. Different initialization strategies
    5. Handling multimodal distributions
    """

    # Create energy function and sampler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    energy_fn = DoubleWellEnergy(barrier_height=2.0)
    sampler = LangevinDynamics(
        energy_function=energy_fn,
        step_size=0.001,
        noise_scale=0.1,
        decay=0.1,  # for stability
        device=device,
    )

    # 1. Generate trajectory with diagnostics
    initial_state = torch.tensor([0.0], device=device)
    trajectory, diagnostics = sampler.sample(
        x=initial_state,
        n_steps=1000,
        return_trajectory=True,
        return_diagnostics=True,
    )

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot trajectory
    # parallel_samples = parallel_samples.cpu().numpy()
    # parallel_diagnostics = [diag['energies'] for diag in parallel_diagnostics]
    # ax1.plot(trajectory.cpu().numpy())
    ax1.plot(trajectory[0, :, 0].cpu().numpy())
    ax1.set_title("Single Chain Trajectory")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Position")

    # Plot energy over time
    # ax2.plot(diagnostics["energies"].cpu().numpy())
    ax2.plot(
        diagnostics[:, 2, 0, 0].cpu().numpy()
    )  # Select the first sample and first dimension

    ax2.set_title("Energy Evolution")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Energy")

    plt.tight_layout()
    plt.show()


# ===================== Examples =====================


def sampling_utilities_example():
    """
    Example demonstrating various utility features:
    1. Chain thinning
    2. Device management
    3. Custom diagnostics
    4. Convergence checking
    """

    # Define simple harmonic oscillator
    class HarmonicEnergy:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return 0.5 * x.pow(2)

        def gradient(self, x: torch.Tensor) -> torch.Tensor:
            return x

    # Initialize sampler with GPU support if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = LangevinDynamics(
        energy_function=HarmonicEnergy(), step_size=0.01, noise_scale=0.1
    ).to(device)

    # Generate samples with thinning
    initial_state = torch.tensor([2.0], device=device)
    samples, diagnostics = sampler.sample_chain(
        x=initial_state,
        n_steps=50,  # steps between samples
        n_samples=1000,  # number of samples
        # thin=10,  # keep every 10th sample -> not supported yet
        return_diagnostics=True,
    )

    # Custom analysis of results
    def analyze_convergence(
        samples: torch.Tensor, diagnostics: list
    ) -> Tuple[float, float]:
        """Example utility function to analyze convergence."""
        mean = samples.mean().item()
        std = samples.std().item()
        return mean, std

    mean, std = analyze_convergence(samples, diagnostics)
    print(f"Sample Statistics - Mean: {mean:.3f}, Std: {std:.3f}")


def langevin_gaussain_sampling():

    energy_fn = GaussianEnergy(mean=torch.zeros(10), cov=torch.eye(10))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize Langevin dynamics model
    langevin_sampler = LangevinDynamics(
        energy_function=energy_fn, step_size=5e-3, device=device
    ).to(device)

    # Initial state: batch of 100 samples, 10-dimensional space
    ts = time.time()
    # Run Langevin sampling for 500 steps
    final_x = langevin_sampler.sample_chain(
        dim=10, n_steps=500, n_samples=10000, return_trajectory=False
    )

    print(final_x.shape)  # Output: (100, 10)  (final state)
    # print(xs.shape)  # Output: (500, 100, 10)  (history of all states)
    print("Time taken: ", time.time() - ts)

    n_samples = 250
    n_steps = 500
    dim = 10
    final_samples, diagnostics = langevin_sampler.sample_chain(
        n_samples=n_samples,
        n_steps=n_steps,
        dim=dim,
        return_trajectory=True,
        return_diagnostics=True,
    )
    print(final_samples.shape)  # Output: (100, 10)  (final state)
    print(diagnostics.shape)  # (500, 3, 100, 10) -> All diagnostics

    x_init = torch.randn(n_samples, dim, dtype=torch.float32, device="cuda")
    samples = langevin_sampler.sample(x=x_init, n_steps=100)
    print(samples.shape)  # Output: (100, 10)  (final state)


# if __name__ == "__main__":
#     print("Running sampling from a Gaussian...")
#     langevin_gaussain_sampling()

# print("Running basic example...")
# basic_example()
#
# print("\nRunning advanced example...")
# advanced_example()
#
# print("\nRunning utilities example...")
# sampling_utilities_example()
