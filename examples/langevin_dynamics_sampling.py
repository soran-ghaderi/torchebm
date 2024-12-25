# import torch
# from torchebm.core.energy_function import EnergyFunction
# from torchebm.samplers.langevin_dynamics import LangevinDynamics
# from matplotlib import pyplot as plt
#
# class QuadraticEnergy(EnergyFunction):
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return 0.5 * torch.sum(x**2, dim=-1)
#
#     def gradient(self, x: torch.Tensor) -> torch.Tensor:
#         return x
#
#
# def plot_samples(samples, energy_function, title):
#     x = torch.linspace(-3, 3, 100)
#     y = torch.linspace(-3, 3, 100)
#     X, Y = torch.meshgrid(x, y)
#     Z = energy_function(torch.stack([X, Y], dim=-1)).numpy()
#
#     plt.figure(figsize=(10, 8))
#     plt.contourf(X.numpy(), Y.numpy(), Z, levels=20, cmap='viridis')
#     plt.colorbar(label='Energy')
#     plt.scatter(samples[:, 0], samples[:, 1], c='red', alpha=0.5)
#     plt.title(title)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.show()
#
#
# def main():
#     try:
#         # Set up the energy function and sampler
#         energy_function = QuadraticEnergy()
#         sampler = LangevinDynamics(energy_function, step_size=0.1, noise_scale=0.1)
#
#         # Generate samples
#         initial_state = torch.tensor([2.0, 2.0])
#         n_steps = 1000
#         n_samples = 500
#
#         samples = sampler.sample_chain(initial_state, n_steps, n_samples)
#
#         # Plot the results
#         plot_samples(samples, energy_function, "Langevin Dynamics Sampling")
#
#         # Visualize a single trajectory
#         trajectory = sampler.sample(initial_state, n_steps, return_trajectory=True)
#         plot_samples(trajectory, energy_function, "Single Langevin Dynamics Trajectory")
#
#         # Demonstrate parallel sampling
#         n_chains = 10
#         initial_states = torch.randn(n_chains, 2) * 2
#         parallel_samples = sampler.sample_parallel(initial_states, n_steps)
#         plot_samples(parallel_samples, energy_function, "Parallel Langevin Dynamics Sampling")
#
#     except ValueError as e:
#         print(f"Error: {e}")
#
#
#
# if __name__ == "__main__":
#     main()


"""
Examples for using the Langevin Dynamics sampler and other MCMC samplers.

This module provides practical examples ranging from basic usage to advanced
features of the sampling framework.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from torch.xpu import device

from torchebm.samplers.langevin_dynamics import LangevinDynamics


# ===================== Basic Usage Examples =====================

def basic_example():
    """
    Basic example demonstrating simple Langevin dynamics sampling from a
    2D Gaussian distribution.
    """

    # Define a simple 2D Gaussian energy function
    class GaussianEnergy:
        def __init__(self, mean: torch.Tensor, cov: torch.Tensor, device=None):
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.mean = mean.to(self.device)
            self.precision = torch.inverse(cov).to(self.device)

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            x = x.to(self.device)
            delta = x - self.mean
            return 0.5 * torch.einsum('...i,...ij,...j->...', delta, self.precision, delta)

        def gradient(self, x: torch.Tensor) -> torch.Tensor:
            x = x.to(self.device)
            return torch.einsum('...ij,...j->...i', self.precision, x - self.mean)

        def to(self, device):
            self.device = device
            self.mean = self.mean.to(device)
            self.precision = self.precision.to(device)
            return self

    # Create energy function for a 2D Gaussian
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean = torch.tensor([1.0, -1.0])
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])
    energy_fn = GaussianEnergy(mean, cov, device=device)

    # Initialize sampler
    sampler = LangevinDynamics(
        energy_function=energy_fn,
        step_size=0.01,
        noise_scale=0.1,
        device=device  # Make sure to pass the same device
    )

    # Generate samples
    initial_state = torch.zeros(2, device=device)
    samples = sampler.sample_chain(
        initial_state=initial_state,
        n_steps=100,  # steps between samples
        n_samples=1000  # number of samples to collect
    )

    # Plot results
    samples = samples.cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1)
    plt.title('Samples from 2D Gaussian using Langevin Dynamics')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.show()


# ===================== Advanced Usage Examples =====================

def advanced_example():
    """
    Advanced example showing:
    1. Custom energy functions
    2. Parallel sampling
    3. Diagnostics and trajectory tracking
    4. Different initialization strategies
    5. Handling multimodal distributions
    """

    # Define a double-well potential
    class DoubleWellEnergy:
        def __init__(self, barrier_height: float = 2.0):
            self.barrier_height = barrier_height

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return self.barrier_height * (x.pow(2) - 1).pow(2)

        def gradient(self, x: torch.Tensor) -> torch.Tensor:
            return 4 * self.barrier_height * x * (x.pow(2) - 1)

        def to(self, device):
            self.device = device
            self.mean = self.mean.to(device)
            self.precision = self.precision.to(device)
            return self

    # Create energy function and sampler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    energy_fn = DoubleWellEnergy(barrier_height=2.0)
    sampler = LangevinDynamics(
        energy_function=energy_fn,
        step_size=0.001,
        noise_scale=0.1,
        decay=0.1,  # Using decay for better stability
        device=device
    )

    # 1. Generate trajectory with diagnostics
    initial_state = torch.tensor([0.0], device=device)
    trajectory, diagnostics = sampler.sample(
        initial_state=initial_state,
        n_steps=1000,
        return_trajectory=True,
        return_diagnostics=True
    )

    # 2. Parallel sampling from multiple initial points
    initial_states = torch.linspace(-2, 2, 10).unsqueeze(1)
    parallel_samples, parallel_diagnostics = sampler.sample_parallel(
        initial_states=initial_states,
        n_steps=1000,
        return_diagnostics=True
    )

    if isinstance(parallel_diagnostics, list) and all(isinstance(diag, dict) for diag in parallel_diagnostics):
        # Extract diagnostics safely
        energies = [diag['energies'] for diag in parallel_diagnostics if 'energies' in diag]
    else:
        energies = None
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot trajectory
    # parallel_samples = parallel_samples.cpu().numpy()
    # parallel_diagnostics = [diag['energies'] for diag in parallel_diagnostics]
    ax1.plot(trajectory.cpu().numpy())
    ax1.set_title('Single Chain Trajectory')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Position')

    # Plot energy over time
    ax2.plot(diagnostics['energies'].cpu().numpy())
    ax2.set_title('Energy Evolution')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Energy')

    # Plot parallel chain results
    parallel_samples_cpu = parallel_samples.cpu().numpy()
    for sample in parallel_samples_cpu:
        ax3.axhline(sample.item(), alpha=0.3, color='blue')
    ax3.set_title('Final States of Parallel Chains')
    ax3.set_ylabel('Position')

    if 'mean_energies' in parallel_diagnostics:
        ax2.plot(parallel_diagnostics['mean_energies'].cpu().numpy(),
                 linestyle='--', label='Parallel Mean Energy')
        ax2.legend()

    plt.tight_layout()
    plt.show()


# ===================== Utility Examples =====================

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sampler = LangevinDynamics(
        energy_function=HarmonicEnergy(),
        step_size=0.01,
        noise_scale=0.1
    ).to(device)

    # Generate samples with thinning
    initial_state = torch.tensor([2.0], device=device)
    samples, diagnostics = sampler.sample_chain(
        initial_state=initial_state,
        n_steps=50,  # steps between samples
        n_samples=1000,  # number of samples
        thin=10,  # keep every 10th sample
        return_diagnostics=True
    )

    # Custom analysis of results
    def analyze_convergence(samples: torch.Tensor, diagnostics: list) -> Tuple[float, float]:
        """Example utility function to analyze convergence."""
        mean = samples.mean().item()
        std = samples.std().item()
        return mean, std

    mean, std = analyze_convergence(samples, diagnostics)
    print(f"Sample Statistics - Mean: {mean:.3f}, Std: {std:.3f}")


if __name__ == "__main__":
    print("Running basic example...")
    basic_example()

    print("\nRunning advanced example...")
    advanced_example()

    print("\nRunning utilities example...")
    sampling_utilities_example()