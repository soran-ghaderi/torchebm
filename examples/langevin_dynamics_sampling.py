# """
# Examples for using the Langevin Dynamics sampler.
# """
#
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from typing import Tuple
#
# # from torch.xpu import device
#
# from torchebm.samplers.langevin_dynamics import LangevinDynamics
#
#
# # ===================== Example 1 =====================
#
#
# def basic_example():
#     """
#     simple Langevin dynamics sampling from a 2D Gaussian distribution.
#     """
#
#     # Define a simple 2D Gaussian energy function
#     class GaussianEnergy:
#         def __init__(self, mean: torch.Tensor, cov: torch.Tensor, device=None):
#             self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#             self.mean = mean.to(self.device)
#             self.cov_inv = torch.inverse(cov).to(self.device)
#
#         def __call__(self, x: torch.Tensor) -> torch.Tensor:
#             x = x.to(self.device)
#             delta = x - self.mean
#             return 0.5 * torch.einsum(
#                 "...i,...ij,...j->...", delta, self.cov_inv, delta
#             )
#
#         def gradient(self, x: torch.Tensor) -> torch.Tensor:
#             x = x.to(self.device)
#             return torch.einsum("...ij,...j->...i", self.cov_inv, x - self.mean)
#
#         def to(self, device):
#             self.device = device
#             self.mean = self.mean.to(device)
#             self.cov_inv = self.cov_inv.to(device)
#             return self
#
#     # Create energy function for a 2D Gaussian
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     mean = torch.tensor([1.0, -1.0])
#     cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])
#     energy_fn = GaussianEnergy(mean, cov, device=device)
#
#     # Initialize sampler
#     sampler = LangevinDynamics(
#         energy_function=energy_fn,
#         step_size=0.01,
#         noise_scale=0.1,
#         device=device,  # Make sure to pass the same device
#     )
#
#     # Generate samples
#     initial_state = torch.zeros(2, device=device)
#     samples = sampler.sample_chain(
#         initial_state=initial_state,
#         n_steps=100,  # steps between samples
#         n_samples=1000,  # number of samples to collect
#     )
#
#     # Plot results
#     samples = samples.cpu().numpy()
#     plt.figure(figsize=(10, 5))
#     plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1)
#     plt.title("Samples from 2D Gaussian using Langevin Dynamics")
#     plt.xlabel("x₁")
#     plt.ylabel("x₂")
#     plt.show()
#
#
# # ===================== Example 2 =====================
#
#
# def advanced_example():
#     """
#     Advanced example showing:
#     1. Custom energy functions
#     2. Parallel sampling
#     3. Diagnostics and trajectory tracking
#     4. Different initialization strategies
#     5. Handling multimodal distributions
#     """
#
#     # Define a double-well potential
#     class DoubleWellEnergy:
#         def __init__(self, barrier_height: float = 2.0):
#             self.barrier_height = barrier_height
#
#         def __call__(self, x: torch.Tensor) -> torch.Tensor:
#             return self.barrier_height * (x.pow(2) - 1).pow(2)
#
#         def gradient(self, x: torch.Tensor) -> torch.Tensor:
#             return 4 * self.barrier_height * x * (x.pow(2) - 1)
#
#         def to(self, device):
#             self.device = device
#             return self
#
#     # Create energy function and sampler
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     energy_fn = DoubleWellEnergy(barrier_height=2.0)
#     sampler = LangevinDynamics(
#         energy_function=energy_fn,
#         step_size=0.001,
#         noise_scale=0.1,
#         decay=0.1,  # for stability
#         device=device,
#     )
#
#     # 1. Generate trajectory with diagnostics
#     initial_state = torch.tensor([0.0], device=device)
#     trajectory, diagnostics = sampler.sample(
#         initial_state=initial_state,
#         n_steps=1000,
#         return_trajectory=True,
#         return_diagnostics=True,
#     )
#
#     # 2. Parallel sampling from multiple initial points
#     initial_states = torch.linspace(-2, 2, 10).unsqueeze(1)
#     parallel_samples, parallel_diagnostics = sampler.sample_parallel(
#         initial_states=initial_states, n_steps=1000, return_diagnostics=True
#     )
#
#     if isinstance(parallel_diagnostics, list) and all(
#         isinstance(diag, dict) for diag in parallel_diagnostics
#     ):
#         # Extract diagnostics safely
#         energies = [
#             diag["energies"] for diag in parallel_diagnostics if "energies" in diag
#         ]
#     else:
#         energies = None
#     # Plot results
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
#
#     # Plot trajectory
#     # parallel_samples = parallel_samples.cpu().numpy()
#     # parallel_diagnostics = [diag['energies'] for diag in parallel_diagnostics]
#     ax1.plot(trajectory.cpu().numpy())
#     ax1.set_title("Single Chain Trajectory")
#     ax1.set_xlabel("Step")
#     ax1.set_ylabel("Position")
#
#     # Plot energy over time
#     ax2.plot(diagnostics["energies"].cpu().numpy())
#     ax2.set_title("Energy Evolution")
#     ax2.set_xlabel("Step")
#     ax2.set_ylabel("Energy")
#
#     # Plot parallel chain results
#     parallel_samples_cpu = parallel_samples.cpu().numpy()
#     for sample in parallel_samples_cpu:
#         ax3.axhline(sample.item(), alpha=0.3, color="blue")
#     ax3.set_title("Final States of Parallel Chains")
#     ax3.set_ylabel("Position")
#
#     if "mean_energies" in parallel_diagnostics:
#         ax2.plot(
#             parallel_diagnostics["mean_energies"].cpu().numpy(),
#             linestyle="--",
#             label="Parallel Mean Energy",
#         )
#         ax2.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#
# # ===================== Examples =====================
#
#
# def sampling_utilities_example():
#     """
#     Example demonstrating various utility features:
#     1. Chain thinning
#     2. Device management
#     3. Custom diagnostics
#     4. Convergence checking
#     """
#
#     # Define simple harmonic oscillator
#     class HarmonicEnergy:
#         def __call__(self, x: torch.Tensor) -> torch.Tensor:
#             return 0.5 * x.pow(2)
#
#         def gradient(self, x: torch.Tensor) -> torch.Tensor:
#             return x
#
#     # Initialize sampler with GPU support if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     sampler = LangevinDynamics(
#         energy_function=HarmonicEnergy(), step_size=0.01, noise_scale=0.1
#     ).to(device)
#
#     # Generate samples with thinning
#     initial_state = torch.tensor([2.0], device=device)
#     samples, diagnostics = sampler.sample_chain(
#         initial_state=initial_state,
#         n_steps=50,  # steps between samples
#         n_samples=1000,  # number of samples
#         thin=10,  # keep every 10th sample
#         return_diagnostics=True,
#     )
#
#     # Custom analysis of results
#     def analyze_convergence(
#         samples: torch.Tensor, diagnostics: list
#     ) -> Tuple[float, float]:
#         """Example utility function to analyze convergence."""
#         mean = samples.mean().item()
#         std = samples.std().item()
#         return mean, std
#
#     mean, std = analyze_convergence(samples, diagnostics)
#     print(f"Sample Statistics - Mean: {mean:.3f}, Std: {std:.3f}")
#
#
# if __name__ == "__main__":
#     print("Running basic example...")
#     basic_example()
#
#     print("\nRunning advanced example...")
#     advanced_example()
#
#     print("\nRunning utilities example...")
#     sampling_utilities_example()
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torchebm.samplers.langevin_dynamics import LangevinDynamics


# class TwoModeEnergy:
#     """Simple 2D energy function with two modes."""
#
#     def __init__(self, device=None):
#         super().__init__()
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         # Define the means and covariances of two Gaussian modes
#         self.mu1 = torch.tensor([-2.0, -2.0], device=self.device)
#         self.mu2 = torch.tensor([2.0, 2.0], device=self.device)
#         self.sigma = torch.eye(2, device=self.device)
#
#     def forward(self, x):
#         # Create a double-well potential
#         x = x.to(self.device)
#         z1 = x - self.mu1
#         z2 = x - self.mu2
#         energy1 = 0.5 * torch.sum(torch.matmul(z1, self.sigma) * z1, dim=-1)
#         energy2 = 0.5 * torch.sum(torch.matmul(z2, self.sigma) * z2, dim=-1)
#         return torch.min(energy1, energy2)
#
#     def __call__(self, x, *args, **kwargs):
#         return self.forward(x)
#
#     def gradient(self, x):
#         x = x.to(self.device)
#         # Compute gradient analytically
#         z1 = x - self.mu1
#         z2 = x - self.mu2
#         grad1 = torch.matmul(z1, self.sigma)
#         grad2 = torch.matmul(z2, self.sigma)
#         energy1 = 0.5 * torch.sum(grad1 * z1, dim=-1, keepdim=True)
#         energy2 = 0.5 * torch.sum(grad2 * z2, dim=-1, keepdim=True)
#         mask = (energy1 < energy2).float().unsqueeze(-1)
#         return mask * grad1 + (1 - mask) * grad2
#
#
# def visualize_sampling(device=None):
#     # Create energy function and sampler
#     device = torch.device(
#         "cuda" if (torch.cuda.is_available() and device == None) else "cpu"
#     )
#     energy_fn = TwoModeEnergy(device=device)
#
#     sampler = LangevinDynamics(
#         energy_function=energy_fn,
#         step_size=0.1,
#         noise_scale=0.1,
#         decay=0.1,
#         device=device,
#     )
#
#     # Create grid for energy landscape visualization
#     x = np.linspace(-4, 4, 100)
#     y = np.linspace(-4, 4, 100)
#
#     X, Y = np.meshgrid(x, y)
#
#     points = torch.tensor(
#         np.stack([X.flatten(), Y.flatten()], axis=1).astype(np.float32), device=device
#     )
#
#     # Compute energy landscape
#     with torch.no_grad():
#         energies = energy_fn(points).cpu().numpy().reshape(X.shape)
#
#     # Initialize samples
#     n_samples = 2
#     initial_states = torch.randn(n_samples, 2, device=device) * 3
#
#     # Run sampling
#     n_steps = 20
#     trajectories = []
#     current_states = initial_states.clone()
#
#     for _ in range(n_steps):
#         with torch.no_grad():
#             next_states = sampler.sample(current_states, n_steps=1)
#
#             trajectories.append(current_states.cpu().clone())
#             current_states = next_states
#             print(
#                 "heref", _, len(trajectories), current_states.shape, next_states.shape
#             )
#             # exit()
#
#     # trajectories = torch.stack(trajectories)
#
#     # Plotting
#     fig, ax = plt.subplots(figsize=(10, 10))
#
#     # Plot energy landscape
#     contour = ax.contour(X, Y, energies, levels=20, cmap="viridis")
#     plt.colorbar(contour, label="Energy")
#
#     # Plot initial points
#     ax.scatter(
#         initial_states.cpu()[:, 0],
#         initial_states.cpu()[:, 1],
#         c="red",
#         marker="o",
#         label="Initial states",
#     )
#
#     # Animation update function
#     def update(frame):
#         ax.clear()
#         ax.contour(X, Y, energies, levels=20, cmap="viridis")
#
#         # Plot trajectories up to current frame
#         for i in range(n_samples):
#             trajectory = trajectories[: int(frame + 1), int(i)]
#             ax.plot(trajectory[:, 0], trajectory[:, 1], "r-", alpha=0.5)
#             ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c="red", marker="o", s=50)
#
#         ax.set_xlim(-4, 4)
#         ax.set_ylim(-4, 4)
#         ax.set_title(f"Langevin Dynamics Sampling - Step {frame}")
#
#     # Create animation
#     anim = FuncAnimation(fig, update, frames=n_steps, interval=50, repeat=False)
#
#     plt.show()


# if __name__ == "__main__":
#     visualize_sampling(device="cpu")


#
# import torch
from torchebm.core.energy_function import EnergyFunction

# from torchebm.samplers.langevin_dynamics import LangevinDynamics
# from matplotlib import pyplot as plt


class QuadraticEnergy(EnergyFunction):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(x**2, dim=-1)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return x


def plot_samples(samples, energy_function, title):
    samples = samples.cpu().numpy()
    x = torch.linspace(-3, 3, 100)
    y = torch.linspace(-3, 3, 100)
    X, Y = torch.meshgrid(x, y)
    Z = energy_function(torch.stack([X, Y], dim=-1)).to("cpu").detach().numpy()

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=20, cmap="viridis")
    plt.colorbar(label="Energy")

    plt.scatter(samples[:, 0], samples[:, 1], c="red", alpha=0.5)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def main():
    try:
        # Set up the energy function and sampler
        energy_function = QuadraticEnergy()
        sampler = LangevinDynamics(energy_function, step_size=0.1, noise_scale=0.1)

        # Generate samples
        initial_state = torch.tensor([2.0, 2.0])
        n_steps = 100
        n_samples = 50

        samples = sampler.sample_chain(initial_state, n_steps, n_samples)

        print(f"Samples shape: {samples.shape}")
        # Plot the results
        plot_samples(samples, energy_function, "Langevin Dynamics Sampling")

        # Visualize a single trajectory
        trajectory = sampler.sample(initial_state, n_steps, return_trajectory=True)

        print(f"trajectory shape: {trajectory.shape}")
        plot_samples(trajectory, energy_function, "Single Langevin Dynamics Trajectory")

        # Demonstrate parallel sampling
        n_chains = 10
        initial_states = torch.randn(n_chains, 2) * 2
        parallel_samples = sampler.sample_parallel(initial_states, n_steps)
        print(f"parallel_samples shape: {parallel_samples.shape}")

        plot_samples(
            parallel_samples, energy_function, "Parallel Langevin Dynamics Sampling"
        )

    except ValueError as e:
        print(f"Error: {e}")


# if __name__ == "__main__":
#     main()


# ============================
# benchmarking
# ============================


# class QuadraticEnergy(EnergyFunction):
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return 0.5 * torch.sum(x**2, dim=-1)
#
#     def gradient(self, x: torch.Tensor) -> torch.Tensor:
#         return x
#
#
# def benchmark_langevin(device="cpu", n_chains=1000, n_steps=100, dim=2, n_samples=100):
#     """Benchmark sequential vs parallel sampling"""
#
#     energy = QuadraticEnergy()
#     sampler = LangevinDynamics(energy, device=device)
#
#     initial_state = torch.randn(1, dim) * 2  # For sequential
#     initial_states = torch.randn(n_chains, dim) * 2  # For parallel
#
#     # warm-up runs
#     print("Warming up...")
#     _ = sampler.sample_chain(initial_state.to(device), 10, 10)
#     _ = sampler.sample_parallel(initial_states.to(device), 10)
#     if "cuda" in device:
#         torch.cuda.synchronize()
#
#     # Benchmark sequential sampling
#     print("\nBenchmarking sequential sampling...")
#     start = time.time()
#     seq_samples = sampler.sample_chain(initial_state, n_steps, n_samples)
#     if "cuda" in device:
#         torch.cuda.synchronize()
#     seq_time = time.time() - start
#
#     # Benchmark parallel sampling
#     print("Benchmarking parallel sampling...")
#     start = time.time()
#     par_samples = sampler.sample_parallel(initial_states, n_steps)
#     if "cuda" in device:
#         torch.cuda.synchronize()
#     par_time = time.time() - start
#
#     # Calculate metrics
#     total_seq_steps = n_samples * n_steps
#     total_par_steps = n_steps * n_chains
#     print(f"Total steps: {total_seq_steps} (sequential), {total_par_steps} (parallel)")
#     seq_speed = total_seq_steps / seq_time
#     par_speed = total_par_steps / par_time
#     speedup = par_speed / seq_speed if seq_speed > 0 else float("inf")
#
#     # Validate results
#     seq_mean = seq_samples.mean(dim=0).norm().item()
#     par_mean = par_samples.mean(dim=0).norm().item()
#
#     print(f"\n=== {device.upper()} Results ===")
#     print(f"Sequential: {seq_time:.2f}s ({seq_speed:.1f} steps/s)")
#     print(f"Parallel: {par_time:.2f}s ({par_speed:.1f} steps/s)")
#     print(f"Speedup: {speedup:.1f}x")
#     print(f"Mean validation (should be near 0):")
#     print(f"  Sequential: {seq_mean:.4f}")
#     print(f"  Parallel: {par_mean:.4f}")
#
#     return {
#         "device": device,
#         "seq_time": seq_time,
#         "par_time": par_time,
#         "speedup": speedup,
#         "samples": (seq_samples, par_samples),
#     }
#
#
# def plot_results(metrics):
#     """Visualize benchmark results"""
#     fig, axs = plt.subplots(1, 2, figsize=(12, 5))
#
#     # Scatter plot of samples
#     seq_samples, par_samples = metrics["samples"]
#     seq_samples, par_samples = seq_samples.cpu().numpy(), par_samples.cpu().numpy()
#     axs[0].scatter(seq_samples[:, 0], seq_samples[:, 2], alpha=0.5, label="Sequential")
#     axs[0].scatter(par_samples[:, 0], par_samples[:, 1], alpha=0.5, label="Parallel")
#     axs[0].set_title(f"Sampled Points ({metrics['device'].upper()})")
#     axs[0].legend()
#
#     # Speed comparison
#     devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
#     speedups = [metrics["speedup"]]
#     if torch.cuda.is_available():
#         cuda_metrics = benchmark_langevin(device="cuda")
#         speedups.append(cuda_metrics["speedup"])
#
#     axs[1].bar(devices, speedups)
#     axs[1].set_title("Parallel Sampling Speedup")
#     axs[1].set_ylabel("Speedup Factor (x)")
#
#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     # Run benchmark on available devices
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     metrics = benchmark_langevin(
#         device=device, n_chains=1000, n_steps=100, dim=2, n_samples=100
#     )
#
#     # Visualize results
#     plot_results(metrics)
