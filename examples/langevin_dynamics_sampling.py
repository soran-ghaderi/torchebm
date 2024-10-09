import torch
from torchebm.core.energy_function import EnergyFunction
from torchebm.core.sampler import LangevinDynamics
from matplotlib import pyplot as plt

class QuadraticEnergy(EnergyFunction):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(x**2, dim=-1)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return x


def plot_samples(samples, energy_function, title):
    x = torch.linspace(-3, 3, 100)
    y = torch.linspace(-3, 3, 100)
    X, Y = torch.meshgrid(x, y)
    Z = energy_function(torch.stack([X, Y], dim=-1)).numpy()

    plt.figure(figsize=(10, 8))
    plt.contourf(X.numpy(), Y.numpy(), Z, levels=20, cmap='viridis')
    plt.colorbar(label='Energy')
    plt.scatter(samples[:, 0], samples[:, 1], c='red', alpha=0.5)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def main():
    try:
        # Set up the energy function and sampler
        energy_function = QuadraticEnergy()
        sampler = LangevinDynamics(energy_function, step_size=0.1, noise_scale=0.1)

        # Generate samples
        initial_state = torch.tensor([2.0, 2.0])
        n_steps = 1000
        n_samples = 500

        samples = sampler.sample_chain(initial_state, n_steps, n_samples)

        # Plot the results
        plot_samples(samples, energy_function, "Langevin Dynamics Sampling")

        # Visualize a single trajectory
        trajectory = sampler.sample(initial_state, n_steps, return_trajectory=True)
        plot_samples(trajectory, energy_function, "Single Langevin Dynamics Trajectory")

        # Demonstrate parallel sampling
        n_chains = 10
        initial_states = torch.randn(n_chains, 2) * 2
        parallel_samples = sampler.sample_parallel(initial_states, n_steps)
        plot_samples(parallel_samples, energy_function, "Parallel Langevin Dynamics Sampling")

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()