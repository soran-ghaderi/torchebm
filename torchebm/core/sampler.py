from abc import ABC, abstractmethod
import torch

from torchebm.core.energy_function import EnergyFunction


class Sampler(ABC):
    @abstractmethod
    def sample(self, energy_function: EnergyFunction, initial_state: torch.Tensor, num_steps: int) -> torch.Tensor:
        pass


import torch
from abc import ABC, abstractmethod
from typing import Optional, Union
from torchebm.core.energy_function import EnergyFunction


class Sampler(ABC):
    def __init__(self, energy_function: EnergyFunction):
        self.energy_function = energy_function

    @abstractmethod
    def sample(self, initial_state: torch.Tensor, n_steps: int) -> torch.Tensor:
        pass


class LangevinDynamics(Sampler):
    def __init__(self, energy_function: EnergyFunction, step_size: float, noise_scale: float):
        """
        Initialize the Langevin dynamics sampler.

        Args:
            energy_function (EnergyFunction): The energy function to sample from.
            step_size (float): The step size for the Langevin dynamics updates.
            noise_scale (float): The scale of the noise added in each step.

        Raises:
            ValueError: If step_size or noise_scale is not positive.
        """
        super().__init__(energy_function)
        if step_size <= 0 or noise_scale <= 0:
            raise ValueError("step_size and noise_scale must be positive")
        self.step_size = step_size
        self.noise_scale = noise_scale

    def sample(self, initial_state: torch.Tensor, n_steps: int, return_trajectory: bool = False) -> torch.Tensor:
        """
        Run Langevin dynamics sampling from the given initial state.

        Args:
            initial_state (torch.Tensor): The initial state of the chain.
            n_steps (int): The number of steps to run the dynamics.
            return_trajectory (bool): If True, return the entire trajectory.

        Returns:
            torch.Tensor: The final state or the entire trajectory if return_trajectory is True.
        """
        current_state = initial_state
        trajectory = [current_state] if return_trajectory else None

        for _ in range(n_steps):
            gradient = self.energy_function.gradient(current_state)
            noise = torch.randn_like(current_state) * self.noise_scale
            current_state = current_state - self.step_size * gradient + noise

            if return_trajectory:
                trajectory.append(current_state)

        if return_trajectory:
            return torch.stack(trajectory)
        else:
            return current_state

    def sample_chain(self, initial_state: torch.Tensor, n_steps: int, n_samples: int) -> torch.Tensor:
        """
        Generate multiple samples using Langevin dynamics.

        Args:
            initial_state (torch.Tensor): The initial state of the chain.
            n_steps (int): The number of steps between each sample.
            n_samples (int): The number of samples to generate.

        Returns:
            torch.Tensor: A tensor containing the generated samples.
        """
        samples = []
        current_state = initial_state

        for _ in range(n_samples):
            current_state = self.sample(current_state, n_steps)
            samples.append(current_state)

        return torch.stack(samples)

    @torch.no_grad()
    def sample_parallel(self, initial_states: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Run parallel Langevin dynamics sampling from the given initial states.

        Args:
            initial_states (torch.Tensor): The initial states of the chains.
            n_steps (int): The number of steps to run the dynamics.

        Returns:
            torch.Tensor: The final states of all chains.
        """
        current_states = initial_states

        for _ in range(n_steps):
            gradients = self.energy_function.gradient(current_states)
            noise = torch.randn_like(current_states) * self.noise_scale
            current_states = current_states - self.step_size * gradients + noise

        return current_states


import torch
import matplotlib.pyplot as plt
from torchebm.core.energy_function import EnergyFunction
# from torchebm.samplers.sampler import LangevinDynamics

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
