import torch

from torchebm.core.energy_function import EnergyFunction
from torchebm.core.sampler import Sampler


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

