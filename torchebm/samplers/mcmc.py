r"""
Hamiltonian Monte Carlo Sampler Module.

This module provides a robust implementation of the Hamiltonian Monte Carlo (HMC) algorithm, a powerful Markov Chain Monte Carlo (MCMC) technique. By leveraging Hamiltonian dynamics, HMC efficiently explores complex, high-dimensional probability distributions, making it ideal for Bayesian inference and statistical modeling.

!!! success "Key Features"
    - Efficient sampling using Hamiltonian dynamics.
    - Customizable step sizes and leapfrog steps for fine-tuned performance.
    - Diagnostic tools to monitor convergence and sampling quality.

---

## Module Components

Classes:
    HamiltonianMonteCarlo: Implements the Hamiltonian Monte Carlo sampler.

---

## Usage Example

!!! example "Sampling from a Gaussian Distribution"
    ```python
    from torchebm.samplers.mcmc import HamiltonianMonteCarlo
    from torchebm.core.energy_function import GaussianEnergy
    import torch

    # Define a 2D Gaussian energy function
    energy_fn = GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2))

    # Initialize HMC sampler
    hmc = HamiltonianMonteCarlo(energy_fn, step_size=0.1, n_leapfrog_steps=10)

    # Starting points for 10 chains
    initial_state = torch.randn(10, 2)

    # Run sampling
    samples, diagnostics = hmc.sample(initial_state, n_steps=100, return_diagnostics=True)
    print(f"Samples: {samples.shape}")
    print(f"Diagnostics: {diagnostics.keys()}")
    ```

---

## Mathematical Foundations

!!! info "Hamiltonian Dynamics in HMC"
    HMC combines statistical sampling with concepts from classical mechanics. It introduces an auxiliary momentum variable \( p \) and defines a Hamiltonian:

    $$
    H(x, p) = U(x) + K(p)
    $$

    - **Potential Energy**: \( U(x) = -\log \pi(x) \), where \( \pi(x) \) is the target distribution.
    - **Kinetic Energy**: \( K(p) = \frac{1}{2} p^T M^{-1} p \), with \( M \) as the mass matrix (often set to the identity matrix).

    This formulation allows HMC to propose new states by simulating trajectories along the energy landscape.

!!! tip "Why Hamiltonian Dynamics?"

    - **Efficient Exploration**: HMC uses gradient information to propose new states, allowing it to explore the state space more efficiently, especially in high-dimensional and complex distributions.
    - **Reduced Correlation**: By simulating Hamiltonian dynamics, HMC reduces the correlation between successive samples, leading to faster convergence to the target distribution.
    - **High Acceptance Rate**: The use of Hamiltonian dynamics and a Metropolis acceptance step ensures that proposed moves are accepted with high probability, provided the numerical integration is accurate.

### Leapfrog Integration

!!! note "Numerical Simulation of Dynamics"
    HMC approximates Hamiltonian trajectories using the leapfrog integrator, a symplectic method that preserves energy. The steps are:

    1. **Momentum Half-Step**:
        $$
        p_{t + \frac{\epsilon}{2}} = p_t - \frac{\epsilon}{2} \nabla U(x_t)
        $$
    2. **Position Full-Step**:
        $$
        x_{t + 1} = x_t + \epsilon M^{-1} p_{t + \frac{\epsilon}{2}}
        $$
    3. **Momentum Half-Step**:
        $$
        p_{t + 1} = p_{t + \frac{\epsilon}{2}} - \frac{\epsilon}{2} \nabla U(x_{t + 1})
        $$

    Here, \( \epsilon \) is the step size, and the process is repeated for \( L \) leapfrog steps.

### Acceptance Step

!!! note "Metropolis-Hastings Correction"
    After proposing a new state \( (x_{t + 1}, p_{t + 1}) \), HMC applies an acceptance criterion to ensure detailed balance:

    $$
    \alpha = \min \left( 1, \exp \left( H(x_t, p_t) - H(x_{t + 1}, p_{t + 1}) \right) \right)
    $$

    The proposal is accepted with probability \( \alpha \), correcting for numerical errors in the leapfrog integration.

---

## Practical Considerations

!!! warning "Tuning Parameters"
    - **Step Size (\( \epsilon \))**: Too large a step size can lead to unstable trajectories; too small reduces efficiency.
    - **Number of Leapfrog Steps (\( L \))**: Affects the distance traveled per proposal—balance exploration vs. computational cost.
    - **Mass Matrix (\( M \))**: Adjusting \( M \) can improve sampling in distributions with varying scales.

!!! question "How to Diagnose Issues?"
    Use `plot_hmc_diagnostics` to check:

    - Acceptance rates (ideal: 0.6–0.8).

    - Divergences (should be rare).

    - Autocorrelation of samples.

!!! warning "Common Pitfalls"

    - **Low Acceptance Rate**: If the acceptance rate is too low, it may indicate that the step size is too large or the number of leapfrog steps is too high. Try reducing the step size or decreasing the number of leapfrog steps.

    - **High Correlation Between Samples**: If samples are highly correlated, it may indicate that the step size is too small or the number of leapfrog steps is too few. Increase the step size or the number of leapfrog steps to improve exploration.

    - **Divergence or NaN Values**: Numerical instability or poor parameter choices can lead to divergent behavior or NaN values. Ensure that the energy function and its gradients are correctly implemented and that parameters are appropriately scaled.

    Use the diagnostic functions provided to identify and address these issues.

---

## Advanced Insights

!!! abstract "Why HMC Outperforms Other MCMC Methods"
    HMC’s use of gradients and dynamics reduces random-walk behavior, making it particularly effective for:

    - High-dimensional spaces.
    - Multimodal distributions (with proper tuning).
    - Models with strong correlations between variables.

???+ info "Further Reading"
    - Inspired by concepts in [this lecture](https://faculty.washington.edu/yenchic/19A_stat535/Lec9_HMC.pdf).
    - Neal, R. M. (2011). "MCMC using Hamiltonian dynamics." *Handbook of Markov Chain Monte Carlo*.
"""

from typing import Optional, Union, Tuple

import torch

from torchebm.core import BaseSampler
from torchebm.core.energy_function import EnergyFunction


class HamiltonianMonteCarlo(BaseSampler):
    def __init__(
        self,
        energy_function: EnergyFunction,
        step_size: float = 0.1,
        n_leapfrog_steps: int = 10,
        mass: Optional[Tuple[float, torch.Tensor]] = None,
        dtype: torch.Tensor = torch.float32,
        device: Optional[Union[Tuple[str, torch.device]]] = None,
    ):
        """"""
        super().__init__(energy_function=energy_function, dtype=dtype, device=device)
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if n_leapfrog_steps <= 0:
            raise ValueError("n_leapfrog_steps must be positive")

        self.step_size = step_size
        self.n_leapfrog_steps = n_leapfrog_steps
        self.energy_function = energy_function
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.mass = mass  # Mass matrix (can be scalar or diagonal tensor)

    def _initialize_momentum(self, shape: torch.Size) -> torch.Tensor:

        p = torch.randn(shape, dtype=self.dtype, device=self.device)

        if self.mass is not None:

            if isinstance(self.mass, float):
                p = p * torch.sqrt(
                    torch.tensor(self.mass, dtype=self.dtype, device=self.device)
                )

            else:
                mass_sqrt = torch.sqrt(self.mass)
                p = p * mass_sqrt.view(*([1] * (len(shape) - 1)), -1).expand_as(p)
        return p

    def _initialize_kenitic_energy(self, p: torch.Tensor) -> torch.Tensor:

        k = 0.5 * (torch.transpose(p, 1, 2)) * p


# class HamiltonianMonteCarloOld(BaseSampler):
#     r"""Hamiltonian Monte Carlo sampler implementation.
#
#     This method simulates a Hamiltonian Monte Carlo.
#
#     Args:
#         energy_function: The energy function to sample from
#         step_size: The step size for leapfrog updates
#         n_leapfrog_steps: Number of leapfrog steps per sample
#         dtype: Tensor dtype to use
#         device: Device to run on
#         mass_matrix: Optional mass matrix for momentum sampling
#
#     Methods:
#         _compute_log_prob(state): Compute the log-probability of a given state
#         _kinetic_energy(momentum): Compute the kinetic energy of a given momentum
#         _compute_hamiltonian(position, momentum): Compute the Hamiltonian of a given position and momentum
#         _setup_diagnostics(): Initialize the diagnostics dictionary
#         _sample_initial_momentum(batch_size, state_shape): Sample the initial momentum for a given state
#         sample(initial_state, n_steps, return_diagnostics): Generate samples using HMC
#         sample_parallel(initial_states, n_steps, return_diagnostics): Implementation of parallel Hamiltonian Monte Carlo sampling
#
#
#     Mathematical Background:
#         Hamiltonian Monte Carlo (HMC) is a method that uses Hamiltonian dynamics to propose new states in the sampling process. The Hamiltonian is defined as:
#
#         $$
#         H(x, p) = U(x) + K(p)
#         $$
#
#         where \( U(x) \) is the potential energy and \( K(p) \) is the kinetic energy. The potential energy is related to the target distribution \( \pi(x) \) by:
#
#         $$
#         U(x) = -\log \pi(x)
#         $$
#
#         The kinetic energy is typically defined as:
#
#         $$
#         K(p) = \frac{1}{2} p^T M^{-1} p
#         $$
#
#         where \( p \) is the momentum and \( M \) is the mass matrix.
#
#             !!! note "Leapfrog Integration"
#
#                 The leapfrog integration method is used to simulate the Hamiltonian dynamics. It consists of the following steps:
#
#                 1.  Half-step update for momentum:
#
#                     $$
#                     p_{t + 1/2} = p_t - \frac{\epsilon}{2} \nabla U(x_t)
#                     $$
#
#                 2.  Full-step update for position:
#
#                     $$
#                     x_{t + 1} = x_t + \epsilon M^{-1} p_{t + \frac{1}{2}}
#                     $$
#
#                 3.  Another half-step update for momentum:
#
#                     $$
#                     p_{t + 1} = p_{t + \frac{1}{2}} - \frac{\epsilon}{2} \nabla U(x_{t + 1})
#                     $$
#
#         Acceptance Probability:
#             After proposing a new state using the leapfrog integration, the acceptance probability is computed as:
#
#             $$
#             \alpha = \min \left(1, \exp \left( H(x_t, p_t) - H(x_{t + 1}, p_{t + 1}) \right) \right)
#             $$
#
#             The new state is accepted with probability \( \alpha \).
#
#     References:
#         - Implements the HMC based on https://faculty.washington.edu/yenchic/19A_stat535/Lec9_HMC.pdf
#
#     """
#
#     def __init__(
#         self,
#         energy_function: EnergyFunction,
#         step_size: float = 1e-3,
#         n_leapfrog_steps: int = 10,
#         dtype: torch.dtype = torch.float32,
#         device: Optional[Union[str, torch.device]] = None,
#         mass_matrix: Optional[torch.Tensor] = None,
#     ):
#         """Initialize Hamiltonian Monte Carlo sampler."""
#         super().__init__(energy_function, dtype, device)
#         self.step_size = step_size
#         self.n_leapfrog_steps = n_leapfrog_steps
#         self.mass_matrix = (
#             mass_matrix.to(device=self.device) if mass_matrix is not None else None
#         )
#
#     def _compute_log_prob(self, state: torch.Tensor) -> torch.Tensor:
#         """Computes the log-probability for a given state."""
#         state = state.to(device=self.device)
#         if not state.requires_grad:
#             state.requires_grad_(True)
#         return -self.energy_function(state)
#
#     def _kinetic_energy(self, momentum: torch.Tensor) -> torch.Tensor:
#         """Compute kinetic energy."""
#         momentum = momentum.to(device=self.device)
#         if self.mass_matrix is None:
#             return torch.sum(0.5 * momentum**2, dim=tuple(range(1, momentum.dim())))
#
#         mass_matrix_inverse = torch.inverse(self.mass_matrix)
#         return torch.sum(
#             0.5 * torch.matmul(momentum, mass_matrix_inverse) * momentum,
#             dim=tuple(range(1, momentum.dim())),
#         )
#
#     def _compute_hamiltonian(
#         self, position: torch.Tensor, momentum: torch.Tensor
#     ) -> torch.Tensor:
#         """Compute Hamiltonian."""
#         position = position.to(device=self.device)
#         momentum = momentum.to(device=self.device)
#
#         return -self._compute_log_prob(position) + self._kinetic_energy(momentum)
#
#     def _setup_diagnostics(self) -> dict:
#         """Initialize diagnostics dictionary."""
#         return {
#             "energies": torch.empty(0, device=self.device),
#             "acceptance_rate": torch.tensor(0.0, device=self.device),
#         }
#
#     def _sample_initial_momentum(
#         self, batch_size: int, state_shape: tuple
#     ) -> torch.Tensor:
#         """sample initial momentum."""
#         momentum = torch.randn(
#             (batch_size, *state_shape), device=self.device, dtype=self.dtype
#         )
#         if self.mass_matrix is not None:
#             momentum = torch.matmul(self.mass_matrix, momentum)
#         return momentum
#
#     @torch.enable_grad()
#     def sample(
#         self,
#         initial_state: torch.Tensor,
#         n_steps: int,
#         return_diagnostics: bool = False,
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
#         """Generate samples using HMC."""
#         # Ensure initial state is on correct device
#         current_position = (
#             initial_state.to(device=self.device).clone().requires_grad_(True)
#         )
#
#         diagnostics = self._setup_diagnostics() if return_diagnostics else None
#
#         batch_size = current_position.shape[0] if len(current_position.shape) > 1 else 1
#         state_shape = (
#             current_position.shape[1:]
#             if len(current_position.shape) > 1
#             else current_position.shape
#         )
#
#         for step in range(n_steps):
#             # sample initial momentum
#             initial_momentum = self._sample_initial_momentum(batch_size, state_shape)
#
#             # Initialize position
#             position = current_position.clone().requires_grad_(True)
#             momentum = initial_momentum.clone()
#
#             # First half-step momentum update
#             log_prob = self._compute_log_prob(position)
#             grad_log_prob = torch.autograd.grad(
#                 log_prob.sum(), position, create_graph=True, retain_graph=True
#             )[0].to(device=self.device)
#             momentum = momentum - 0.5 * self.step_size * grad_log_prob
#
#             # Leapfrog integration
#             for _ in range(self.n_leapfrog_steps - 1):
#                 position = (position + self.step_size * momentum).requires_grad_(True)
#                 log_prob = self._compute_log_prob(position)
#                 grad_log_prob = torch.autograd.grad(
#                     log_prob.sum(), position, create_graph=True, retain_graph=True
#                 )[0].to(device=self.device)
#                 momentum = momentum - self.step_size * grad_log_prob
#
#             # Last position and momentum updates
#             position = (position + self.step_size * momentum).requires_grad_(True)
#             log_prob = self._compute_log_prob(position)
#             grad_log_prob = torch.autograd.grad(
#                 log_prob.sum(), position, create_graph=True, retain_graph=True
#             )[0].to(device=self.device)
#             momentum = momentum - 0.5 * self.step_size * grad_log_prob
#
#             # Compute acceptance probability
#             initial_hamiltonian = self._compute_hamiltonian(
#                 current_position, initial_momentum
#             )
#             proposed_hamiltonian = self._compute_hamiltonian(position, momentum)
#             energy_diff = proposed_hamiltonian - initial_hamiltonian
#             acceptance_prob = torch.exp(-energy_diff)
#
#             # Accept/reject step
#             uniform_rand = torch.rand_like(acceptance_prob, device=self.device)
#             accepted = uniform_rand < torch.minimum(
#                 torch.ones_like(acceptance_prob, device=self.device), acceptance_prob
#             )
#
#             # Update state
#             current_position = torch.where(
#                 accepted.unsqueeze(-1), position.detach(), current_position.detach()
#             ).requires_grad_(True)
#
#             # Update diagnostics
#             if return_diagnostics:
#                 diagnostics["energies"] = torch.cat(
#                     [
#                         diagnostics["energies"],
#                         initial_hamiltonian.detach().mean().unsqueeze(0),
#                     ]
#                 )
#                 diagnostics["acceptance_rate"] = (
#                     diagnostics["acceptance_rate"] * step + accepted.float().mean()
#                 ) / (step + 1)
#
#         if return_diagnostics:
#             return current_position, diagnostics
#         return current_position
#
#     @torch.enable_grad()
#     def sample_parallel(
#         self,
#         initial_states: torch.Tensor,
#         n_steps: int,
#         return_diagnostics: bool = False,
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
#         """Implementation of parallel Hamiltonian Monte Carlo sampling."""
#         # Ensure initial states are on correct device
#         current_states = initial_states.to(
#             device=self.device, dtype=self.dtype
#         ).requires_grad_(True)
#
#         diagnostics = (
#             {
#                 "mean_energies": torch.empty(0, device=self.device),
#                 "acceptance_rates": torch.empty(0, device=self.device),
#             }
#             if return_diagnostics
#             else None
#         )
#
#         batch_size = current_states.shape[0]
#         state_shape = current_states.shape[1:]
#
#         for step in range(n_steps):
#             # sample initial momentum (already on correct device from _sample_initial_momentum)
#             momenta = self._sample_initial_momentum(batch_size, state_shape)
#
#             # Initialize new states and momenta
#             new_states = current_states.clone().requires_grad_(True)
#             new_momenta = momenta.clone()
#
#             # First half-step for momentum
#             log_prob = self._compute_log_prob(new_states)
#             grad_log_prob = torch.autograd.grad(
#                 log_prob.sum(), new_states, create_graph=True, retain_graph=True
#             )[0].to(device=self.device)
#             new_momenta = new_momenta - 0.5 * self.step_size * grad_log_prob
#
#             # Leapfrog steps
#             for _ in range(self.n_leapfrog_steps - 1):
#                 # Full step for position
#                 new_states = (new_states + self.step_size * new_momenta).requires_grad_(
#                     True
#                 )
#
#                 # Full step for momentum
#                 log_prob = self._compute_log_prob(new_states)
#                 grad_log_prob = torch.autograd.grad(
#                     log_prob.sum(), new_states, create_graph=True, retain_graph=True
#                 )[0].to(device=self.device)
#                 new_momenta = new_momenta - self.step_size * grad_log_prob
#
#             # Last position update
#             new_states = (new_states + self.step_size * new_momenta).requires_grad_(
#                 True
#             )
#
#             # Final half-step for momentum
#             log_prob = self._compute_log_prob(new_states)
#             grad_log_prob = torch.autograd.grad(
#                 log_prob.sum(), new_states, create_graph=True, retain_graph=True
#             )[0].to(device=self.device)
#             new_momenta = new_momenta - 0.5 * self.step_size * grad_log_prob
#
#             # Compute Hamiltonians (both tensors will be on correct device from _compute_hamiltonian)
#             initial_hamiltonian = self._compute_hamiltonian(current_states, momenta)
#             proposed_hamiltonian = self._compute_hamiltonian(new_states, new_momenta)
#
#             # Metropolis acceptance step
#             energy_diff = proposed_hamiltonian - initial_hamiltonian
#             acceptance_prob = torch.minimum(
#                 torch.ones_like(energy_diff, device=self.device),
#                 torch.exp(-energy_diff),
#             )
#
#             # Accept/reject step
#             accept = (
#                 torch.rand_like(acceptance_prob, device=self.device) < acceptance_prob
#             )
#             current_states = torch.where(
#                 accept.unsqueeze(-1), new_states.detach(), current_states.detach()
#             ).requires_grad_(True)
#
#             if return_diagnostics:
#                 diagnostics["mean_energies"] = torch.cat(
#                     [
#                         diagnostics["mean_energies"],
#                         initial_hamiltonian.mean().unsqueeze(0),
#                     ]
#                 )
#                 diagnostics["acceptance_rates"] = torch.cat(
#                     [
#                         diagnostics["acceptance_rates"],
#                         accept.float().mean().unsqueeze(0),
#                     ]
#                 )
#
#         if return_diagnostics:
#             return current_states, diagnostics
#         return current_states


# def visualizing_sampling_trajectory():
#     import matplotlib.pyplot as plt
#
#     device = "cpu"
#     torch.manual_seed(0)
#     energy_function = GaussianEnergy(
#         mean=torch.zeros(2), cov=torch.eye(2), device=device
#     )
#     hmc = HamiltonianMonteCarlo(
#         energy_function, step_size=0.1, n_leapfrog_steps=10, device=device
#     )
#
#     initial_state = torch.tensor([[-2.0, 0.0]], dtype=torch.float32)
#     samples, diagnostics = hmc.sample(
#         initial_state, n_steps=100, return_diagnostics=True
#     )
#
#     plt.figure(figsize=(6, 6))
#     plt.plot(samples[:, 0].numpy(), samples[:, 1].numpy(), marker="o", color="blue")
#     plt.xlabel("x1")
#     plt.ylabel("x2")
#     plt.title("HMC Sampling Trajectory")
#     plt.show()

# class HamiltonianMonteCarlo1(BaseSampler):
#     """Hamiltonian Monte Carlo sampler implementation.
#
#     Args:
#         energy_function (EnergyFunction): Energy function to sample from.
#         step_size (float): Step size for leapfrog updates.
#         n_leapfrog_steps (int): Number of leapfrog steps per sample.
#         dtype (torch.dtype): Data type to use for the computations.
#         device (str): Device to run the computations on (e.g., "cpu" or "cuda").
#         mass_matrix (torch.Tensor): Mass matrix for momentum sampling.
#
#     Methods:
#         _compute_log_prob(state): Compute the log-probability of a given state.
#         _kinetic_energy(momentum): Compute the kinetic energy of a given momentum.
#         _compute_hamiltonian(position, momentum): Compute the Hamiltonian of a given position and momentum.
#         _setup_diagnostics(): Initialize the diagnostics dictionary.
#         _sample_initial_momentum(batch_size, state_shape): Sample the initial momentum for a given state.
#         sample(initial_state, n_steps, return_diagnostics): Generate samples using HMC.
#         sample_parallel(initial_states, n_steps, return_diagnostics): Implementation of parallel Hamiltonian Monte Carlo sampling.
#
#     References:
#         - Implements the HMC based on https://faculty.washington.edu/yenchic/19A_stat535/Lec9_HMC.pdf.
#     """
#
#     def __init__(
#         self,
#         energy_function: EnergyFunction,
#         step_size: float = 1e-3,
#         n_leapfrog_steps: int = 10,
#         dtype: torch.dtype = torch.float32,
#         device: Optional[Union[str, torch.device]] = None,
#         mass_matrix: Optional[torch.Tensor] = None,
#     ):
#         """Initialize Hamiltonian Monte Carlo sampler.
#
#         Args:
#             energy_function: The energy function to sample from
#             step_size: The step size for leapfrog updates
#             n_leapfrog_steps: Number of leapfrog steps per sample
#             dtype: Tensor dtype to use
#             device: Device to run on
#             mass_matrix: Optional mass matrix for momentum sampling
#         """
#         super().__init__(energy_function, dtype, device)
#         self.step_size = step_size
#         self.n_leapfrog_steps = n_leapfrog_steps
#
#         # Ensure mass matrix is on correct device and dtype
#         if mass_matrix is not None:
#             self.mass_matrix = mass_matrix.to(device=self.device, dtype=self.dtype)
#         else:
#             self.mass_matrix = None
#
#     def _compute_log_prob(self, state: torch.Tensor) -> torch.Tensor:
#         """Computes the log-probability (up to a constant) for a given state (position)."""
#         state = state.to(device=self.device, dtype=self.dtype)
#         if not state.requires_grad:
#             state.requires_grad_(True)
#         return -self.energy_function(state)
#
#     def _sample_initial_momentum(
#         self, batch_size: int, state_shape: tuple
#     ) -> torch.Tensor:
#         """Samples the initial momentum for a given state (position): ω0 ~ N(0, M^(-1))."""
#         momentum = torch.randn(
#             (batch_size, *state_shape), device=self.device, dtype=self.dtype
#         )
#
#         if self.mass_matrix is not None:
#             # Ensure mass matrix is properly shaped and on correct device
#             mass_matrix = self.mass_matrix.to(device=self.device, dtype=self.dtype)
#             momentum = torch.matmul(mass_matrix, momentum)
#
#         return momentum
#
#     def _kinetic_energy(self, momentum: torch.Tensor) -> torch.Tensor:
#         """Compute kinetic energy K(ω) = 1/2 ω^T M^(-1) ω."""
#         momentum = momentum.to(device=self.device, dtype=self.dtype)
#
#         if self.mass_matrix is None:
#             return torch.sum(0.5 * momentum**2, dim=tuple(range(1, momentum.dim())))
#
#         # Ensure mass matrix is on correct device and properly shaped
#         mass_matrix = self.mass_matrix.to(device=self.device, dtype=self.dtype)
#         mass_matrix_inverse = torch.inverse(mass_matrix)
#
#         # Compute kinetic energy with mass matrix
#         return torch.sum(
#             0.5 * torch.matmul(momentum, mass_matrix_inverse) * momentum,
#             dim=tuple(range(1, momentum.dim())),
#         )
#
#     def _compute_hamiltonian(
#         self, position: torch.Tensor, momentum: torch.Tensor
#     ) -> torch.Tensor:
#         """Compute H(x,ω) = -log r(x) + K(ω)."""
#         # Ensure both tensors are on the same device
#         position = position.to(device=self.device, dtype=self.dtype)
#         momentum = momentum.to(device=self.device, dtype=self.dtype)
#
#         return -self._compute_log_prob(position) + self._kinetic_energy(momentum)
#
#     @torch.enable_grad()
#     def sample(
#         self,
#         initial_state: torch.Tensor,
#         n_steps: int,
#         return_diagnostics: bool = False,
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
#         """Generate samples using HMC following the specified steps."""
#         # Ensure initial state is on correct device and requires gradients
#         current_position = (
#             initial_state.to(device=self.device, dtype=self.dtype)
#             .clone()
#             .requires_grad_(True)
#         )
#
#         # Initialize diagnostics if needed
#         diagnostics = self._setup_diagnostics() if return_diagnostics else None
#
#         # Handle batch dimension properly
#         batch_size = current_position.shape[0] if len(current_position.shape) > 1 else 1
#         state_shape = (
#             current_position.shape[1:]
#             if len(current_position.shape) > 1
#             else current_position.shape
#         )
#
#         for step in range(n_steps):
#             # 1. Generate initial momentum (already on correct device)
#             initial_momentum = self._sample_initial_momentum(batch_size, state_shape)
#
#             # 2. Initialize position
#             position = current_position.clone().requires_grad_(True)
#
#             # 3. First half-step momentum update
#             log_prob = self._compute_log_prob(position)
#             grad_log_prob = torch.autograd.grad(
#                 log_prob.sum(), position, create_graph=True, retain_graph=True
#             )[0]
#             momentum = initial_momentum - 0.5 * self.step_size * grad_log_prob
#
#             # 4. Main leapfrog steps
#             for _ in range(self.n_leapfrog_steps - 1):
#                 # Update position
#                 position = (position + self.step_size * momentum).requires_grad_(True)
#
#                 # Update momentum
#                 log_prob = self._compute_log_prob(position)
#                 grad_log_prob = torch.autograd.grad(
#                     log_prob.sum(), position, create_graph=True, retain_graph=True
#                 )[0]
#                 momentum = momentum - self.step_size * grad_log_prob
#
#             # 5. Last position update
#             position = (position + self.step_size * momentum).requires_grad_(True)
#
#             # 6. Last half-step momentum update
#             log_prob = self._compute_log_prob(position)
#             grad_log_prob = torch.autograd.grad(
#                 log_prob.sum(), position, create_graph=True, retain_graph=True
#             )[0]
#             momentum = momentum - 0.5 * self.step_size * grad_log_prob
#
#             # 7. Compute acceptance probability
#             initial_hamiltonian = self._compute_hamiltonian(
#                 current_position, initial_momentum
#             )
#             proposed_hamiltonian = self._compute_hamiltonian(position, momentum)
#             energy_diff = proposed_hamiltonian - initial_hamiltonian
#             acceptance_prob = torch.exp(-energy_diff)
#
#             # 8. Accept/reject step
#             accepted = torch.rand_like(
#                 acceptance_prob, device=self.device
#             ) < torch.minimum(
#                 torch.ones_like(acceptance_prob, device=self.device), acceptance_prob
#             )
#
#             # Update state
#             current_position = torch.where(
#                 accepted.unsqueeze(-1),
#                 position.detach(),  # Important: detach accepted states
#                 current_position.detach(),  # Important: detach rejected states
#             ).requires_grad_(
#                 True
#             )  # Ensure the result requires gradients
#
#             # Update diagnostics if needed
#             if return_diagnostics:
#                 diagnostics["energies"] = torch.cat(
#                     [
#                         diagnostics["energies"],
#                         initial_hamiltonian.detach().mean().unsqueeze(0),
#                     ]
#                 )
#                 diagnostics["acceptance_rate"] = (
#                     diagnostics["acceptance_rate"] * step + accepted.float().mean()
#                 ) / (step + 1)
#
#         # Return results
#         if return_diagnostics:
#             return current_position, diagnostics
#         return current_position
#
#     # @torch.enable_grad()
#     # def sample(
#     #     self,
#     #     initial_state: torch.Tensor,
#     #     n_steps: int,
#     #     return_diagnostics: bool = False,
#     # ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
#     #     """Generate samples using HMC following the specified steps."""
#     #     device = initial_state.device
#     #
#     #     diagnostics = self._setup_diagnostics() if return_diagnostics else None
#     #     current_position = initial_state.clone().to(device).requires_grad_(True)
#     #     batch_size = current_position.shape[0] if len(current_position.shape) > 1 else 1
#     #     state_shape = (
#     #         current_position.shape[1:]
#     #         if len(current_position.shape) > 1
#     #         else current_position.shape
#     #     )
#     #
#     #     for step in range(n_steps):
#     #         # 1. generate initial momentum
#     #         initial_momentum = self._sample_initial_momentum(
#     #             batch_size, state_shape
#     #         ).to(device)
#     #
#     #         # 2. x(0) = x0
#     #         position = current_position.clone().to(device).requires_grad_(True)
#     #
#     #         # 3. half-step update - momentum
#     #         log_prob = self._compute_log_prob(position)
#     #         grad_log_prob = torch.autograd.grad(
#     #             log_prob.sum(), position, create_graph=False, retain_graph=True
#     #         )[0].to(device)
#     #         momentum = initial_momentum - 0.5 * self.step_size * grad_log_prob
#     #
#     #         # 4. main leapfrog steps
#     #         for l in range(self.n_leapfrog_steps - 1):
#     #             # (a) update position
#     #             position = (position + self.step_size * momentum).to(device)
#     #             position.requires_grad_(True)
#     #
#     #             # (b) update momentum
#     #             log_prob = self._compute_log_prob(position)
#     #             grad_log_prob = torch.autograd.grad(
#     #                 log_prob.sum(), position, create_graph=False, retain_graph=True
#     #             )[0].to(device)
#     #             momentum = momentum - self.step_size * grad_log_prob
#     #
#     #         # 5. last position update
#     #         position = (position + self.step_size * momentum).to(device)
#     #         position.requires_grad_(True)
#     #
#     #         # 6. last half-step momentum update
#     #         log_prob = self._compute_log_prob(position)
#     #         grad_log_prob = torch.autograd.grad(
#     #             log_prob.sum(), position, create_graph=False, retain_graph=True
#     #         )[0].to(device)
#     #         momentum = momentum - 0.5 * self.step_size * grad_log_prob
#     #
#     #         # 7. compute acceptance probability
#     #         initial_hamiltonian = self._compute_hamiltonian(
#     #             current_position, initial_momentum
#     #         ).to(device)
#     #         proposed_hamiltonian = self._compute_hamiltonian(position, momentum).to(
#     #             device
#     #         )
#     #         energy_diff = proposed_hamiltonian - initial_hamiltonian
#     #         acceptance_prob = torch.exp(-energy_diff)
#     #
#     #         # 8. accept/reject step
#     #         accepted = torch.rand_like(acceptance_prob).to(device) < torch.minimum(
#     #             torch.ones_like(acceptance_prob), acceptance_prob
#     #         )
#     #
#     #         # Update state
#     #         current_position = torch.where(
#     #             accepted.unsqueeze(-1), position, current_position
#     #         )
#     #
#     #         # if return_diagnostics:
#     #         #     diagnostics["energies"].append(
#     #         #         initial_hamiltonian.detach().mean().item()
#     #         #     )
#     #         #     diagnostics["acceptance_rate"] = (
#     #         #         diagnostics["acceptance_rate"] * step
#     #         #         + accepted.float().mean().item()
#     #         #     ) / (step + 1)
#     #
#     #         if return_diagnostics:
#     #             diagnostics["energies"] = torch.cat(
#     #                 (
#     #                     diagnostics["energies"],
#     #                     initial_hamiltonian.detach().mean().unsqueeze(0),
#     #                 )
#     #             )
#     #             diagnostics["acceptance_rate"] = (
#     #                 diagnostics["acceptance_rate"] * step + accepted.float().mean()
#     #             ) / (step + 1)
#     #
#     #     # Step 9: Return new state
#     #     if return_diagnostics:
#     #         return current_position, diagnostics
#     #     return current_position
#
#     # @torch.no_grad()
#     # def sample_parallel(
#     #     self,
#     #     initial_states: torch.Tensor,
#     #     n_steps: int,
#     #     return_diagnostics: bool = False,
#     # ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
#     #     """Implementation of parallel Hamiltonian Monte Carlo sampling."""
#     #     current_states = initial_states.to(device=self.device, dtype=self.dtype)
#     #     diagnostics = (
#     #         {"mean_energies": [], "acceptance_rates": []}
#     #         if return_diagnostics
#     #         else None
#     #     )
#     #
#     #     batch_size = current_states.shape[0]
#     #     state_shape = current_states.shape[1:]
#     #
#     #     for _ in range(n_steps):
#     #         # sample initial momentum
#     #         momenta = self._sample_initial_momentum(batch_size, state_shape)
#     #
#     #         # Perform leapfrog integration
#     #         new_states = current_states.clone().requires_grad_(True)
#     #         new_momenta = momenta.clone()
#     #
#     #         for _ in range(self.n_leapfrog_steps):
#     #             # Half-step momentum update
#     #             log_prob = self._compute_log_prob(new_states)
#     #             grad_log_prob = torch.autograd.grad(
#     #                 log_prob.sum(), new_states, create_graph=False, retain_graph=True
#     #             )[0]
#     #             new_momenta -= 0.5 * self.step_size * grad_log_prob
#     #
#     #             # Full-step position update
#     #             new_states = new_states + self.step_size * new_momenta
#     #             new_states.requires_grad_(True)
#     #
#     #             # Full-step momentum update
#     #             log_prob = self._compute_log_prob(new_states)
#     #             grad_log_prob = torch.autograd.grad(
#     #                 log_prob.sum(), new_states, create_graph=False, retain_graph=True
#     #             )[0]
#     #             new_momenta -= self.step_size * grad_log_prob
#     #
#     #         # Final half-step momentum update
#     #         log_prob = self._compute_log_prob(new_states)
#     #         grad_log_prob = torch.autograd.grad(
#     #             log_prob.sum(), new_states, create_graph=False, retain_graph=True
#     #         )[0]
#     #         new_momenta -= 0.5 * self.step_size * grad_log_prob
#     #
#     #         # Compute Hamiltonian
#     #         initial_hamiltonian = self._compute_hamiltonian(current_states, momenta)
#     #         proposed_hamiltonian = self._compute_hamiltonian(new_states, new_momenta)
#     #         energy_diff = proposed_hamiltonian - initial_hamiltonian
#     #         acceptance_prob = torch.exp(-energy_diff)
#     #
#     #         # Accept/reject step
#     #         accept = torch.rand(batch_size, device=self.device) < acceptance_prob
#     #         current_states = torch.where(
#     #             accept.unsqueeze(-1), new_states, current_states
#     #         )
#     #
#     #         if return_diagnostics:
#     #             diagnostics["mean_energies"].append(initial_hamiltonian.mean().item())
#     #             diagnostics["acceptance_rates"].append(accept.float().mean().item())
#     #
#     #     if return_diagnostics:
#     #         diagnostics["mean_energies"] = torch.tensor(diagnostics["mean_energies"])
#     #         diagnostics["acceptance_rates"] = torch.tensor(
#     #             diagnostics["acceptance_rates"]
#     #         )
#     #         return current_states, diagnostics
#     #     return current_states
#
#     @torch.enable_grad()
#     def sample_parallel(
#         self,
#         initial_states: torch.Tensor,
#         n_steps: int,
#         return_diagnostics: bool = False,
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
#         """Implementation of parallel Hamiltonian Monte Carlo sampling."""
#         # Ensure initial states are on correct device
#         current_states = initial_states.to(
#             device=self.device, dtype=self.dtype
#         ).requires_grad_(True)
#
#         diagnostics = (
#             {
#                 "mean_energies": torch.empty(0, device=self.device),
#                 "acceptance_rates": torch.empty(0, device=self.device),
#             }
#             if return_diagnostics
#             else None
#         )
#
#         batch_size = current_states.shape[0]
#         state_shape = current_states.shape[1:]
#
#         for step in range(n_steps):
#             # sample initial momentum (already on correct device from _sample_initial_momentum)
#             momenta = self._sample_initial_momentum(batch_size, state_shape)
#
#             # Initialize new states and momenta
#             new_states = current_states.clone().requires_grad_(True)
#             new_momenta = momenta.clone()
#
#             # First half-step for momentum
#             log_prob = self._compute_log_prob(new_states)
#             grad_log_prob = torch.autograd.grad(
#                 log_prob.sum(), new_states, create_graph=True, retain_graph=True
#             )[0].to(device=self.device)
#             new_momenta = new_momenta - 0.5 * self.step_size * grad_log_prob
#
#             # Leapfrog steps
#             for _ in range(self.n_leapfrog_steps - 1):
#                 # Full step for position
#                 new_states = (new_states + self.step_size * new_momenta).requires_grad_(
#                     True
#                 )
#
#                 # Full step for momentum
#                 log_prob = self._compute_log_prob(new_states)
#                 grad_log_prob = torch.autograd.grad(
#                     log_prob.sum(), new_states, create_graph=True, retain_graph=True
#                 )[0].to(device=self.device)
#                 new_momenta = new_momenta - self.step_size * grad_log_prob
#
#             # Last position update
#             new_states = (new_states + self.step_size * new_momenta).requires_grad_(
#                 True
#             )
#
#             # Final half-step for momentum
#             log_prob = self._compute_log_prob(new_states)
#             grad_log_prob = torch.autograd.grad(
#                 log_prob.sum(), new_states, create_graph=True, retain_graph=True
#             )[0].to(device=self.device)
#             new_momenta = new_momenta - 0.5 * self.step_size * grad_log_prob
#
#             # Compute Hamiltonians (both tensors will be on correct device from _compute_hamiltonian)
#             initial_hamiltonian = self._compute_hamiltonian(current_states, momenta)
#             proposed_hamiltonian = self._compute_hamiltonian(new_states, new_momenta)
#
#             # Metropolis acceptance step
#             energy_diff = proposed_hamiltonian - initial_hamiltonian
#             acceptance_prob = torch.minimum(
#                 torch.ones_like(energy_diff, device=self.device),
#                 torch.exp(-energy_diff),
#             )
#
#             # Accept/reject step
#             accept = (
#                 torch.rand_like(acceptance_prob, device=self.device) < acceptance_prob
#             )
#             current_states = torch.where(
#                 accept.unsqueeze(-1), new_states.detach(), current_states.detach()
#             ).requires_grad_(True)
#
#             if return_diagnostics:
#                 diagnostics["mean_energies"] = torch.cat(
#                     [
#                         diagnostics["mean_energies"],
#                         initial_hamiltonian.mean().unsqueeze(0),
#                     ]
#                 )
#                 diagnostics["acceptance_rates"] = torch.cat(
#                     [
#                         diagnostics["acceptance_rates"],
#                         accept.float().mean().unsqueeze(0),
#                     ]
#                 )
#
#         if return_diagnostics:
#             return current_states, diagnostics
#         return current_states
