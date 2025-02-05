from typing import Optional, Union, Tuple

import numpy as np
import torch

from torchebm.core import Sampler
from torchebm.core.energy_function import EnergyFunction, GaussianEnergy


class HamiltonianMonteCarlo1(Sampler):
    """Hamiltonian Monte Carlo sampler implementation.

    References:
        - Implements the HMC based on https://faculty.washington.edu/yenchic/19A_stat535/Lec9_HMC.pdf.
    """

    def __init__(
        self,
        energy_function: EnergyFunction,
        step_size: float = 1e-3,
        n_leapfrog_steps: int = 10,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        mass_matrix: Optional[torch.Tensor] = None,
    ):
        """Initialize Hamiltonian Monte Carlo sampler.

        Args:
            energy_function: The energy function to sample from
            step_size: The step size for leapfrog updates
            n_leapfrog_steps: Number of leapfrog steps per sample
            dtype: Tensor dtype to use
            device: Device to run on
            mass_matrix: Optional mass matrix for momentum sampling
        """
        super().__init__(energy_function, dtype, device)
        self.step_size = step_size
        self.n_leapfrog_steps = n_leapfrog_steps

        # Ensure mass matrix is on correct device and dtype
        if mass_matrix is not None:
            self.mass_matrix = mass_matrix.to(device=self.device, dtype=self.dtype)
        else:
            self.mass_matrix = None

    def _compute_log_prob(self, state: torch.Tensor) -> torch.Tensor:
        """Computes the log-probability (up to a constant) for a given state (position)."""
        state = state.to(device=self.device, dtype=self.dtype)
        if not state.requires_grad:
            state.requires_grad_(True)
        return -self.energy_function(state)

    def _sample_initial_momentum(
        self, batch_size: int, state_shape: tuple
    ) -> torch.Tensor:
        """Samples the initial momentum for a given state (position): ω0 ~ N(0, M^(-1))."""
        momentum = torch.randn(
            (batch_size, *state_shape), device=self.device, dtype=self.dtype
        )

        if self.mass_matrix is not None:
            # Ensure mass matrix is properly shaped and on correct device
            mass_matrix = self.mass_matrix.to(device=self.device, dtype=self.dtype)
            momentum = torch.matmul(mass_matrix, momentum)

        return momentum

    def _kinetic_energy(self, momentum: torch.Tensor) -> torch.Tensor:
        """Compute kinetic energy K(ω) = 1/2 ω^T M^(-1) ω."""
        momentum = momentum.to(device=self.device, dtype=self.dtype)

        if self.mass_matrix is None:
            return torch.sum(0.5 * momentum**2, dim=tuple(range(1, momentum.dim())))

        # Ensure mass matrix is on correct device and properly shaped
        mass_matrix = self.mass_matrix.to(device=self.device, dtype=self.dtype)
        mass_matrix_inverse = torch.inverse(mass_matrix)

        # Compute kinetic energy with mass matrix
        return torch.sum(
            0.5 * torch.matmul(momentum, mass_matrix_inverse) * momentum,
            dim=tuple(range(1, momentum.dim())),
        )

    def _compute_hamiltonian(
        self, position: torch.Tensor, momentum: torch.Tensor
    ) -> torch.Tensor:
        """Compute H(x,ω) = -log r(x) + K(ω)."""
        # Ensure both tensors are on the same device
        position = position.to(device=self.device, dtype=self.dtype)
        momentum = momentum.to(device=self.device, dtype=self.dtype)

        return -self._compute_log_prob(position) + self._kinetic_energy(momentum)

    @torch.enable_grad()
    def sample(
        self,
        initial_state: torch.Tensor,
        n_steps: int,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Generate samples using HMC following the specified steps."""
        # Ensure initial state is on correct device and requires gradients
        current_position = (
            initial_state.to(device=self.device, dtype=self.dtype)
            .clone()
            .requires_grad_(True)
        )

        # Initialize diagnostics if needed
        diagnostics = self._setup_diagnostics() if return_diagnostics else None

        # Handle batch dimension properly
        batch_size = current_position.shape[0] if len(current_position.shape) > 1 else 1
        state_shape = (
            current_position.shape[1:]
            if len(current_position.shape) > 1
            else current_position.shape
        )

        for step in range(n_steps):
            # 1. Generate initial momentum (already on correct device)
            initial_momentum = self._sample_initial_momentum(batch_size, state_shape)

            # 2. Initialize position
            position = current_position.clone().requires_grad_(True)

            # 3. First half-step momentum update
            log_prob = self._compute_log_prob(position)
            grad_log_prob = torch.autograd.grad(
                log_prob.sum(), position, create_graph=True, retain_graph=True
            )[0]
            momentum = initial_momentum - 0.5 * self.step_size * grad_log_prob

            # 4. Main leapfrog steps
            for _ in range(self.n_leapfrog_steps - 1):
                # Update position
                position = (position + self.step_size * momentum).requires_grad_(True)

                # Update momentum
                log_prob = self._compute_log_prob(position)
                grad_log_prob = torch.autograd.grad(
                    log_prob.sum(), position, create_graph=True, retain_graph=True
                )[0]
                momentum = momentum - self.step_size * grad_log_prob

            # 5. Last position update
            position = (position + self.step_size * momentum).requires_grad_(True)

            # 6. Last half-step momentum update
            log_prob = self._compute_log_prob(position)
            grad_log_prob = torch.autograd.grad(
                log_prob.sum(), position, create_graph=True, retain_graph=True
            )[0]
            momentum = momentum - 0.5 * self.step_size * grad_log_prob

            # 7. Compute acceptance probability
            initial_hamiltonian = self._compute_hamiltonian(
                current_position, initial_momentum
            )
            proposed_hamiltonian = self._compute_hamiltonian(position, momentum)
            energy_diff = proposed_hamiltonian - initial_hamiltonian
            acceptance_prob = torch.exp(-energy_diff)

            # 8. Accept/reject step
            accepted = torch.rand_like(
                acceptance_prob, device=self.device
            ) < torch.minimum(
                torch.ones_like(acceptance_prob, device=self.device), acceptance_prob
            )

            # Update state
            current_position = torch.where(
                accepted.unsqueeze(-1),
                position.detach(),  # Important: detach accepted states
                current_position.detach(),  # Important: detach rejected states
            ).requires_grad_(
                True
            )  # Ensure the result requires gradients

            # Update diagnostics if needed
            if return_diagnostics:
                diagnostics["energies"] = torch.cat(
                    [
                        diagnostics["energies"],
                        initial_hamiltonian.detach().mean().unsqueeze(0),
                    ]
                )
                diagnostics["acceptance_rate"] = (
                    diagnostics["acceptance_rate"] * step + accepted.float().mean()
                ) / (step + 1)

        # Return results
        if return_diagnostics:
            return current_position, diagnostics
        return current_position

    # @torch.enable_grad()
    # def sample(
    #     self,
    #     initial_state: torch.Tensor,
    #     n_steps: int,
    #     return_diagnostics: bool = False,
    # ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    #     """Generate samples using HMC following the specified steps."""
    #     device = initial_state.device
    #
    #     diagnostics = self._setup_diagnostics() if return_diagnostics else None
    #     current_position = initial_state.clone().to(device).requires_grad_(True)
    #     batch_size = current_position.shape[0] if len(current_position.shape) > 1 else 1
    #     state_shape = (
    #         current_position.shape[1:]
    #         if len(current_position.shape) > 1
    #         else current_position.shape
    #     )
    #
    #     for step in range(n_steps):
    #         # 1. generate initial momentum
    #         initial_momentum = self._sample_initial_momentum(
    #             batch_size, state_shape
    #         ).to(device)
    #
    #         # 2. x(0) = x0
    #         position = current_position.clone().to(device).requires_grad_(True)
    #
    #         # 3. half-step update - momentum
    #         log_prob = self._compute_log_prob(position)
    #         grad_log_prob = torch.autograd.grad(
    #             log_prob.sum(), position, create_graph=False, retain_graph=True
    #         )[0].to(device)
    #         momentum = initial_momentum - 0.5 * self.step_size * grad_log_prob
    #
    #         # 4. main leapfrog steps
    #         for l in range(self.n_leapfrog_steps - 1):
    #             # (a) update position
    #             position = (position + self.step_size * momentum).to(device)
    #             position.requires_grad_(True)
    #
    #             # (b) update momentum
    #             log_prob = self._compute_log_prob(position)
    #             grad_log_prob = torch.autograd.grad(
    #                 log_prob.sum(), position, create_graph=False, retain_graph=True
    #             )[0].to(device)
    #             momentum = momentum - self.step_size * grad_log_prob
    #
    #         # 5. last position update
    #         position = (position + self.step_size * momentum).to(device)
    #         position.requires_grad_(True)
    #
    #         # 6. last half-step momentum update
    #         log_prob = self._compute_log_prob(position)
    #         grad_log_prob = torch.autograd.grad(
    #             log_prob.sum(), position, create_graph=False, retain_graph=True
    #         )[0].to(device)
    #         momentum = momentum - 0.5 * self.step_size * grad_log_prob
    #
    #         # 7. compute acceptance probability
    #         initial_hamiltonian = self._compute_hamiltonian(
    #             current_position, initial_momentum
    #         ).to(device)
    #         proposed_hamiltonian = self._compute_hamiltonian(position, momentum).to(
    #             device
    #         )
    #         energy_diff = proposed_hamiltonian - initial_hamiltonian
    #         acceptance_prob = torch.exp(-energy_diff)
    #
    #         # 8. accept/reject step
    #         accepted = torch.rand_like(acceptance_prob).to(device) < torch.minimum(
    #             torch.ones_like(acceptance_prob), acceptance_prob
    #         )
    #
    #         # Update state
    #         current_position = torch.where(
    #             accepted.unsqueeze(-1), position, current_position
    #         )
    #
    #         # if return_diagnostics:
    #         #     diagnostics["energies"].append(
    #         #         initial_hamiltonian.detach().mean().item()
    #         #     )
    #         #     diagnostics["acceptance_rate"] = (
    #         #         diagnostics["acceptance_rate"] * step
    #         #         + accepted.float().mean().item()
    #         #     ) / (step + 1)
    #
    #         if return_diagnostics:
    #             diagnostics["energies"] = torch.cat(
    #                 (
    #                     diagnostics["energies"],
    #                     initial_hamiltonian.detach().mean().unsqueeze(0),
    #                 )
    #             )
    #             diagnostics["acceptance_rate"] = (
    #                 diagnostics["acceptance_rate"] * step + accepted.float().mean()
    #             ) / (step + 1)
    #
    #     # Step 9: Return new state
    #     if return_diagnostics:
    #         return current_position, diagnostics
    #     return current_position

    # @torch.no_grad()
    # def sample_parallel(
    #     self,
    #     initial_states: torch.Tensor,
    #     n_steps: int,
    #     return_diagnostics: bool = False,
    # ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    #     """Implementation of parallel Hamiltonian Monte Carlo sampling."""
    #     current_states = initial_states.to(device=self.device, dtype=self.dtype)
    #     diagnostics = (
    #         {"mean_energies": [], "acceptance_rates": []}
    #         if return_diagnostics
    #         else None
    #     )
    #
    #     batch_size = current_states.shape[0]
    #     state_shape = current_states.shape[1:]
    #
    #     for _ in range(n_steps):
    #         # sample initial momentum
    #         momenta = self._sample_initial_momentum(batch_size, state_shape)
    #
    #         # Perform leapfrog integration
    #         new_states = current_states.clone().requires_grad_(True)
    #         new_momenta = momenta.clone()
    #
    #         for _ in range(self.n_leapfrog_steps):
    #             # Half-step momentum update
    #             log_prob = self._compute_log_prob(new_states)
    #             grad_log_prob = torch.autograd.grad(
    #                 log_prob.sum(), new_states, create_graph=False, retain_graph=True
    #             )[0]
    #             new_momenta -= 0.5 * self.step_size * grad_log_prob
    #
    #             # Full-step position update
    #             new_states = new_states + self.step_size * new_momenta
    #             new_states.requires_grad_(True)
    #
    #             # Full-step momentum update
    #             log_prob = self._compute_log_prob(new_states)
    #             grad_log_prob = torch.autograd.grad(
    #                 log_prob.sum(), new_states, create_graph=False, retain_graph=True
    #             )[0]
    #             new_momenta -= self.step_size * grad_log_prob
    #
    #         # Final half-step momentum update
    #         log_prob = self._compute_log_prob(new_states)
    #         grad_log_prob = torch.autograd.grad(
    #             log_prob.sum(), new_states, create_graph=False, retain_graph=True
    #         )[0]
    #         new_momenta -= 0.5 * self.step_size * grad_log_prob
    #
    #         # Compute Hamiltonian
    #         initial_hamiltonian = self._compute_hamiltonian(current_states, momenta)
    #         proposed_hamiltonian = self._compute_hamiltonian(new_states, new_momenta)
    #         energy_diff = proposed_hamiltonian - initial_hamiltonian
    #         acceptance_prob = torch.exp(-energy_diff)
    #
    #         # Accept/reject step
    #         accept = torch.rand(batch_size, device=self.device) < acceptance_prob
    #         current_states = torch.where(
    #             accept.unsqueeze(-1), new_states, current_states
    #         )
    #
    #         if return_diagnostics:
    #             diagnostics["mean_energies"].append(initial_hamiltonian.mean().item())
    #             diagnostics["acceptance_rates"].append(accept.float().mean().item())
    #
    #     if return_diagnostics:
    #         diagnostics["mean_energies"] = torch.tensor(diagnostics["mean_energies"])
    #         diagnostics["acceptance_rates"] = torch.tensor(
    #             diagnostics["acceptance_rates"]
    #         )
    #         return current_states, diagnostics
    #     return current_states

    @torch.enable_grad()
    def sample_parallel(
        self,
        initial_states: torch.Tensor,
        n_steps: int,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Implementation of parallel Hamiltonian Monte Carlo sampling."""
        # Ensure initial states are on correct device
        current_states = initial_states.to(
            device=self.device, dtype=self.dtype
        ).requires_grad_(True)

        diagnostics = (
            {
                "mean_energies": torch.empty(0, device=self.device),
                "acceptance_rates": torch.empty(0, device=self.device),
            }
            if return_diagnostics
            else None
        )

        batch_size = current_states.shape[0]
        state_shape = current_states.shape[1:]

        for step in range(n_steps):
            # sample initial momentum (already on correct device from _sample_initial_momentum)
            momenta = self._sample_initial_momentum(batch_size, state_shape)

            # Initialize new states and momenta
            new_states = current_states.clone().requires_grad_(True)
            new_momenta = momenta.clone()

            # First half-step for momentum
            log_prob = self._compute_log_prob(new_states)
            grad_log_prob = torch.autograd.grad(
                log_prob.sum(), new_states, create_graph=True, retain_graph=True
            )[0].to(device=self.device)
            new_momenta = new_momenta - 0.5 * self.step_size * grad_log_prob

            # Leapfrog steps
            for _ in range(self.n_leapfrog_steps - 1):
                # Full step for position
                new_states = (new_states + self.step_size * new_momenta).requires_grad_(
                    True
                )

                # Full step for momentum
                log_prob = self._compute_log_prob(new_states)
                grad_log_prob = torch.autograd.grad(
                    log_prob.sum(), new_states, create_graph=True, retain_graph=True
                )[0].to(device=self.device)
                new_momenta = new_momenta - self.step_size * grad_log_prob

            # Last position update
            new_states = (new_states + self.step_size * new_momenta).requires_grad_(
                True
            )

            # Final half-step for momentum
            log_prob = self._compute_log_prob(new_states)
            grad_log_prob = torch.autograd.grad(
                log_prob.sum(), new_states, create_graph=True, retain_graph=True
            )[0].to(device=self.device)
            new_momenta = new_momenta - 0.5 * self.step_size * grad_log_prob

            # Compute Hamiltonians (both tensors will be on correct device from _compute_hamiltonian)
            initial_hamiltonian = self._compute_hamiltonian(current_states, momenta)
            proposed_hamiltonian = self._compute_hamiltonian(new_states, new_momenta)

            # Metropolis acceptance step
            energy_diff = proposed_hamiltonian - initial_hamiltonian
            acceptance_prob = torch.minimum(
                torch.ones_like(energy_diff, device=self.device),
                torch.exp(-energy_diff),
            )

            # Accept/reject step
            accept = (
                torch.rand_like(acceptance_prob, device=self.device) < acceptance_prob
            )
            current_states = torch.where(
                accept.unsqueeze(-1), new_states.detach(), current_states.detach()
            ).requires_grad_(True)

            if return_diagnostics:
                diagnostics["mean_energies"] = torch.cat(
                    [
                        diagnostics["mean_energies"],
                        initial_hamiltonian.mean().unsqueeze(0),
                    ]
                )
                diagnostics["acceptance_rates"] = torch.cat(
                    [
                        diagnostics["acceptance_rates"],
                        accept.float().mean().unsqueeze(0),
                    ]
                )

        if return_diagnostics:
            return current_states, diagnostics
        return current_states


class HamiltonianMonteCarlo(Sampler):
    def __init__(
        self,
        energy_function: EnergyFunction,
        step_size: float = 1e-3,
        n_leapfrog_steps: int = 10,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        mass_matrix: Optional[torch.Tensor] = None,
    ):
        """Initialize Hamiltonian Monte Carlo sampler."""
        super().__init__(energy_function, dtype, device)
        self.step_size = step_size
        self.n_leapfrog_steps = n_leapfrog_steps
        self.mass_matrix = (
            mass_matrix.to(device=self.device) if mass_matrix is not None else None
        )

    def _compute_log_prob(self, state: torch.Tensor) -> torch.Tensor:
        """Computes the log-probability for a given state."""
        state = state.to(device=self.device)
        if not state.requires_grad:
            state.requires_grad_(True)
        return -self.energy_function(state)

    def _kinetic_energy(self, momentum: torch.Tensor) -> torch.Tensor:
        """Compute kinetic energy."""
        momentum = momentum.to(device=self.device)
        if self.mass_matrix is None:
            return torch.sum(0.5 * momentum**2, dim=tuple(range(1, momentum.dim())))

        mass_matrix_inverse = torch.inverse(self.mass_matrix)
        return torch.sum(
            0.5 * torch.matmul(momentum, mass_matrix_inverse) * momentum,
            dim=tuple(range(1, momentum.dim())),
        )

    def _compute_hamiltonian(
        self, position: torch.Tensor, momentum: torch.Tensor
    ) -> torch.Tensor:
        """Compute Hamiltonian."""
        position = position.to(device=self.device)
        momentum = momentum.to(device=self.device)

        return -self._compute_log_prob(position) + self._kinetic_energy(momentum)

    def _setup_diagnostics(self) -> dict:
        """Initialize diagnostics dictionary."""
        return {
            "energies": torch.empty(0, device=self.device),
            "acceptance_rate": torch.tensor(0.0, device=self.device),
        }

    def _sample_initial_momentum(
        self, batch_size: int, state_shape: tuple
    ) -> torch.Tensor:
        """sample initial momentum."""
        momentum = torch.randn(
            (batch_size, *state_shape), device=self.device, dtype=self.dtype
        )
        if self.mass_matrix is not None:
            momentum = torch.matmul(self.mass_matrix, momentum)
        return momentum

    @torch.enable_grad()
    def sample(
        self,
        initial_state: torch.Tensor,
        n_steps: int,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Generate samples using HMC."""
        # Ensure initial state is on correct device
        current_position = (
            initial_state.to(device=self.device).clone().requires_grad_(True)
        )

        diagnostics = self._setup_diagnostics() if return_diagnostics else None

        batch_size = current_position.shape[0] if len(current_position.shape) > 1 else 1
        state_shape = (
            current_position.shape[1:]
            if len(current_position.shape) > 1
            else current_position.shape
        )

        for step in range(n_steps):
            # sample initial momentum
            initial_momentum = self._sample_initial_momentum(batch_size, state_shape)

            # Initialize position
            position = current_position.clone().requires_grad_(True)
            momentum = initial_momentum.clone()

            # First half-step momentum update
            log_prob = self._compute_log_prob(position)
            grad_log_prob = torch.autograd.grad(
                log_prob.sum(), position, create_graph=True, retain_graph=True
            )[0].to(device=self.device)
            momentum = momentum - 0.5 * self.step_size * grad_log_prob

            # Leapfrog integration
            for _ in range(self.n_leapfrog_steps - 1):
                position = (position + self.step_size * momentum).requires_grad_(True)
                log_prob = self._compute_log_prob(position)
                grad_log_prob = torch.autograd.grad(
                    log_prob.sum(), position, create_graph=True, retain_graph=True
                )[0].to(device=self.device)
                momentum = momentum - self.step_size * grad_log_prob

            # Last position and momentum updates
            position = (position + self.step_size * momentum).requires_grad_(True)
            log_prob = self._compute_log_prob(position)
            grad_log_prob = torch.autograd.grad(
                log_prob.sum(), position, create_graph=True, retain_graph=True
            )[0].to(device=self.device)
            momentum = momentum - 0.5 * self.step_size * grad_log_prob

            # Compute acceptance probability
            initial_hamiltonian = self._compute_hamiltonian(
                current_position, initial_momentum
            )
            proposed_hamiltonian = self._compute_hamiltonian(position, momentum)
            energy_diff = proposed_hamiltonian - initial_hamiltonian
            acceptance_prob = torch.exp(-energy_diff)

            # Accept/reject step
            uniform_rand = torch.rand_like(acceptance_prob, device=self.device)
            accepted = uniform_rand < torch.minimum(
                torch.ones_like(acceptance_prob, device=self.device), acceptance_prob
            )

            # Update state
            current_position = torch.where(
                accepted.unsqueeze(-1), position.detach(), current_position.detach()
            ).requires_grad_(True)

            # Update diagnostics
            if return_diagnostics:
                diagnostics["energies"] = torch.cat(
                    [
                        diagnostics["energies"],
                        initial_hamiltonian.detach().mean().unsqueeze(0),
                    ]
                )
                diagnostics["acceptance_rate"] = (
                    diagnostics["acceptance_rate"] * step + accepted.float().mean()
                ) / (step + 1)

        if return_diagnostics:
            return current_position, diagnostics
        return current_position

    @torch.enable_grad()
    def sample_parallel(
        self,
        initial_states: torch.Tensor,
        n_steps: int,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Implementation of parallel Hamiltonian Monte Carlo sampling."""
        # Ensure initial states are on correct device
        current_states = initial_states.to(
            device=self.device, dtype=self.dtype
        ).requires_grad_(True)

        diagnostics = (
            {
                "mean_energies": torch.empty(0, device=self.device),
                "acceptance_rates": torch.empty(0, device=self.device),
            }
            if return_diagnostics
            else None
        )

        batch_size = current_states.shape[0]
        state_shape = current_states.shape[1:]

        for step in range(n_steps):
            # sample initial momentum (already on correct device from _sample_initial_momentum)
            momenta = self._sample_initial_momentum(batch_size, state_shape)

            # Initialize new states and momenta
            new_states = current_states.clone().requires_grad_(True)
            new_momenta = momenta.clone()

            # First half-step for momentum
            log_prob = self._compute_log_prob(new_states)
            grad_log_prob = torch.autograd.grad(
                log_prob.sum(), new_states, create_graph=True, retain_graph=True
            )[0].to(device=self.device)
            new_momenta = new_momenta - 0.5 * self.step_size * grad_log_prob

            # Leapfrog steps
            for _ in range(self.n_leapfrog_steps - 1):
                # Full step for position
                new_states = (new_states + self.step_size * new_momenta).requires_grad_(
                    True
                )

                # Full step for momentum
                log_prob = self._compute_log_prob(new_states)
                grad_log_prob = torch.autograd.grad(
                    log_prob.sum(), new_states, create_graph=True, retain_graph=True
                )[0].to(device=self.device)
                new_momenta = new_momenta - self.step_size * grad_log_prob

            # Last position update
            new_states = (new_states + self.step_size * new_momenta).requires_grad_(
                True
            )

            # Final half-step for momentum
            log_prob = self._compute_log_prob(new_states)
            grad_log_prob = torch.autograd.grad(
                log_prob.sum(), new_states, create_graph=True, retain_graph=True
            )[0].to(device=self.device)
            new_momenta = new_momenta - 0.5 * self.step_size * grad_log_prob

            # Compute Hamiltonians (both tensors will be on correct device from _compute_hamiltonian)
            initial_hamiltonian = self._compute_hamiltonian(current_states, momenta)
            proposed_hamiltonian = self._compute_hamiltonian(new_states, new_momenta)

            # Metropolis acceptance step
            energy_diff = proposed_hamiltonian - initial_hamiltonian
            acceptance_prob = torch.minimum(
                torch.ones_like(energy_diff, device=self.device),
                torch.exp(-energy_diff),
            )

            # Accept/reject step
            accept = (
                torch.rand_like(acceptance_prob, device=self.device) < acceptance_prob
            )
            current_states = torch.where(
                accept.unsqueeze(-1), new_states.detach(), current_states.detach()
            ).requires_grad_(True)

            if return_diagnostics:
                diagnostics["mean_energies"] = torch.cat(
                    [
                        diagnostics["mean_energies"],
                        initial_hamiltonian.mean().unsqueeze(0),
                    ]
                )
                diagnostics["acceptance_rates"] = torch.cat(
                    [
                        diagnostics["acceptance_rates"],
                        accept.float().mean().unsqueeze(0),
                    ]
                )

        if return_diagnostics:
            return current_states, diagnostics
        return current_states


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


# test_hmc()


import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple


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


if __name__ == "__main__":
    # visualize_sampling_trajectory(n_steps=100, step_size=0.1, n_leapfrog_steps=10)

    visualize_sampling_trajectory(
        n_steps=200,
        step_size=0.05,
        n_leapfrog_steps=15,
        figsize=(18, 6),
        save_path="hmc_analysis.png",
    )
