r"""Hamiltonian Monte Carlo Sampler Module."""

from typing import Optional, Union, Tuple

import torch

from torchebm.core import (
    BaseSampler,
    BaseScheduler,
    ConstantScheduler,
)
from torchebm.integrators import LeapfrogIntegrator
from torchebm.core.base_model import BaseModel


class HamiltonianMonteCarlo(BaseSampler):
    r"""
    Hamiltonian Monte Carlo sampler.

    Uses Hamiltonian dynamics with Metropolis-Hastings acceptance.

    Args:
        model: Energy-based model to sample from.
        step_size: Step size for leapfrog integration.
        n_leapfrog_steps: Number of leapfrog steps per trajectory.
        mass: Mass matrix (scalar or tensor).
        dtype: Data type for computations.
        device: Device for computations.
    """

    def __init__(
        self,
        model: BaseModel,
        step_size: Union[float, BaseScheduler] = 1e-3,
        n_leapfrog_steps: int = 10,
        mass: Optional[Union[float, torch.Tensor]] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(model=model, dtype=dtype, device=device)
        if isinstance(step_size, BaseScheduler):
            self.register_scheduler("step_size", step_size)
        else:
            if step_size <= 0:
                raise ValueError("step_size must be positive")
            self.register_scheduler("step_size", ConstantScheduler(step_size))

        if n_leapfrog_steps <= 0:
            raise ValueError("n_leapfrog_steps must be positive")

        self.n_leapfrog_steps = n_leapfrog_steps
        self.mass = (
            mass.to(self.device)
            if (mass is not None and not isinstance(mass, float))
            else mass
        )
        self.integrator = LeapfrogIntegrator(
            n_steps=n_leapfrog_steps, device=self.device, dtype=self.dtype
        )

    def _initialize_momentum(self, shape: torch.Size) -> torch.Tensor:
        """
        Initializes momentum variables from a Gaussian distribution.

        The momentum is sampled from \(\mathcal{N}(0, M)\), where `M` is the mass matrix.

        Args:
            shape (torch.Size): The shape of the momentum tensor to generate.

        Returns:
            torch.Tensor: The initialized momentum tensor.
        """
        p = torch.randn(shape, dtype=self.dtype, device=self.device)

        if self.mass is not None:
            # Apply mass matrix (equivalent to sampling from N(0, M))
            if isinstance(self.mass, float):
                p = p * torch.sqrt(
                    torch.tensor(self.mass, dtype=self.dtype, device=self.device)
                )
            else:
                mass_sqrt = torch.sqrt(self.mass)
                p = p * mass_sqrt.view(*([1] * (len(shape) - 1)), -1).expand_as(p)
        return p

    def _compute_kinetic_energy(self, p: torch.Tensor) -> torch.Tensor:
        """
        Computes the kinetic energy of the momentum.

        The kinetic energy is \(K(p) = \frac{1}{2} p^T M^{-1} p\).

        Args:
            p (torch.Tensor): The momentum tensor.

        Returns:
            torch.Tensor: The kinetic energy for each sample in the batch.
        """
        if self.mass is None:
            return 0.5 * torch.sum(p**2, dim=-1)
        elif isinstance(self.mass, float):
            return 0.5 * torch.sum(p**2, dim=-1) / self.mass
        else:
            return 0.5 * torch.sum(
                p**2 / self.mass.view(*([1] * (len(p.shape) - 1)), -1), dim=-1
            )

    def hmc_step(
        self, current_position: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform single HMC step with momentum sampling and Metropolis-Hastings acceptance."""
        batch_size = current_position.shape[0]

        # Sample initial momentum
        current_momentum = self._initialize_momentum(current_position.shape)

        current_energy = torch.clamp(self.model(current_position), min=-1e10, max=1e10)
        current_kinetic = torch.clamp(
            self._compute_kinetic_energy(current_momentum), min=0, max=1e10
        )

        current_hamiltonian = current_energy + current_kinetic

        # Perform leapfrog integration to get proposal
        state = {"x": current_position, "p": current_momentum}
        proposed = self.integrator.step(
            state,
            self.model,
            self.get_scheduled_value("step_size"),
            self.n_leapfrog_steps,
            self.mass,
        )
        proposed_position, proposed_momentum = proposed["x"], proposed["p"]

        # Compute proposed Hamiltonian with similar numerical stability
        proposed_energy = torch.clamp(
            self.model(proposed_position), min=-1e10, max=1e10
        )
        proposed_kinetic = torch.clamp(
            self._compute_kinetic_energy(proposed_momentum), min=0, max=1e10
        )

        proposed_hamiltonian = proposed_energy + proposed_kinetic

        # Metropolis-Hastings acceptance criterion
        # Clamp hamiltonian_diff to avoid overflow in exp()
        hamiltonian_diff = current_hamiltonian - proposed_hamiltonian
        hamiltonian_diff = torch.clamp(hamiltonian_diff, max=50, min=-50)

        acceptance_prob = torch.minimum(
            torch.ones(batch_size, device=self.device), torch.exp(hamiltonian_diff)
        )

        # Accept/reject based on acceptance probability
        random_uniform = torch.rand(batch_size, device=self.device)
        accepted = random_uniform < acceptance_prob
        accepted_mask = accepted.float().view(
            -1, *([1] * (len(current_position.shape) - 1))
        )

        # Update position based on acceptance
        new_position = (
            accepted_mask * proposed_position + (1.0 - accepted_mask) * current_position
        )

        return new_position, acceptance_prob, accepted

    @torch.no_grad()
    def sample(
        self,
        x: Optional[torch.Tensor] = None,
        dim: Optional[int] = None,
        n_steps: int = 100,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples using Hamiltonian Monte Carlo.

        Args:
            x (Optional[torch.Tensor]): The initial state to start sampling from.
            dim (Optional[int]): The dimension of the state space (if `x` is `None`).
            n_steps (int): The number of HMC steps to perform.
            n_samples (int): The number of parallel chains to run.
            return_trajectory (bool): Whether to return the full sample trajectory.
            return_diagnostics (bool): Whether to return sampling diagnostics.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - The final samples.
                - If `return_trajectory` or `return_diagnostics` is `True`, a tuple
                  containing the samples and/or diagnostics.
        """
        # Reset schedulers to their initial values at the start of sampling
        self.reset_schedulers()

        if x is None:
            if dim is None:
                # Try to infer dimension from model
                if hasattr(self.model, "mean") and isinstance(
                    self.model.mean, torch.Tensor
                ):
                    dim = self.model.mean.shape[0]
                else:
                    raise ValueError(
                        "dim must be provided when x is None and cannot be inferred from model"
                    )
            x = torch.randn(n_samples, dim, dtype=self.dtype, device=self.device)
        else:
            x = x.to(device=self.device, dtype=self.dtype)

        # Get dimension from x for later use
        dim = x.shape[1]

        if return_trajectory:
            trajectory = torch.empty(
                (n_samples, n_steps, dim), dtype=self.dtype, device=self.device
            )

        if return_diagnostics:
            diagnostics = self._setup_diagnostics(dim, n_steps, n_samples=n_samples)

        with self.autocast_context():
            for i in range(n_steps):
                self.step_schedulers()

                # Perform single HMC step
                x, acceptance_prob, accepted = self.hmc_step(x)

                if return_trajectory:
                    trajectory[:, i, :] = x

                if return_diagnostics:
                    if n_samples > 1:
                        mean_x = x.mean(dim=0, keepdim=True)
                        var_x = torch.clamp(
                            x.var(dim=0, unbiased=False, keepdim=True),
                            min=1e-10,
                            max=1e10,
                        )
                    else:
                        mean_x = x
                        var_x = torch.zeros_like(x)

                    energy = torch.clamp(self.model(x), min=-1e-10, max=1e10)
                    acceptance_rate = accepted.float().mean()

                    diagnostics[i, 0, :, :] = (
                        mean_x if n_samples > 1 else mean_x.unsqueeze(0)
                    )
                    diagnostics[i, 1, :, :] = (
                        var_x if n_samples > 1 else var_x.unsqueeze(0)
                    )
                    diagnostics[i, 2, :, :] = energy.view(-1, 1).expand(n_samples, dim)
                    diagnostics[i, 3, :, :] = acceptance_rate.expand(n_samples, dim)

        if return_trajectory:
            if return_diagnostics:
                return trajectory.to(dtype=self.dtype), diagnostics.to(
                    dtype=self.dtype
                )  # , acceptance_rates
            return trajectory.to(dtype=self.dtype)

        if return_diagnostics:
            return x.to(dtype=self.dtype), diagnostics.to(
                dtype=self.dtype
            )  # , acceptance_rates

        return x.to(dtype=self.dtype)

    def _setup_diagnostics(
        self, dim: int, n_steps: int, n_samples: int = None
    ) -> torch.Tensor:
        """
        Initializes a tensor to store diagnostics during sampling.

        Args:
            dim (int): The dimensionality of the state space.
            n_steps (int): The number of sampling steps.
            n_samples (Optional[int]): The number of parallel chains.

        Returns:
            torch.Tensor: An empty tensor for storing diagnostics.
        """
        if n_samples is not None:
            return torch.empty(
                (n_steps, 4, n_samples, dim), device=self.device, dtype=self.dtype
            )
        else:
            return torch.empty((n_steps, 4, dim), device=self.device, dtype=self.dtype)
