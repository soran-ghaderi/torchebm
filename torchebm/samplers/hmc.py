r"""Hamiltonian Monte Carlo Sampler Module."""

from functools import partial
from typing import Optional, Union, Tuple, Callable

import torch

from torchebm.core import BaseSampler, BaseScheduler, ConstantScheduler
from torchebm.core.base_model import BaseModel


class HamiltonianMonteCarlo(BaseSampler):
    r"""
    Hamiltonian Monte Carlo (HMC) sampler.

    HMC is an MCMC algorithm that uses Hamiltonian dynamics to generate efficient
    proposals for exploring complex probability distributions. It is particularly
    effective in high-dimensional spaces.

    Args:
        model (BaseModel): The energy-based model to sample from.
        step_size (Union[float, BaseScheduler]): The step size for leapfrog integration.
        n_leapfrog_steps (int): The number of leapfrog steps per trajectory.
        mass (Optional[Union[float, torch.Tensor]]): The mass matrix (or scalar)
            for momentum. Defaults to identity.
        dtype (torch.dtype): The data type for computations.
        device (Optional[Union[str, torch.device]]): The device for computations.
    """

    def __init__(
        self,
        model: BaseModel,
        step_size: Union[float, BaseScheduler] = 1e-3,
        n_leapfrog_steps: int = 10,
        mass: Optional[Tuple[float, torch.Tensor]] = None,
        dtype: torch.Tensor = torch.float32,
        device: Optional[Union[Tuple[str, torch.device]]] = None,
        *args,
        **kwargs,
    ):
        """Initialize the Hamiltonian Monte Carlo sampler.

        Args:
            model: Energy function to sample from.
            step_size: Step size for leapfrog integration (epsilon in equations).
            n_leapfrog_steps: Number of leapfrog steps per HMC trajectory.
            mass: Optional mass parameter or matrix for momentum.
                If float: Uses scalar mass for all dimensions.
                If Tensor: Uses diagonal mass matrix.
                If None: Uses identity mass matrix.
            dtype: Data type for computations.
            device: Device to run computations on ("cpu" or "cuda").

        Raises:
            ValueError: If step_size or n_leapfrog_steps is non-positive.
        """
        super().__init__(model=model, dtype=dtype, device=device)
        if isinstance(step_size, BaseScheduler):
            self.register_scheduler("step_size", step_size)
        else:
            if step_size <= 0:
                raise ValueError("step_size must be positive")
            self.register_scheduler("step_size", ConstantScheduler(step_size))

        if n_leapfrog_steps <= 0:
            raise ValueError("n_leapfrog_steps must be positive")

        # Ensure device consistency: convert device to torch.device and move energy_function
        # if device is not None:
        #     self.device = torch.device(device)
        #     energy_function = energy_function.to(self.device)
        # else:
        #     self.device = torch.device("cpu")

        # Respect user-provided dtype
        self.dtype = dtype
        self.n_leapfrog_steps = n_leapfrog_steps
        if mass is not None and not isinstance(mass, float):
            self.mass = mass.to(self.device)
        else:
            self.mass = mass

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

    def _leapfrog_step(
        self, position: torch.Tensor, momentum: torch.Tensor, gradient_fn: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Performs a single leapfrog integration step to simulate Hamiltonian dynamics.

        Args:
            position (torch.Tensor): The current position tensor.
            momentum (torch.Tensor): The current momentum tensor.
            gradient_fn (Callable): A function that computes the gradient of the potential energy.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The updated position and momentum.
        """
        step_size = self.get_scheduled_value("step_size")

        # Calculate gradient for half-step momentum update with numerical safeguards
        gradient = gradient_fn(position)
        # Clip extreme gradient values to prevent instability
        gradient = torch.clamp(gradient, min=-1e6, max=1e6)

        # Half-step momentum update
        p_half = momentum - 0.5 * step_size * gradient

        # Full-step position update with mass matrix adjustment
        if self.mass is None:
            x_new = position + step_size * p_half
        else:
            if isinstance(self.mass, float):
                # Ensure mass is positive to avoid division issues
                safe_mass = max(self.mass, 1e-10)
                x_new = position + step_size * p_half / safe_mass
            else:
                # Create safe mass tensor avoiding zeros or negative values
                safe_mass = torch.clamp(self.mass, min=1e-10)
                x_new = position + step_size * p_half / safe_mass.view(
                    *([1] * (len(position.shape) - 1)), -1
                )

        # Half-step momentum update with gradient clamping
        grad_new = gradient_fn(x_new)
        grad_new = torch.clamp(grad_new, min=-1e6, max=1e6)
        p_new = p_half - 0.5 * step_size * grad_new

        return x_new, p_new

    def _leapfrog_integration(
        self, position: torch.Tensor, momentum: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a full leapfrog integration trajectory.

        Args:
            position (torch.Tensor): The initial position tensor.
            momentum (torch.Tensor): The initial momentum tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The final position and momentum.
        """
        gradient_fn = partial(self.model.gradient)
        x = position
        p = momentum

        # Add check for NaN values before starting integration
        if torch.isnan(x).any() or torch.isnan(p).any():
            # Replace NaN values with zeros
            x = torch.nan_to_num(x, nan=0.0)
            p = torch.nan_to_num(p, nan=0.0)

        for _ in range(self.n_leapfrog_steps):
            x, p = self._leapfrog_step(x, p, gradient_fn)

            # Check for NaN values after each step
            if torch.isnan(x).any() or torch.isnan(p).any():
                # If NaN values appear, break the integration
                # Replace NaN with zeros and return current state
                x = torch.nan_to_num(x, nan=0.0)
                p = torch.nan_to_num(p, nan=0.0)
                break

        return x, p

    def hmc_step(
        self, current_position: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a single HMC step, including momentum sampling, leapfrog
        integration, and a Metropolis-Hastings acceptance test.

        Args:
            current_position (torch.Tensor): The current position tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - The new position tensor.
                - The acceptance probability for each sample.
                - A boolean mask indicating which proposals were accepted.
        """
        batch_size = current_position.shape[0]

        # Sample initial momentum
        current_momentum = self._initialize_momentum(current_position.shape)

        # Compute current Hamiltonian: H = U(q) + K(p)
        # Add numerical stability with clamping
        current_energy = self.model(current_position)
        current_energy = torch.clamp(
            current_energy, min=-1e10, max=1e10
        )  # Prevent extreme energy values

        current_kinetic = self._compute_kinetic_energy(current_momentum)
        current_kinetic = torch.clamp(
            current_kinetic, min=0, max=1e10
        )  # Kinetic energy should be non-negative

        current_hamiltonian = current_energy + current_kinetic

        # Perform leapfrog integration to get proposal
        proposed_position, proposed_momentum = self._leapfrog_integration(
            current_position, current_momentum
        )

        # Compute proposed Hamiltonian with similar numerical stability
        proposed_energy = self.model(proposed_position)
        proposed_energy = torch.clamp(proposed_energy, min=-1e10, max=1e10)

        proposed_kinetic = self._compute_kinetic_energy(proposed_momentum)
        proposed_kinetic = torch.clamp(proposed_kinetic, min=0, max=1e10)

        proposed_hamiltonian = proposed_energy + proposed_kinetic

        # Metropolis-Hastings acceptance criterion
        # Clamp hamiltonian_diff to avoid overflow in exp()
        hamiltonian_diff = current_hamiltonian - proposed_hamiltonian
        hamiltonian_diff = torch.clamp(hamiltonian_diff, max=50, min=-50)

        acceptance_prob = torch.min(
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
        dim: int = None,
        n_steps: int = 100,
        n_samples: int = 1,
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
            # If dim is not provided, try to infer from the energy function
            if dim is None:
                # Check if it's GaussianEnergy which has mean attribute
                if hasattr(self.model, "mean"):
                    dim = self.model.mean.shape[0]
                else:
                    raise ValueError(
                        "dim must be provided when x is None and cannot be inferred from the energy function"
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
            acceptance_rates = torch.zeros(
                n_steps, device=self.device, dtype=self.dtype
            )

        with self.autocast_context():
            for i in range(n_steps):
                # Perform single HMC step
                x, acceptance_prob, accepted = self.hmc_step(x)

                # Step all schedulers after each HMC step
                scheduler_values = self.step_schedulers()

                if return_trajectory:
                    trajectory[:, i, :] = x

                if return_diagnostics:
                    # Calculate diagnostics with numerical stability safeguards

                    if n_samples > 1:
                        # mean_x = x.mean(dim=0).unsqueeze(0).expand_as(x)
                        mean_x = x.mean(dim=0, keepdim=True)

                        # Clamp variance calculations to prevent NaN values
                        # First compute variance in a numerically stable way
                        # and then clamp to ensure positive finite values
                        # x_centered = x - mean_x
                        # var_x = torch.mean(x_centered**2, dim=0)
                        var_x = x.var(dim=0, unbiased=False, keepdim=True)
                        var_x = torch.clamp(
                            var_x, min=1e-10, max=1e10
                        )  # Prevent zero/extreme variances
                        # var_x = var_x.unsqueeze(0).expand_as(x)
                    else:
                        # For single sample, mean and variance are trivial
                        mean_x = x.clone()
                        var_x = torch.zeros_like(x)

                    # Energy values (ensure finite values)
                    energy = self.model(
                        x
                    )  # assumed to have shape (n_samples,)
                    energy = torch.clamp(
                        energy, min=-1e10, max=1e10
                    )  # Prevent extreme energy values
                    energy = energy.unsqueeze(1).expand_as(x)

                    # Acceptance rate is already between 0 and 1
                    acceptance_rate = accepted.float().mean()
                    acceptance_rate_expanded = torch.ones_like(x) * acceptance_rate

                    # Stack diagnostics
                    diagnostics[i, 0, :, :] = mean_x
                    diagnostics[i, 1, :, :] = var_x
                    diagnostics[i, 2, :, :] = energy
                    diagnostics[i, 3, :, :] = acceptance_rate_expanded

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
