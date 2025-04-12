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
    samples, diagnostics = hmc.sample_chain(initial_state, n_steps=100, return_diagnostics=True)
    print(f"Samples: {samples.shape}")
    print(f"Diagnostics: {diagnostics.keys()}")
    ```

---

## Mathematical Foundations

!!! info "Hamiltonian Dynamics in HMC"
    HMC combines statistical sampling with concepts from classical mechanics. It introduces an auxiliary momentum variable \( p \) and defines a Hamiltonian:

    $$
    H(q, p) = U(q) + K(p)
    $$

    - **Potential Energy**: \( U(q) = -\log \pi(q) \), where \( \pi(q) \) is the target distribution.
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
        p_{t + \frac{\epsilon}{2}} = p_t - \frac{\epsilon}{2} \nabla U(q_t)
        $$
    2. **Position Full-Step**:
        $$
        q_{t + 1} = q_t + \epsilon M^{-1} p_{t + \frac{\epsilon}{2}}
        $$
    3. **Momentum Half-Step**:
        $$
        p_{t + 1} = p_{t + \frac{\epsilon}{2}} - \frac{\epsilon}{2} \nabla U(q_{t + 1})
        $$

    Here, \( \epsilon \) is the step size, and the process is repeated for \( L \) leapfrog steps.

### Acceptance Step

!!! note "Metropolis-Hastings Correction"
    After proposing a new state \( (q_{t + 1}, p_{t + 1}) \), HMC applies an acceptance criterion to ensure detailed balance:

    $$
    \alpha = \min \left( 1, \exp \left( H(q_t, p_t) - H(q_{t + 1}, p_{t + 1}) \right) \right)
    $$

    The proposal is accepted with probability \( \alpha \), correcting for numerical errors in the leapfrog integration.

---

## Practical Considerations

!!! warning "Tuning Parameters"
    - **Step Size (\( \epsilon \))**: Too large a step size can lead to unstable trajectories; too small reduces efficiency.
    - **Number of Leapfrog Steps (\( L \))**: Affects the distance traveled per proposal—balance exploration vs. computational cost.
    - **Mass Matrix (\( M \))**: Adjusting \( M \) can improve sampling in distributions with varying scales.

!!! question "How to Diagnose Issues?"
    Use diagnostics to check:
    - Acceptance rates (ideal: 0.6–0.8).
    - Energy conservation (should be relatively stable).
    - Autocorrelation of samples (should decrease with lag).

!!! warning "Common Pitfalls"
    - **Low Acceptance Rate**: If the acceptance rate is too low, it may indicate that the step size is too large or the number of leapfrog steps is too high. Try reducing the step size or decreasing the number of leapfrog steps.
    - **High Correlation Between Samples**: If samples are highly correlated, it may indicate that the step size is too small or the number of leapfrog steps is too few. Increase the step size or the number of leapfrog steps to improve exploration.
    - **Divergence or NaN Values**: Numerical instability or poor parameter choices can lead to divergent behavior or NaN values. Ensure that the energy function and its gradients are correctly implemented and that parameters are appropriately scaled.

---

## Advanced Insights

!!! abstract "Why HMC Outperforms Other MCMC Methods"
    HMC's use of gradients and dynamics reduces random-walk behavior, making it particularly effective for:
    - High-dimensional spaces.
    - Multimodal distributions (with proper tuning).
    - Models with strong correlations between variables.

???+ info "Further Reading"
    - [Hamiltonian Mechanics Explained](docs/blog/posts/hamiltonian-mechanics.md)
    - Neal, R. M. (2011). "MCMC using Hamiltonian dynamics." *Handbook of Markov Chain Monte Carlo*.
"""

from functools import partial
from typing import Optional, Union, Tuple, Callable

import torch

from torchebm.core import BaseSampler
from torchebm.core.base_energy_function import BaseEnergyFunction


class HamiltonianMonteCarlo(BaseSampler):
    r"""
    Hamiltonian Monte Carlo sampler for efficient exploration of complex probability distributions.

    This class implements the Hamiltonian Monte Carlo algorithm, which uses concepts from
    Hamiltonian mechanics to generate more efficient proposals than traditional random-walk
    methods. By introducing an auxiliary momentum variable and simulating Hamiltonian dynamics,
    HMC can make distant proposals with high acceptance probability, particularly in
    high-dimensional spaces.

    The method works by:
    1. Augmenting the state space with momentum variables
    2. Simulating Hamiltonian dynamics using leapfrog integration
    3. Accepting or rejecting proposals using a Metropolis-Hastings criterion

    !!! note "Algorithm Summary"
        1. If `x` is not provided, initialize it with Gaussian noise.
        2. For each step:
           a. Sample momentum from Gaussian distribution.
           b. Perform leapfrog integration for `n_leapfrog_steps` steps.
           c. Accept or reject the proposal based on Metropolis-Hastings criterion.
        3. Optionally track trajectory and diagnostics.

    !!! tip "Key Advantages"
        - **Efficiency**: Performs well in high dimensions by avoiding random walk behavior
        - **Exploration**: Can efficiently traverse complex probability landscapes
        - **Energy Conservation**: Uses symplectic integrators that approximately preserve energy
        - **Adaptability**: Can be adjusted through mass matrices to handle varying scales

    Args:
        energy_function (BaseEnergyFunction): Energy function to sample from.
        step_size (float): Step size for leapfrog updates.
        n_leapfrog_steps (int): Number of leapfrog steps per proposal.
        mass (Optional[Tuple[float, torch.Tensor]]): Optional mass matrix or scalar for momentum sampling.
        dtype (torch.dtype): Data type to use for computations.
        device (Optional[Union[Tuple[str, torch.device]]]): Device to run computations on.

    Raises:
        ValueError: For invalid parameter ranges

    Methods:
        _initialize_momentum(shape): Generate initial momentum from Gaussian distribution.
        _compute_kinetic_energy(p): Compute the kinetic energy of the momentum.
        _leapfrog_step(position, momentum, gradient_fn): Perform a single leapfrog step.
        _leapfrog_integration(position, momentum): Perform full leapfrog integration.
        hmc_step(current_position): Perform one HMC step with Metropolis-Hastings acceptance.
        sample_chain(x, dim, n_steps, n_samples, return_trajectory, return_diagnostics): Run the sampling process.
        _setup_diagnostics(dim, n_steps, n_samples): Initialize the diagnostics.

    !!! example "Basic Usage"
        ```python
        # Define energy function for a 2D Gaussian
        energy_fn = GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2))

        # Initialize HMC sampler
        sampler = HamiltonianMonteCarlo(
            energy_function=energy_fn,
            step_size=0.1,
            n_leapfrog_steps=10
        )

        # Sample 100 points from 5 parallel chains
        samples = sampler.sample_chain(
            dim=2,
            n_steps=100,
            n_samples=5
        )
        ```

    !!! warning "Parameter Relationships"
        - Decreasing `step_size` improves stability but may reduce mixing.
        - Increasing `n_leapfrog_steps` allows exploring more distant regions but increases computation.
        - The `mass` parameter can be tuned to match the geometry of the target distribution.
    """

    def __init__(
        self,
        energy_function: BaseEnergyFunction,
        step_size: float = 0.1,
        n_leapfrog_steps: int = 10,
        mass: Optional[Tuple[float, torch.Tensor]] = None,
        dtype: torch.Tensor = torch.float32,
        device: Optional[Union[Tuple[str, torch.device]]] = None,
    ):
        """Initialize the Hamiltonian Monte Carlo sampler.

        Args:
            energy_function: Energy function to sample from.
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
        super().__init__(energy_function=energy_function, dtype=dtype, device=device)
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if n_leapfrog_steps <= 0:
            raise ValueError("n_leapfrog_steps must be positive")

        # Ensure device consistency: convert device to torch.device and move energy_function
        if device is not None:
            self.device = torch.device(device)
            energy_function = energy_function.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.step_size = step_size
        self.n_leapfrog_steps = n_leapfrog_steps
        self.energy_function = energy_function
        if mass is not None and not isinstance(mass, float):
            self.mass = mass.to(self.device)
        else:
            self.mass = mass

    def _initialize_momentum(self, shape: torch.Size) -> torch.Tensor:
        """Initialize momentum variables from Gaussian distribution.

        For HMC, momentum variables are sampled from a multivariate Gaussian distribution
        determined by the mass matrix. The kinetic energy is then:
        K(p) = p^T M^(-1) p / 2

        Args:
            shape: Size of the momentum tensor to generate.

        Returns:
            Momentum tensor drawn from appropriate Gaussian distribution.

        Note:
            When using a mass matrix M, we sample from N(0, M) rather than
            transforming samples from N(0, I).
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
        """Compute the kinetic energy given momentum.

        The kinetic energy is defined as:
        $$ K(p) = p^T M^(-1) p / 2 $$

        Args:
            p: Momentum tensor.

        Returns:
            Kinetic energy for each sample in the batch.
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
        """Perform a single leapfrog integration step.

        Implements the symplectic leapfrog integrator for Hamiltonian dynamics:
        1. Half-step momentum update: p(t+ε/2) = p(t) - (ε/2)∇U(q(t))
        2. Full-step position update: q(t+ε) = q(t) + εM^(-1)p(t+ε/2)
        3. Half-step momentum update: p(t+ε) = p(t+ε/2) - (ε/2)∇U(q(t+ε))

        Args:
            position: Current position tensor.
            momentum: Current momentum tensor.
            gradient_fn: Function to compute gradient of potential energy.

        Returns:
            Tuple of (new_position, new_momentum).
        """
        # Calculate gradient for half-step momentum update with numerical safeguards
        grad = gradient_fn(position)
        # Clip extreme gradient values to prevent instability
        grad = torch.clamp(grad, min=-1e6, max=1e6)

        # Half-step momentum update
        p_half = momentum - 0.5 * self.step_size * grad

        # Full-step position update with mass matrix adjustment
        if self.mass is None:
            x_new = position + self.step_size * p_half
        else:
            if isinstance(self.mass, float):
                # Ensure mass is positive to avoid division issues
                safe_mass = max(self.mass, 1e-10)
                x_new = position + self.step_size * p_half / safe_mass
            else:
                # Create safe mass tensor avoiding zeros or negative values
                safe_mass = torch.clamp(self.mass, min=1e-10)
                x_new = position + self.step_size * p_half / safe_mass.view(
                    *([1] * (len(position.shape) - 1)), -1
                )

        # Half-step momentum update with gradient clamping
        grad_new = gradient_fn(x_new)
        grad_new = torch.clamp(grad_new, min=-1e6, max=1e6)
        p_new = p_half - 0.5 * self.step_size * grad_new

        return x_new, p_new

    def _leapfrog_integration(
        self, position: torch.Tensor, momentum: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a full leapfrog integration for n_leapfrog_steps.

        Applies multiple leapfrog steps to simulate Hamiltonian dynamics
        for a trajectory of specified length. This is the core of the HMC
        proposal generation.

        Args:
            position: Initial position tensor.
            momentum: Initial momentum tensor.

        Returns:
            Tuple of (final_position, final_momentum) after integration.
        """
        gradient_fn = partial(self.energy_function.gradient)
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
        """Perform a single HMC step with Metropolis-Hastings acceptance.

        This implements the core HMC algorithm:
        1. Sample initial momentum
        2. Compute initial Hamiltonian
        3. Perform leapfrog integration to propose new state
        4. Compute final Hamiltonian
        5. Accept/reject based on Metropolis-Hastings criterion

        Args:
            current_position: Current position tensor of shape (batch_size, dim).

        Returns:
            Tuple containing:
            - new_position: Updated position tensor
            - acceptance_prob: Probability of accepting each proposal
            - accepted: Boolean mask indicating which proposals were accepted
        """
        batch_size = current_position.shape[0]

        # Sample initial momentum
        current_momentum = self._initialize_momentum(current_position.shape)

        # Compute current Hamiltonian: H = U(q) + K(p)
        # Add numerical stability with clamping
        current_energy = self.energy_function(current_position)
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
        proposed_energy = self.energy_function(proposed_position)
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
    def sample_chain(
        self,
        x: Optional[torch.Tensor] = None,
        dim: int = None,
        n_steps: int = 100,
        n_samples: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples using Hamiltonian Monte Carlo.

        Runs an HMC chain for a specified number of steps, optionally returning
        the entire trajectory and/or diagnostics. The HMC algorithm uses Hamiltonian
        dynamics with leapfrog integration to propose samples efficiently, particularly
        in high-dimensional spaces.

        Args:
            x: Initial state to start sampling from. If None, random initialization is used.
            dim: Dimension of the state space when x is None. If None, will attempt to infer from the energy function.
            n_steps: Number of HMC steps to perform.
            n_samples: Number of parallel chains to run.
            return_trajectory: If True, return the entire trajectory of samples.
            return_diagnostics: If True, return diagnostics about the sampling process.

        Returns:
            If return_trajectory=False and return_diagnostics=False:
                Tensor of shape (n_samples, dim) with final samples.
            If return_trajectory=True and return_diagnostics=False:
                Tensor of shape (n_samples, n_steps, dim) with the trajectory of all samples.
            If return_diagnostics=True:
                Tuple of (samples, diagnostics) where diagnostics contains information about
                the sampling process, including mean, variance, energy values, and acceptance rates.

        Note:
            This method uses automatic mixed precision when available on CUDA devices
            to improve performance while maintaining numerical stability for the
            Hamiltonian dynamics simulation.

        Example:
            ```python
            # Run 10 parallel chains for 1000 steps
            samples, diagnostics = hmc.sample_chain(
                dim=10,
                n_steps=1000,
                n_samples=10,
                return_diagnostics=True
            )

            # Plot acceptance rates
            import matplotlib.pyplot as plt
            plt.plot(diagnostics[:-1, 3, 0, 0].cpu().numpy())
            plt.ylabel('Acceptance Rate')
            plt.xlabel('Step')
            plt.show()
            ```
        """
        if x is None:
            # If dim is not provided, try to infer from the energy function
            if dim is None:
                # Check if it's GaussianEnergy which has mean attribute
                if hasattr(self.energy_function, "mean"):
                    dim = self.energy_function.mean.shape[0]
                else:
                    raise ValueError(
                        "dim must be provided when x is None and cannot be inferred from the energy function"
                    )
            x = torch.randn(n_samples, dim, dtype=self.dtype, device=self.device)
        else:
            x = x.to(self.device)

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

        with torch.amp.autocast(
            device_type="cuda" if self.device.type == "cuda" else "cpu"
        ):
            for i in range(n_steps):
                # Perform single HMC step
                x, acceptance_prob, accepted = self.hmc_step(x)

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
                    energy = self.energy_function(
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
                return trajectory, diagnostics  # , acceptance_rates
            return trajectory

        if return_diagnostics:
            return x, diagnostics  # , acceptance_rates

        return x

    def _setup_diagnostics(
        self, dim: int, n_steps: int, n_samples: int = None
    ) -> torch.Tensor:
        """Initialize diagnostics tensor to track HMC sampling metrics.

        Creates a tensor to store diagnostics information during sampling, including:
        - Mean of samples (dimension 0)
        - Variance of samples (dimension 1)
        - Energy values (dimension 2)
        - Acceptance rates (dimension 3)

        Args:
            dim: Dimensionality of the state space.
            n_steps: Number of sampling steps.
            n_samples: Number of parallel chains (if None, assumed to be 1).

        Returns:
            Empty tensor of shape (n_steps, 4, n_samples, dim) to store diagnostics.
        """
        if n_samples is not None:
            return torch.empty(
                (n_steps, 4, n_samples, dim), device=self.device, dtype=self.dtype
            )
        else:
            return torch.empty((n_steps, 4, dim), device=self.device, dtype=self.dtype)
