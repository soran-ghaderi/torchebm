r"""
Langevin Dynamics Sampler Module.

This module provides an implementation of the Langevin Dynamics algorithm, a gradient-based Markov Chain Monte Carlo (MCMC) method. It leverages stochastic differential equations to sample from complex probability distributions, making it a lightweight yet effective tool for Bayesian inference and generative modeling.

!!! success "Key Features"
    - Gradient-based sampling with stochastic updates.
    - Customizable step sizes and noise scales for flexible tuning.
    - Optional diagnostics and trajectory tracking for analysis.

---

## Module Components

Classes:
    LangevinDynamics: Core class implementing the Langevin Dynamics sampler.

---

## Usage Example

!!! example "Sampling from a Custom Energy Function"
    ```python
    from torchebm.samplers.mcmc.langevin import LangevinDynamics
    from torchebm.core.energy_function import GaussianEnergy
    import torch

    # Define a 2D Gaussian energy function
    energy_fn = GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2))

    # Initialize Langevin sampler
    sampler = LangevinDynamics(energy_fn, step_size=0.01, noise_scale=0.1)

    # Starting points for 5 chains
    initial_state = torch.randn(5, 2)

    # Run sampling
    samples, diagnostics = sampler.sample_chain(
        x=initial_state, n_steps=100, n_samples=5, return_diagnostics=True
    )
    print(f"Samples shape: {samples.shape}")
    print(f"Diagnostics keys: {diagnostics.shape}")
    ```

---

## Mathematical Foundations

!!! info "Langevin Dynamics Overview"
    Langevin Dynamics simulates a stochastic process governed by the Langevin equation. For a state \( x_t \), the discretized update rule is:

    $$
    x_{t+1} = x_t - \eta \nabla U(x_t) + \sqrt{2\eta} \epsilon_t
    $$

    - \( U(x) \): Potential energy, where \( U(x) = -\log p(x) \) and \( p(x) \) is the target distribution.
    - \( \eta \): Step size controlling the gradient descent.
    - \( \epsilon_t \sim \mathcal{N}(0, I) \): Gaussian noise introducing stochasticity.

    Over time, this process converges to samples from the Boltzmann distribution:

    $$
    p(x) \propto e^{-U(x)}
    $$

!!! tip "Why Use Langevin Dynamics?"
    - **Simplicity**: Requires only first-order gradients, making it computationally lighter than methods like HMC.
    - **Exploration**: The noise term prevents the sampler from getting stuck in local minima.
    - **Flexibility**: Applicable to a wide range of energy-based models and score-based generative tasks.

---

## Practical Considerations

!!! warning "Parameter Tuning Guide"
    - **Step Size ($\eta$)**:
        - Too large: Instability and divergence
        - Too small: Slow convergence
        - Rule of thumb: Start with $\eta \approx 10^{-3}$ to $10^{-5}$
    - **Noise Scale ($\beta^{-1/2}$)**:
        - Controls exploration-exploitation tradeoff
        - Higher values help escape local minima
    - **Decay Rate** (future implementation):
        - Momentum-like term for accelerated convergence

!!! tip "Diagnostics Interpretation"
    Use `return_diagnostics=True` to monitor:
    - **Mean/Variance**: Track distribution stationarity
    - **Energy Gradients**: Check for vanishing/exploding gradients
    - **Autocorrelation**: Assess mixing efficiency

!!! question "When to Choose Langevin Over HMC?"
    | Criterion        | Langevin          | HMC              |
    |------------------|-------------------|------------------|
    | Computational Cost | Lower            | Higher           |
    | Tuning Complexity | Simpler          | More involved    |
    | High Dimensions   | Efficient         | More efficient   |
    | Multimodal Targets| May need annealing| Better exploration|


!!! question "How to Diagnose Sampling?"
    Check diagnostics for:
    - Sample mean and variance convergence.
    - Gradient magnitudes (should stabilize).
    - Energy trends over iterations.

???+ info "Further Reading"
    - [Langevin Dynamics Basics](https://friedmanroy.github.io/blog/2022/Langevin/)
    - [Score-Based Models and Langevin](https://yang-song.net/blog/2021/score/)
    - [Practical Langevin Tutorial](https://ericmjl.github.io/score-models/notebooks/02-langevin-dynamics.html)
"""

import time
from typing import Optional, Union, Tuple, List
from functools import partial

import torch

from torchebm.core.base_energy_function import BaseEnergyFunction, GaussianEnergy
from torchebm.core.base_sampler import BaseSampler


class LangevinDynamics(BaseSampler):
    r"""
    Langevin Dynamics sampler implementing discretized gradient-based MCMC.

    This class implements the Langevin Dynamics algorithm, a gradient-based MCMC method that samples from a target
    distribution defined by an energy function. It uses a stochastic update rule combining gradient descent with Gaussian noise to explore the energy landscape.

    Each step updates the state $x_t$ according to the discretized Langevin equation:

    $$x_{t+1} = x_t - \eta \nabla_x U(x_t) + \sqrt{2\eta} \epsilon_t$$

    where $\epsilon_t \sim \mathcal{N}(0, I)$ and $\eta$ is the step size.

    This process generates samples that asymptotically follow the Boltzmann distribution:


    $$p(x) \propto e^{-U(x)}$$

    where $U(x)$ defines the energy landscape.

    !!! note "Algorithm Summary"

        1. If `x` is not provided, initialize it with Gaussian noise.
        2. Iteratively update `x` for `n_steps` using `self.langevin_step()`.
        3. Optionally track trajectory (`return_trajectory=True`).
        4. Optionally collect diagnostics such as mean, variance, and energy gradients.

    Args:
        energy_function (BaseEnergyFunction): Energy function to sample from.
        step_size (float): Step size for updates.
        noise_scale (float): Scale of the noise.
        decay (float): Damping coefficient (not supported yet).
        dtype (torch.dtype): Data type to use for the computations.
        device (str): Device to run the computations on (e.g., "cpu" or "cuda").

    Raises:
        ValueError: For invalid parameter ranges

    Methods:
        langevin_step(prev_x, noise): Perform a Langevin step.
        sample_chain(x, dim, n_steps, n_samples, return_trajectory, return_diagnostics): Run the sampling process.
        _setup_diagnostics(dim, n_steps, n_samples): Initialize the diagnostics

    !!! example "Basic Usage"
        ```python
        # Define energy function
        energy_fn = QuadraticEnergy(A=torch.eye(2), b=torch.zeros(2))

        # Initialize sampler
        sampler = LangevinDynamics(
            energy_function=energy_fn,
            step_size=0.01,
            noise_scale=0.1
        )

        # Sample 100 points from 5 parallel chains
        samples = sampler.sample_chain(
            dim=2,
            n_steps=50,
            n_samples=100
        )
        ```
    !!! warning "Parameter Relationships"
        The effective temperature is controlled by:
        $$\text{Temperature} = \frac{\text{noise_scale}^2}{2 \cdot \text{step_size}}$$
        Adjust both parameters together to maintain constant temperature.
    """

    def __init__(
        self,
        energy_function: BaseEnergyFunction,
        step_size: float = 1e-3,
        noise_scale: float = 1.0,
        decay: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(energy_function, dtype, device)

        if step_size <= 0 or noise_scale <= 0:
            raise ValueError("step_size and noise_scale must be positive")
        if not 0 <= decay <= 1:
            raise ValueError("decay must be between 0 and 1")

        if device is not None:
            self.device = torch.device(device)
            energy_function = energy_function.to(self.device)
        else:
            self.device = torch.device("cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.energy_function = energy_function
        self.step_size = step_size
        self.noise_scale = noise_scale
        self.decay = decay

    def langevin_step(self, prev_x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        r"""
        Perform a single Langevin dynamics update step.

        Implements the discrete Langevin equation:

        $$x_{t+1} = x_t - \eta \nabla_x U(x_t) + \sqrt{2\eta} \epsilon_t$$

        Args:
            prev_x (torch.Tensor): Current state tensor of shape (batch_size, dim)
            noise (torch.Tensor): Gaussian noise tensor of shape (batch_size, dim)

        Returns:
            torch.Tensor: Updated state tensor of same shape as prev_x

        Example:
            ```python
            # Single step for 10 particles in 2D space
            current_state = torch.randn(10, 2)
            noise = torch.randn_like(current_state)
            next_state = langevin.langevin_step(current_state, noise)
            ```
        """

        # gradient_fn = partial(self.energy_function.gradient)
        # new_x = (
        #     prev_x
        #     - self.step_size * gradient_fn(prev_x)
        #     + torch.sqrt(torch.tensor(2.0 * self.step_size, device=prev_x.device))
        #     * noise
        # )
        # return new_x

        gradient = self.energy_function.gradient(prev_x)

        # Apply noise scaling
        scaled_noise = self.noise_scale * noise

        # Apply proper step size and noise scaling
        new_x = (
            prev_x
            - self.step_size * gradient
            + torch.sqrt(torch.tensor(2.0 * self.step_size, device=prev_x.device))
            * scaled_noise
        )
        return new_x

    @torch.no_grad()
    def sample_chain(
        self,
        x: Optional[torch.Tensor] = None,
        dim: int = 10,
        n_steps: int = 100,
        n_samples: int = 1,
        thin: int = 1,
        return_trajectory: bool = False,
        return_diagnostics: bool = False,
        *args,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
        """
        Generate Markov chain samples using Langevin dynamics.

        Args:
            x: Initial state to start the sampling from.
            dim: Dimension of the state space.
            n_steps: Number of steps to take between samples.
            n_samples: Number of samples to generate.
            return_trajectory: Whether to return the trajectory of the samples.
            return_diagnostics: Whether to return the diagnostics of the sampling process.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List[dict]]]:
                - If `return_trajectory=False` and `return_diagnostics=False`, returns the final
                  samples of shape `(n_samples, dim)`.
                - If `return_trajectory=True`, returns a tensor of shape `(n_samples, n_steps, dim)`,
                  containing the sampled trajectory.
                - If `return_diagnostics=True`, returns a tuple `(samples, diagnostics)`, where
                  `diagnostics` is a list of dictionaries storing per-step statistics.

        Raises:
            ValueError: If input dimensions mismatch

        Note:
            - Automatically handles device placement (CPU/GPU)
            - Uses mixed-precision training when available
            - Diagnostics include:
                * Mean and variance across dimensions
                * Energy gradients
                * Noise statistics

        Example:
            ```python
            # Generate 100 samples from 5 parallel chains
            samples = sampler.sample_chain(
                dim=32,
                n_steps=500,
                n_samples=100,
                return_diagnostics=True
            )
            ```

        """
        if x is None:
            x = torch.randn(n_samples, dim, dtype=self.dtype, device=self.device)
        else:
            x = x.to(self.device)  # Initial batch

        if return_trajectory:
            trajectory = torch.empty(
                (n_samples, n_steps, dim), dtype=self.dtype, device=self.device
            )

        if return_diagnostics:
            diagnostics = self._setup_diagnostics(dim, n_steps, n_samples=n_samples)

        with torch.amp.autocast(
            device_type="cuda" if self.device.type == "cuda" else "cpu"
        ):
            for i in range(n_steps):
                # Generate fresh noise for each step
                noise = torch.randn_like(x, device=self.device)

                x = self.langevin_step(x, noise)

                if return_trajectory:
                    trajectory[:, i, :] = x

                if return_diagnostics:
                    # Handle mean and variance safely regardless of batch size
                    if n_samples > 1:
                        mean_x = x.mean(dim=0, keepdim=True)
                        var_x = x.var(dim=0, unbiased=False, keepdim=True)
                        var_x = torch.clamp(var_x, min=1e-10, max=1e10)
                    else:
                        # For single sample, just use the value and zeros for variance
                        mean_x = x.clone()
                        var_x = torch.zeros_like(x)

                    # Compute energy values
                    energy = self.energy_function(x)

                    # Store the diagnostics safely
                    for b in range(n_samples):
                        diagnostics[i, 0, b, :] = mean_x[b if n_samples > 1 else 0]
                        diagnostics[i, 1, b, :] = var_x[b if n_samples > 1 else 0]
                        diagnostics[i, 2, b, :] = energy[b].reshape(-1)

        if return_trajectory:
            if return_diagnostics:
                return trajectory, diagnostics
            return trajectory
        if return_diagnostics:
            return x, diagnostics
        return x

    def _setup_diagnostics(
        self, dim: int, n_steps: int, n_samples: int = None
    ) -> torch.Tensor:
        if n_samples is not None:
            return torch.empty(
                (n_steps, 3, n_samples, dim), device=self.device, dtype=self.dtype
            )
        else:
            return torch.empty((n_steps, 3, dim), device=self.device, dtype=self.dtype)
