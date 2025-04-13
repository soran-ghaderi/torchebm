---
title: Samplers Implementation
description: Detailed implementation details of TorchEBM's samplers
icon: material/access-point
---

# Samplers Implementation

!!! abstract "Implementation Details"
    This guide provides detailed information about the implementation of sampling algorithms in TorchEBM, including mathematical foundations, code structure, and optimization techniques.

## Mathematical Foundation

Sampling algorithms in energy-based models aim to generate samples from the distribution:

$$p(x) = \frac{e^{-E(x)}}{Z}$$

where $E(x)$ is the energy function and $Z = \int e^{-E(x)} dx$ is the normalization constant.

## Base Sampler Implementation

The `Sampler` base class provides the foundation for all sampling algorithms:

```python
from abc import ABC, abstractmethod
import torch
from typing import Optional, Union, Tuple

from torchebm.core import BaseEnergyFunction


class Sampler(ABC):
    """Base class for all sampling algorithms."""

    def __init__(self, energy_function: BaseEnergyFunction):
        """Initialize sampler with an energy function.
        
        Args:
            energy_function: The energy function to sample from
        """
        self.energy_function = energy_function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to(self, device):
        """Move sampler to specified device."""
        self.device = device
        return self

    @abstractmethod
    def sample(self, n_samples: int, **kwargs) -> torch.Tensor:
        """Generate samples from the energy-based distribution.
        
        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional sampler-specific parameters
            
        Returns:
            Tensor of shape (n_samples, dim) containing samples
        """
        pass

    @abstractmethod
    def sample_chain(self, dim: int, n_steps: int, n_samples: int = 1, **kwargs) -> torch.Tensor:
        """Generate samples using a Markov chain.
        
        Args:
            dim: Dimensionality of samples
            n_steps: Number of steps in the chain
            n_samples: Number of parallel chains to run
            **kwargs: Additional sampler-specific parameters
            
        Returns:
            Tensor of shape (n_samples, dim) containing final samples
        """
        pass
```

## Langevin Dynamics

### Mathematical Background

Langevin dynamics uses the score function (gradient of log-probability) to guide sampling with Brownian motion:

$$dx_t = -\nabla E(x_t)dt + \sqrt{2}dW_t$$

where $W_t$ is the Wiener process (Brownian motion).

### Implementation

```python
import torch
import numpy as np
from typing import Optional, Union, Tuple

from torchebm.core import BaseEnergyFunction
from torchebm.samplers.base import Sampler


class LangevinDynamics(Sampler):
    """Langevin dynamics sampler."""

    def __init__(
            self,
            energy_function: BaseEnergyFunction,
            step_size: float = 0.01,
            noise_scale: float = 1.0
    ):
        """Initialize Langevin dynamics sampler.
        
        Args:
            energy_function: Energy function to sample from
            step_size: Step size for updates
            noise_scale: Scale of noise added at each step
        """
        super().__init__(energy_function)
        self.step_size = step_size
        self.noise_scale = noise_scale

    def sample_step(self, x: torch.Tensor) -> torch.Tensor:
        """Perform one step of Langevin dynamics.
        
        Args:
            x: Current samples of shape (n_samples, dim)
            
        Returns:
            Updated samples of shape (n_samples, dim)
        """
        # Compute score (gradient of log probability)
        score = -self.energy_function.score(x)

        # Add drift term and noise
        noise = torch.randn_like(x) * np.sqrt(2 * self.step_size * self.noise_scale)
        x_new = x + self.step_size * score + noise

        return x_new

    def sample_chain(
            self,
            dim: int,
            n_steps: int,
            n_samples: int = 1,
            initial_samples: Optional[torch.Tensor] = None,
            return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples using a Langevin dynamics chain.
        
        Args:
            dim: Dimensionality of samples
            n_steps: Number of steps in the chain
            n_samples: Number of parallel chains to run
            initial_samples: Optional initial samples
            return_trajectory: Whether to return the full trajectory
            
        Returns:
            Samples or (samples, trajectory)
        """
        # Initialize samples
        if initial_samples is None:
            x = torch.randn(n_samples, dim, device=self.device)
        else:
            x = initial_samples.clone().to(self.device)

        # Initialize trajectory if needed
        if return_trajectory:
            trajectory = torch.zeros(n_steps + 1, n_samples, dim, device=self.device)
            trajectory[0] = x

        # Run sampling chain
        for i in range(n_steps):
            x = self.sample_step(x)
            if return_trajectory:
                trajectory[i + 1] = x

        if return_trajectory:
            return x, trajectory
        else:
            return x

    def sample(self, n_samples: int, dim: int, n_steps: int = 100, **kwargs) -> torch.Tensor:
        """Generate samples from the energy-based distribution."""
        return self.sample_chain(dim=dim, n_steps=n_steps, n_samples=n_samples, **kwargs)
```

## Hamiltonian Monte Carlo

### Mathematical Background

Hamiltonian Monte Carlo (HMC) introduces momentum variables to help explore the distribution more efficiently:

$$H(x, p) = E(x) + \frac{1}{2}p^Tp$$

where $p$ is the momentum variable and $H$ is the Hamiltonian.

### Implementation

```python
class HamiltonianMonteCarlo(Sampler):
    """Hamiltonian Monte Carlo sampler."""
    
    def __init__(
        self,
        energy_function: BaseEnergyFunction,
        step_size: float = 0.1,
        n_leapfrog_steps: int = 10,
        mass_matrix: Optional[torch.Tensor] = None
    ):
        """Initialize HMC sampler.
        
        Args:
            energy_function: Energy function to sample from
            step_size: Step size for leapfrog integration
            n_leapfrog_steps: Number of leapfrog steps
            mass_matrix: Mass matrix for momentum (identity by default)
        """
        super().__init__(energy_function)
        self.step_size = step_size
        self.n_leapfrog_steps = n_leapfrog_steps
        self.mass_matrix = mass_matrix
    
    def _leapfrog_step(self, x: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one leapfrog step.
        
        Args:
            x: Position tensor of shape (n_samples, dim)
            p: Momentum tensor of shape (n_samples, dim)
            
        Returns:
            New position and momentum
        """
        # Half step for momentum
        grad_x = self.energy_function.score(x)
        p = p - 0.5 * self.step_size * grad_x
        
        # Full step for position
        if self.mass_matrix is not None:
            x = x + self.step_size * torch.matmul(p, self.mass_matrix)
        else:
            x = x + self.step_size * p
        
        # Half step for momentum
        grad_x = self.energy_function.score(x)
        p = p - 0.5 * self.step_size * grad_x
        
        return x, p
    
    def _compute_hamiltonian(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute the Hamiltonian value.
        
        Args:
            x: Position tensor of shape (n_samples, dim)
            p: Momentum tensor of shape (n_samples, dim)
            
        Returns:
            Hamiltonian value of shape (n_samples,)
        """
        energy = self.energy_function(x)
        
        if self.mass_matrix is not None:
            kinetic = 0.5 * torch.sum(p * torch.matmul(p, self.mass_matrix), dim=1)
        else:
            kinetic = 0.5 * torch.sum(p * p, dim=1)
        
        return energy + kinetic
    
    def sample_step(self, x: torch.Tensor) -> torch.Tensor:
        """Perform one step of HMC.
        
        Args:
            x: Current samples of shape (n_samples, dim)
            
        Returns:
            Updated samples of shape (n_samples, dim)
        """
        # Sample initial momentum
        p = torch.randn_like(x)
        
        # Compute initial Hamiltonian
        x_old, p_old = x.clone(), p.clone()
        h_old = self._compute_hamiltonian(x_old, p_old)
        
        # Leapfrog integration
        x_new, p_new = x_old.clone(), p_old.clone()
        for _ in range(self.n_leapfrog_steps):
            x_new, p_new = self._leapfrog_step(x_new, p_new)
        
        # Metropolis-Hastings correction
        h_new = self._compute_hamiltonian(x_new, p_new)
        accept_prob = torch.exp(h_old - h_new)
        accept = torch.rand_like(accept_prob) < accept_prob
        
        # Accept or reject
        x_out = torch.where(accept.unsqueeze(1), x_new, x_old)
        
        return x_out
    
    def sample_chain(self, dim: int, n_steps: int, n_samples: int = 1, **kwargs) -> torch.Tensor:
        """Generate samples using an HMC chain."""
        # Implementation similar to LangevinDynamics.sample_chain
        pass
    
    def sample(self, n_samples: int, dim: int, n_steps: int = 100, **kwargs) -> torch.Tensor:
        """Generate samples from the energy-based distribution."""
        return self.sample_chain(dim=dim, n_steps=n_steps, n_samples=n_samples, **kwargs)
```

## Metropolis-Hastings Sampler

```python
class MetropolisHastings(Sampler):
    """Metropolis-Hastings sampler."""
    
    def __init__(
        self,
        energy_function: BaseEnergyFunction,
        proposal_scale: float = 0.1
    ):
        """Initialize Metropolis-Hastings sampler.
        
        Args:
            energy_function: Energy function to sample from
            proposal_scale: Scale of proposal distribution
        """
        super().__init__(energy_function)
        self.proposal_scale = proposal_scale
    
    def sample_step(self, x: torch.Tensor) -> torch.Tensor:
        """Perform one step of Metropolis-Hastings.
        
        Args:
            x: Current samples of shape (n_samples, dim)
            
        Returns:
            Updated samples of shape (n_samples, dim)
        """
        # Compute energy of current state
        energy_x = self.energy_function(x)
        
        # Propose new state
        proposal = x + self.proposal_scale * torch.randn_like(x)
        
        # Compute energy of proposed state
        energy_proposal = self.energy_function(proposal)
        
        # Compute acceptance probability
        accept_prob = torch.exp(energy_x - energy_proposal)
        accept = torch.rand_like(accept_prob) < accept_prob
        
        # Accept or reject
        x_new = torch.where(accept.unsqueeze(1), proposal, x)
        
        return x_new
    
    def sample_chain(self, dim: int, n_steps: int, n_samples: int = 1, **kwargs) -> torch.Tensor:
        """Generate samples using a Metropolis-Hastings chain."""
        # Implementation similar to LangevinDynamics.sample_chain
        pass
```

## Performance Optimizations

### CUDA Acceleration

For performance-critical operations, we implement CUDA-optimized versions:

```python
from torchebm.cuda import langevin_step_cuda

class CUDALangevinDynamics(LangevinDynamics):
    """CUDA-optimized Langevin dynamics sampler."""
    
    def sample_step(self, x: torch.Tensor) -> torch.Tensor:
        """Perform one step of Langevin dynamics with CUDA optimization."""
        if not torch.cuda.is_available() or not x.is_cuda:
            return super().sample_step(x)
        
        return langevin_step_cuda(
            x, 
            self.energy_function,
            self.step_size,
            self.noise_scale
        )
```

### Batch Processing

To handle large numbers of samples efficiently:

```python
def batch_sample_chain(
    sampler: Sampler,
    dim: int,
    n_steps: int,
    n_samples: int,
    batch_size: int = 1000
) -> torch.Tensor:
    """Sample in batches to avoid memory issues."""
    samples = []
    
    for i in range(0, n_samples, batch_size):
        batch_n = min(batch_size, n_samples - i)
        batch_samples = sampler.sample_chain(
            dim=dim,
            n_steps=n_steps,
            n_samples=batch_n
        )
        samples.append(batch_samples)
    
    return torch.cat(samples, dim=0)
```

## Best Practices for Custom Samplers

When implementing custom samplers, follow these best practices:

<div class="grid" markdown>

<div markdown>
### Do

* Subclass the `Sampler` base class
* Implement both `sample` and `sample_chain` methods
* Handle device placement correctly
* Support batched execution
* Add diagnostics when appropriate
</div>

<div markdown>
### Don't

* Modify input tensors in-place
* Allocate new tensors unnecessarily
* Ignore numerical stability
* Forget to validate inputs
* Implement complex logic in sampling loops
</div>

</div>

!!! example "Custom Sampler Example"
    ```python
    class CustomSampler(Sampler):
        """Custom sampler example."""
        
        def __init__(self, energy_function, step_size=0.01):
            super().__init__(energy_function)
            self.step_size = step_size
            
        def sample_step(self, x):
            # Custom sampling logic
            return x + self.step_size * torch.randn_like(x)
            
        def sample_chain(self, dim, n_steps, n_samples=1):
            # Initialize
            x = torch.randn(n_samples, dim, device=self.device)
            
            # Run chain
            for _ in range(n_steps):
                x = self.sample_step(x)
                
            return x
            
        def sample(self, n_samples, dim, n_steps=100):
            return self.sample_chain(dim, n_steps, n_samples)
    ```

## Resources

<div class="grid cards" markdown>

-   :material-function:{ .lg .middle } __Core Components__

    ---

    Learn about core components and their interactions.

    [:octicons-arrow-right-24: Core Components](core_components.md)

-   :material-code-json:{ .lg .middle } __Energy Functions__

    ---

    Explore energy function implementation details.

    [:octicons-arrow-right-24: Energy Functions](implementation_energy.md)

-   :material-function-variant:{ .lg .middle } __Loss Functions__

    ---

    Understand loss function implementation details.

    [:octicons-arrow-right-24: BaseLoss Functions](implementation_losses.md)

</div> 