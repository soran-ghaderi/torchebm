---
title: Core Components
description: Detailed explanation of TorchEBM's core components and their implementation
icon: material/function
---

# Core Components

!!! abstract "Building Blocks"
    TorchEBM is built around several core components that form the foundation of the library. This guide provides in-depth information about these components and how they interact.

## Component Overview

<div class="grid cards" markdown>

-   :material-function-variant:{ .lg .middle } __Energy Functions__

    ---

    Define the energy landscape for probability distributions.

    ```python
    energy = energy_fn(x)  # Evaluate energy at point x
    ```

-   :material-chart-scatter-plot:{ .lg .middle } __Samplers__

    ---

    Generate samples from energy-based distributions.

    ```python
    samples = sampler.sample(n_samples=1000)  # Generate 1000 samples
    ```

-   :material-abacus:{ .lg .middle } __Loss Functions__

    ---

    Train energy-based models from data.

    ```python
    loss = loss_fn(model, data_samples)  # Compute training loss
    ```

-   :material-neural-network:{ .lg .middle } __Models__

    ---

    Parameterize energy functions with neural networks.

    ```python
    model = EnergyModel(network=nn.Sequential(...))
    ```

</div>

## Energy Functions

Energy functions are the core building block of TorchEBM. They define a scalar energy value for each point in the sample space.

### Base Energy Function

The `BaseEnergyFunction` class is the foundation for all energy functions:

```python
class BaseEnergyFunction(nn.Module):
    """Base class for all energy functions.
    
    An energy function maps points in the sample space to scalar energy values.
    Lower energy corresponds to higher probability density.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy for input points.
        
        Args:
            x: Input tensor of shape (batch_size, dim)
            
        Returns:
            Tensor of shape (batch_size,) containing energy values
        """
        raise NotImplementedError
        
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute score function (gradient of energy) for input points.
        
        Args:
            x: Input tensor of shape (batch_size, dim)
            
        Returns:
            Tensor of shape (batch_size, dim) containing score values
        """
        x = x.requires_grad_(True)
        energy = self.forward(x)
        return torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
```

### Analytical Energy Functions

TorchEBM provides several analytical energy functions for testing and benchmarking:

=== "Gaussian Energy"

    ```python
    class GaussianEnergy(BaseEnergyFunction):
        """Gaussian energy function.
        
        Energy function defined by a multivariate Gaussian distribution:
        E(x) = 0.5 * (x - mean)^T * precision * (x - mean)
        """
        
        def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
            """Initialize Gaussian energy function.
            
            Args:
                mean: Mean vector of shape (dim,)
                cov: Covariance matrix of shape (dim, dim)
            """
            super().__init__()
            self.register_buffer("mean", mean)
            self.register_buffer("cov", cov)
            self.register_buffer("precision", torch.inverse(cov))
            self._dim = mean.size(0)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Compute Gaussian energy.
            
            Args:
                x: Input tensor of shape (batch_size, dim)
                
            Returns:
                Tensor of shape (batch_size,) containing energy values
            """
            centered = x - self.mean
            return 0.5 * torch.sum(
                centered * (self.precision @ centered.T).T,
                dim=1
            )
    ```

=== "Double Well Energy"

    ```python
    class DoubleWellEnergy(BaseEnergyFunction):
        """Double well energy function.
        
        Energy function with two local minima:
        E(x) = a * (x^2 - b)^2
        """
        
        def __init__(self, a: float = 1.0, b: float = 2.0):
            """Initialize double well energy function.
            
            Args:
                a: Scale parameter
                b: Parameter controlling the distance between wells
            """
            super().__init__()
            self.a = a
            self.b = b
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Compute double well energy.
            
            Args:
                x: Input tensor of shape (batch_size, dim)
                
            Returns:
                Tensor of shape (batch_size,) containing energy values
            """
            return self.a * torch.sum((x**2 - self.b)**2, dim=1)
    ```

### Composite Energy Functions

Energy functions can be composed to create more complex landscapes:

```python
class CompositeEnergy(BaseEnergyFunction):
    """Composite energy function.
    
    Combines multiple energy functions through addition.
    """
    
    def __init__(self, energy_functions: List[BaseEnergyFunction], weights: Optional[List[float]] = None):
        """Initialize composite energy function.
        
        Args:
            energy_functions: List of energy functions to combine
            weights: Optional weights for each energy function
        """
        super().__init__()
        self.energy_functions = nn.ModuleList(energy_functions)
        if weights is None:
            weights = [1.0] * len(energy_functions)
        self.weights = weights
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute composite energy.
        
        Args:
            x: Input tensor of shape (batch_size, dim)
            
        Returns:
            Tensor of shape (batch_size,) containing energy values
        """
        return sum(w * f(x) for w, f in zip(self.weights, self.energy_functions))
```

## Samplers

Samplers generate samples from energy-based distributions. They provide methods to initialize and update samples based on the energy landscape.

### Base Sampler

The `Sampler` class is the foundation for all sampling algorithms:

```python
class Sampler(ABC):
    """Base class for all samplers.
    
    A sampler generates samples from an energy-based distribution.
    """
    
    def __init__(self, energy_function: BaseEnergyFunction):
        """Initialize sampler.
        
        Args:
            energy_function: Energy function to sample from
        """
        self.energy_function = energy_function
        
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

### Langevin Dynamics

The `LangevinDynamics` sampler implements Langevin Monte Carlo:

```python
class LangevinDynamics(Sampler):
    """Langevin dynamics sampler.
    
    Uses Langevin dynamics to sample from an energy-based distribution.
    """
    
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
        # Compute score (gradient of energy)
        score = self.energy_function.score(x)
        
        # Update samples
        noise = torch.randn_like(x) * np.sqrt(2 * self.step_size * self.noise_scale)
        x_new = x - self.step_size * score + noise
        
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
            Tensor of shape (n_samples, dim) containing final samples,
            or a tuple of (samples, trajectory) if return_trajectory is True
        """
        # Initialize samples
        if initial_samples is None:
            x = torch.randn(n_samples, dim)
        else:
            x = initial_samples.clone()
            
        # Initialize trajectory if needed
        if return_trajectory:
            trajectory = torch.zeros(n_steps + 1, n_samples, dim)
            trajectory[0] = x
            
        # Run chain
        for i in range(n_steps):
            x = self.sample_step(x)
            if return_trajectory:
                trajectory[i + 1] = x
                
        if return_trajectory:
            return x, trajectory
        else:
            return x
            
    def sample(self, n_samples: int, dim: int, n_steps: int = 100, **kwargs) -> torch.Tensor:
        """Generate samples from the energy-based distribution.
        
        Args:
            n_samples: Number of samples to generate
            dim: Dimensionality of samples
            n_steps: Number of steps in the chain
            **kwargs: Additional parameters passed to sample_chain
            
        Returns:
            Tensor of shape (n_samples, dim) containing samples
        """
        return self.sample_chain(dim=dim, n_steps=n_steps, n_samples=n_samples, **kwargs)
```

## BaseLoss Functions

BaseLoss functions are used to train energy-based models from data. They provide methods to compute gradients for model updates.

### Base BaseLoss Function

The `BaseLoss` class is the foundation for all loss functions:

```python
class BaseLoss(ABC):
    """Base class for all loss functions.
    
    A loss function computes a loss value for an energy-based model.
    """
    
    @abstractmethod
    def __call__(
        self,
        model: nn.Module,
        data_samples: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute loss for the model.
        
        Args:
            model: Energy-based model
            data_samples: Samples from the target distribution
            **kwargs: Additional loss-specific parameters
            
        Returns:
            Scalar loss value
        """
        pass
```

### Contrastive Divergence

The `ContrastiveDivergence` loss implements the contrastive divergence algorithm:

```python
class ContrastiveDivergence(BaseLoss):
    """Contrastive divergence loss.
    
    Uses contrastive divergence to train energy-based models.
    """
    
    def __init__(
        self,
        sampler: Sampler,
        k: int = 1,
        batch_size: Optional[int] = None
    ):
        """Initialize contrastive divergence loss.
        
        Args:
            sampler: Sampler to generate model samples
            k: Number of sampling steps (CD-n_steps)
            batch_size: Optional batch size for sampling
        """
        super().__init__()
        self.sampler = sampler
        self.k = k
        self.batch_size = batch_size
        
    def __call__(
        self,
        model: nn.Module,
        data_samples: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute contrastive divergence loss.
        
        Args:
            model: Energy-based model
            data_samples: Samples from the target distribution
            **kwargs: Additional parameters passed to the sampler
            
        Returns:
            Scalar loss value
        """
        # Get data statistics
        batch_size = self.batch_size or data_samples.size(0)
        dim = data_samples.size(1)
        
        # Set the model as the sampler's energy function
        self.sampler.energy_function = model
        
        # Generate model samples
        model_samples = self.sampler.sample_chain(
            dim=dim,
            n_steps=self.k,
            n_samples=batch_size,
            **kwargs
        )
        
        # Compute energies
        data_energy = model(data_samples).mean()
        model_energy = model(model_samples).mean()
        
        # Compute loss
        loss = data_energy - model_energy
        
        return loss
```

## Models

Models parameterize energy functions using neural networks.

### Energy Model

The `EnergyModel` class wraps a neural network as an energy function:

```python
class EnergyModel(BaseEnergyFunction):
    """Neural network-based energy model.
    
    Uses a neural network to parameterize an energy function.
    """
    
    def __init__(self, network: nn.Module):
        """Initialize energy model.
        
        Args:
            network: Neural network that outputs scalar energy values
        """
        super().__init__()
        self.network = network
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy using the neural network.
        
        Args:
            x: Input tensor of shape (batch_size, dim)
            
        Returns:
            Tensor of shape (batch_size,) containing energy values
        """
        return self.network(x).squeeze(-1)
```

## Component Interactions

The following diagram illustrates how the core components interact:

```mermaid
graph TD
    A[Energy Function] -->|Defines landscape| B[Sampler]
    B -->|Generates samples| C[Training Process]
    D[BaseLoss Function] -->|Guides training| C
    C -->|Updates| E[Energy Model]
    E -->|Parameterizes| A
```

### Typical Usage Flow

1. **Define an energy function** - Either analytical or neural network-based
2. **Create a sampler** - Using the energy function
3. **Generate samples** - Using the sampler
4. **Train a model** - Using the loss function and sampler
5. **Use the trained model** - For tasks like generation or density estimation

```python
# Define energy function
energy_fn = GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2))

# Create sampler
sampler = LangevinDynamics(energy_function=energy_fn, step_size=0.01)

# Generate samples
samples = sampler.sample_chain(dim=2, n_steps=1000, n_samples=100)

# Create and train a model
model = EnergyModel(network=MLP(input_dim=2, hidden_dims=[32, 32], output_dim=1))
loss_fn = ContrastiveDivergence(sampler=sampler, k=10)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(100):
    optimizer.zero_grad()
    loss = loss_fn(model, data_samples)
    loss.backward()
    optimizer.step()
```

## Extension Points

TorchEBM is designed to be extensible at several points:

* **New Energy Functions** - Create by subclassing `BaseEnergyFunction`
* **New Samplers** - Create by subclassing `Sampler`
* **New BaseLoss Functions** - Create by subclassing `BaseLoss`
* **New Models** - Create by subclassing `EnergyModel` or using custom networks

## Component Lifecycle

Each component in TorchEBM has a typical lifecycle:

1. **Initialization** - Configure the component with parameters
2. **Usage** - Use the component to perform its intended function
3. **Composition** - Combine with other components
4. **Extension** - Extend with new functionality

Understanding this lifecycle helps when implementing new components or extending existing ones.

## Best Practices

When working with TorchEBM components, follow these best practices:

* **Energy Functions**: Ensure they're properly normalized for stable training
* **Samplers**: Check mixing time and adjust parameters accordingly
* **BaseLoss Functions**: Monitor training stability and adjust hyperparameters
* **Models**: Use appropriate architecture for the problem domain

!!! tip "Performance Optimization"
    For large-scale applications, consider using CUDA-optimized implementations and batch processing for better performance.

<div class="grid cards" markdown>

-   :material-code-json:{ .lg .middle } __Energy Functions__

    ---

    Learn about energy function implementation details.

    [:octicons-arrow-right-24: Energy Functions](implementation_energy.md)

-   :material-access-point:{ .lg .middle } __Samplers__

    ---

    Explore sampler implementation details.

    [:octicons-arrow-right-24: Samplers](implementation_samplers.md)

-   :material-function-variant:{ .lg .middle } __Loss Functions__

    ---

    Understand loss function implementation details.

    [:octicons-arrow-right-24: BaseLoss Functions](implementation_losses.md)

</div> 