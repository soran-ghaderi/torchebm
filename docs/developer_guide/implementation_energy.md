---
title: Energy Functions Implementation
description: Detailed implementation details of TorchEBM's energy functions
icon: material/code-json
---

# Energy Functions Implementation

!!! abstract "Implementation Details"
    This guide provides detailed information about the implementation of energy functions in TorchEBM, including mathematical foundations, code structure, and optimization techniques.

## Mathematical Foundation

Energy-based models define a probability distribution through an energy function:

$$p(x) = \frac{e^{-E(x)}}{Z}$$

where $E(x)$ is the energy function and $Z = \int e^{-E(x)} dx$ is the normalization constant (partition function).

The score function is the gradient of the log-probability:

$$\nabla_x \log p(x) = -\nabla_x E(x)$$

This relationship is fundamental to many sampling and training methods in TorchEBM.

## Base Energy Function Implementation

The `BaseEnergyFunction` base class provides the foundation for all energy functions:

```python
class BaseEnergyFunction(nn.Module):
    """Base class for all energy functions."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy for input x."""
        raise NotImplementedError
        
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute score (gradient of energy) for input x."""
        x = x.requires_grad_(True)
        energy = self.forward(x)
        return torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
```

Key design decisions:

1. **PyTorch `nn.Module` Base**: Allows energy functions to have learnable parameters and use PyTorch's optimization tools
2. **Automatic Differentiation**: Uses PyTorch's autograd for computing the score function
3. **Batched Computation**: All methods support batched inputs for efficiency

## Analytical Energy Functions

TorchEBM includes several analytical energy functions for testing and benchmarking. Here are detailed implementations of some key ones:

### Gaussian Energy

The Gaussian energy function is defined as:

$$E(x) = \frac{1}{2}(x - \mu)^T\Sigma^{-1}(x - \mu)$$

Where $\mu$ is the mean vector and $\Sigma$ is the covariance matrix.

```python
class GaussianEnergy(BaseEnergyFunction):
    """Gaussian energy function."""
    
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
        
        # Compute log determinant for normalization (optional)
        self.register_buffer("log_det", torch.logdet(cov))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian energy.
        
        Args:
            x: Input tensor of shape (batch_size, dim)
            
        Returns:
            Tensor of shape (batch_size,) containing energy values
        """
        # Ensure x has the right shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Center the data
        centered = x - self.mean
        
        # Compute quadratic form efficiently
        return 0.5 * torch.sum(
            centered * torch.matmul(centered, self.precision),
            dim=1
        )
        
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute score function analytically.
        
        This is more efficient than using automatic differentiation.
        
        Args:
            x: Input tensor of shape (batch_size, dim)
            
        Returns:
            Tensor of shape (batch_size, dim) containing score values
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        return -torch.matmul(x - self.mean, self.precision)
```

Implementation notes:

* We precompute the precision matrix (inverse covariance) for efficiency
* A specialized `score` method is provided that uses the analytical formula rather than automatic differentiation
* Input shape handling ensures both single samples and batches work correctly

### Double Well Energy

The double well energy function creates a bimodal distribution:

$$E(x) = a(x^2 - b)^2$$

```python
class DoubleWellEnergy(BaseEnergyFunction):
    """Double well energy function."""
    
    def __init__(self, a: float = 1.0, b: float = 2.0):
        """Initialize double well energy function.
        
        Args:
            a: Scale parameter controlling depth of wells
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
        # Compute (x^2 - b)^2 for each dimension, then sum
        return self.a * torch.sum((x**2 - self.b)**2, dim=1)
```

### Rosenbrock Energy

The Rosenbrock function is a challenging test case with a narrow curved valley:

$$E(x) = \sum_{i=1}^{d-1} \left[ a(x_{i+1} - x_i^2)^2 + (x_i - 1)^2 \right]$$

```python
class RosenbrockEnergy(BaseEnergyFunction):
    """Rosenbrock energy function."""
    
    def __init__(self, a: float = 1.0, b: float = 100.0):
        """Initialize Rosenbrock energy function.
        
        Args:
            a: Scale parameter for the first term
            b: Scale parameter for the second term (usually 100)
        """
        super().__init__()
        self.a = a
        self.b = b
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Rosenbrock energy.
        
        Args:
            x: Input tensor of shape (batch_size, dim)
            
        Returns:
            Tensor of shape (batch_size,) containing energy values
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        batch_size, dim = x.shape
        energy = torch.zeros(batch_size, device=x.device)
        
        for i in range(dim - 1):
            term1 = self.b * (x[:, i+1] - x[:, i]**2)**2
            term2 = (x[:, i] - 1)**2
            energy += term1 + term2
            
        return energy
```

## Composite Energy Functions

TorchEBM supports composing energy functions to create more complex landscapes:

```python
class CompositeEnergy(BaseEnergyFunction):
    """Composite energy function."""
    
    def __init__(
        self,
        energy_functions: List[BaseEnergyFunction],
        weights: Optional[List[float]] = None,
        operation: str = "sum"
    ):
        """Initialize composite energy function.
        
        Args:
            energy_functions: List of energy functions to combine
            weights: Optional weights for each energy function
            operation: How to combine energy functions ("sum", "product", "min", "max")
        """
        super().__init__()
        self.energy_functions = nn.ModuleList(energy_functions)
        
        if weights is None:
            weights = [1.0] * len(energy_functions)
        self.register_buffer("weights", torch.tensor(weights))
        
        if operation not in ["sum", "product", "min", "max"]:
            raise ValueError(f"Unknown operation: {operation}")
        self.operation = operation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute composite energy.
        
        Args:
            x: Input tensor of shape (batch_size, dim)
            
        Returns:
            Tensor of shape (batch_size,) containing energy values
        """
        energies = [f(x) * w for f, w in zip(self.energy_functions, self.weights)]
        
        if self.operation == "sum":
            return torch.sum(torch.stack(energies), dim=0)
        elif self.operation == "product":
            return torch.prod(torch.stack(energies), dim=0)
        elif self.operation == "min":
            return torch.min(torch.stack(energies), dim=0)[0]
        elif self.operation == "max":
            return torch.max(torch.stack(energies), dim=0)[0]
```

## Neural Network Energy Functions

Neural networks can parameterize energy functions for flexibility:

```python
class MLPEnergy(BaseEnergyFunction):
    """Multi-layer perceptron energy function."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: Callable = nn.SiLU
    ):
        """Initialize MLP energy function.
        
        Args:
            input_dim: Input dimensionality
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
        """
        super().__init__()
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
            
        # Final layer with scalar output
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy using the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size,) containing energy values
        """
        return self.network(x).squeeze(-1)
```

## Performance Optimizations

### Efficient Gradient Computation

For gradients, TorchEBM provides optimized implementations:

```python
def efficient_grad(energy_fn: BaseEnergyFunction, x: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
    """Compute gradient of energy function efficiently.
    
    Args:
        energy_fn: Energy function
        x: Input tensor of shape (batch_size, dim)
        create_graph: Whether to create gradient graph (for higher-order gradients)
        
    Returns:
        Gradient tensor of shape (batch_size, dim)
    """
    x.requires_grad_(True)
    with torch.enable_grad():
        energy = energy_fn(x)
        
    grad = torch.autograd.grad(
        energy.sum(), x, create_graph=create_graph
    )[0]
    
    return grad
```

### CUDA Implementations

For performance-critical operations, TorchEBM includes CUDA implementations:

```python
def cuda_score_function(energy_fn, x):
    """CUDA-optimized score function computation."""
    # Use energy_fn's custom CUDA implementation if available
    if hasattr(energy_fn, 'cuda_score') and torch.cuda.is_available():
        return energy_fn.cuda_score(x)
    else:
        # Fall back to autograd
        return energy_fn.score(x)
```

## Factory Methods

Factory methods provide convenient ways to create energy functions:

```python
@classmethod
def create_standard_gaussian(cls, dim: int) -> 'GaussianEnergy':
    """Create a standard Gaussian energy function.
    
    Args:
        dim: Dimensionality
        
    Returns:
        GaussianEnergy with zero mean and identity covariance
    """
    return cls(mean=torch.zeros(dim), cov=torch.eye(dim))
    
@classmethod
def from_samples(cls, samples: torch.Tensor, regularization: float = 1e-4) -> 'GaussianEnergy':
    """Create a Gaussian energy function from data samples.
    
    Args:
        samples: Data samples of shape (n_samples, dim)
        regularization: Small value added to diagonal for numerical stability
        
    Returns:
        GaussianEnergy fit to the samples
    """
    mean = samples.mean(dim=0)
    cov = torch.cov(samples.T) + regularization * torch.eye(samples.size(1))
    return cls(mean=mean, cov=cov)
```

## Implementation Challenges and Solutions

### Numerical Stability

Energy functions must be numerically stable:

```python
class NumericallyStableEnergy(BaseEnergyFunction):
    """Energy function with numerical stability considerations."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy with numerical stability.
        
        Uses log-sum-exp trick for numerical stability.
        """
        # Example of numerical stability in computation
        terms = self.compute_terms(x)
        max_term = torch.max(terms, dim=1, keepdim=True)[0]
        stable_energy = max_term + torch.log(torch.sum(
            torch.exp(terms - max_term), dim=1
        ))
        return stable_energy
```

### Handling Multi-Modal Distributions

For multi-modal distributions:

```python
class MixtureEnergy(BaseEnergyFunction):
    """Mixture of energy functions."""
    
    def __init__(self, components: List[BaseEnergyFunction], weights: Optional[List[float]] = None):
        """Initialize mixture energy function.
        
        Args:
            components: List of component energy functions
            weights: Optional weights for each component
        """
        super().__init__()
        self.components = nn.ModuleList(components)
        
        if weights is None:
            weights = [1.0] * len(components)
        self.register_buffer("log_weights", torch.log(torch.tensor(weights)))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mixture energy using log-sum-exp for stability."""
        energies = torch.stack([f(x) for f in self.components], dim=1)
        weighted_energies = -self.log_weights - energies
        
        # Use log-sum-exp trick for numerical stability
        max_val = torch.max(weighted_energies, dim=1, keepdim=True)[0]
        stable_energy = -max_val - torch.log(torch.sum(
            torch.exp(weighted_energies - max_val), dim=1
        ))
        
        return stable_energy
```

## Testing Energy Functions

TorchEBM includes comprehensive testing utilities for energy functions:

```python
def test_energy_function(energy_fn: BaseEnergyFunction, dim: int, n_samples: int = 1000) -> dict:
    """Test an energy function for correctness and properties.
    
    Args:
        energy_fn: Energy function to test
        dim: Input dimensionality
        n_samples: Number of test samples
        
    Returns:
        Dictionary with test results
    """
    # Generate random samples
    x = torch.randn(n_samples, dim)
    
    # Test energy computation
    energy = energy_fn(x)
    assert energy.shape == (n_samples,)
    
    # Test score computation
    score = energy_fn.score(x)
    assert score.shape == (n_samples, dim)
    
    # Test gradient consistency
    manual_grad = torch.autograd.grad(
        energy_fn(x).sum(), x, create_graph=True
    )[0]
    assert torch.allclose(score, -manual_grad, atol=1e-5, rtol=1e-5)
    
    return {
        "energy_mean": energy.mean().item(),
        "energy_std": energy.std().item(),
        "score_mean": score.mean().item(),
        "score_std": score.std().item(),
    }
```

## Best Practices for Custom Energy Functions

When implementing custom energy functions, follow these best practices:

<div class="grid" markdown>

<div markdown>
### Do

* Implement a custom `score` method if an analytical gradient is available
* Use vectorized operations for performance
* Register parameters and buffers properly
* Handle batched inputs consistently
* Add factory methods for common use cases
</div>

<div markdown>
### Don't

* Use loops when vectorized operations are possible
* Recompute values that could be cached
* Modify inputs in-place
* Forget to handle edge cases
* Ignore numerical stability
</div>

</div>

!!! example "Custom Energy Function Example"
    ```python
    class CustomEnergy(BaseEnergyFunction):
        """Custom energy function example."""
        
        def __init__(self, scale: float = 1.0):
            super().__init__()
            self.scale = scale
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Ensure correct input shape
            if x.dim() == 1:
                x = x.unsqueeze(0)
                
            # Compute energy using vectorized operations
            return self.scale * torch.sum(torch.sin(x) ** 2, dim=1)
            
        def score(self, x: torch.Tensor) -> torch.Tensor:
            # Analytical gradient
            if x.dim() == 1:
                x = x.unsqueeze(0)
                
            return -2 * self.scale * torch.sin(x) * torch.cos(x)
    ```

## Debugging Energy Functions

Common issues with energy functions include:

1. **NaN/Inf Values**: Check for division by zero or log of negative numbers
2. **Poor Sampling**: Energy may not be well-defined or have numerical issues
3. **Training Instability**: Energy might grow unbounded or collapse

Debugging techniques:

```python
def debug_energy_function(energy_fn: BaseEnergyFunction, x: torch.Tensor) -> None:
    """Debug an energy function for common issues."""
    # Check for NaN/Inf in energy
    energy = energy_fn(x)
    if torch.isnan(energy).any() or torch.isinf(energy).any():
        print("Warning: Energy contains NaN or Inf values")
        
    # Check for NaN/Inf in score
    score = energy_fn.score(x)
    if torch.isnan(score).any() or torch.isinf(score).any():
        print("Warning: Score contains NaN or Inf values")
        
    # Check score magnitude
    score_norm = torch.norm(score, dim=1)
    if (score_norm > 1e3).any():
        print("Warning: Score has very large values")
        
    # Check energy range
    if energy.max() - energy.min() > 1e6:
        print("Warning: Energy has a very large range")
```

## Advanced Topics

### Spherical Energy Functions

Energy functions on constrained domains:

```python
class SphericalEnergy(BaseEnergyFunction):
    """Energy function defined on a unit sphere."""
    
    def __init__(self, base_energy: BaseEnergyFunction):
        """Initialize spherical energy function.
        
        Args:
            base_energy: Base energy function
        """
        super().__init__()
        self.base_energy = base_energy
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy on unit sphere.
        
        Args:
            x: Input tensor of shape (batch_size, dim)
            
        Returns:
            Tensor of shape (batch_size,) containing energy values
        """
        # Project to unit sphere
        x_normalized = F.normalize(x, p=2, dim=1)
        return self.base_energy(x_normalized)
```

### Energy from Density Model

Creating an energy function from a density model:

```python
class DensityModelEnergy(BaseEnergyFunction):
    """Energy function from a density model."""
    
    def __init__(self, density_model: Callable):
        """Initialize energy function from density model.
        
        Args:
            density_model: Model that computes log probability
        """
        super().__init__()
        self.density_model = density_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy as negative log probability.
        
        Args:
            x: Input tensor of shape (batch_size, dim)
            
        Returns:
            Tensor of shape (batch_size,) containing energy values
        """
        log_prob = self.density_model.log_prob(x)
        return -log_prob
```

## Resources

<div class="grid cards" markdown>

-   :material-function:{ .lg .middle } __Core Components__

    ---

    Learn about core components and their interactions.

    [:octicons-arrow-right-24: Core Components](core_components.md)

-   :material-access-point:{ .lg .middle } __Samplers__

    ---

    Explore how samplers work with energy functions.

    [:octicons-arrow-right-24: Samplers](implementation_samplers.md)

-   :material-code-tags:{ .lg .middle } __Code Style__

    ---

    Follow coding standards when implementing energy functions.

    [:octicons-arrow-right-24: Code Style](code_style.md)

</div> 