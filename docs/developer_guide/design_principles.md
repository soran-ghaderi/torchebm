---
title: Design Principles
description: Core design principles and philosophy behind TorchEBM
icon: material/lightbulb-outline
---

# Design Principles

!!! info "Project Philosophy"
    TorchEBM is built on a set of core design principles that guide its development. Understanding these principles will help you contribute in a way that aligns with the project's vision.

## Core Philosophy

TorchEBM aims to be:

<div class="grid cards" markdown>

-   :material-flash:{ .lg .middle } __Performant__

    ---

    High-performance implementations that leverage PyTorch's capabilities and CUDA acceleration.

-   :material-puzzle:{ .lg .middle } __Modular__

    ---

    Components that can be easily combined, extended, and customized.

-   :material-book-open-variant:{ .lg .middle } __Intuitive__

    ---

    Clear, well-documented APIs that are easy to understand and use.

-   :material-school:{ .lg .middle } __Educational__

    ---

    Serves as both a practical tool and a learning resource for energy-based modeling.

</div>

## Key Design Patterns

### Composable Base Classes

TorchEBM is built around a set of extensible base classes that provide common interface:

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

This design allows for:

* **Composition**: Combining energy functions via addition, multiplication, etc.
* **Extension**: Creating new energy functions by subclassing
* **Integration**: Using energy functions with any sampler that follows the interface

### Factory Methods

Factory methods create configured instances with sensible defaults:

```python
@classmethod
def create_standard(cls, dim: int = 2) -> 'GaussianEnergy':
    """Create a standard Gaussian energy function."""
    return cls(mean=torch.zeros(dim), cov=torch.eye(dim))
```

### Configuration through Constructor

Classes are configured through their constructor rather than setter methods:

```python
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.01,
    noise_scale=1.0
)
```

This approach:

* Makes the configuration explicit and clear
* Encourages immutability of key parameters
* Simplifies object creation and usage

### Method Chaining

Methods return the object itself when appropriate to allow method chaining:

```python
result = (
    sampler
    .set_device("cuda" if torch.cuda.is_available() else "cpu")
    .set_seed(42)
    .sample_chain(dim=2, n_steps=1000)
)
```

### Lazily-Evaluated Operations

Computations are performed lazily when possible to avoid unnecessary work:

```python
# Create a sampler with a sampling trajectory
sampler = LangevinDynamics(energy_fn)
trajectory = sampler.sample_trajectory(dim=2, n_steps=1000)

# Compute statistics only when needed
mean = trajectory.mean()  # Computation happens here
variance = trajectory.variance()  # Computation happens here
```

## Architecture Principles

### Separation of Concerns

Components have clearly defined responsibilities:

* **Energy Functions**: Define the energy landscape
* **Samplers**: Generate samples from energy functions
* **Losses**: Train energy functions from data
* **Models**: Parameterize energy functions using neural networks
* **Utils**: Provide supporting functionality

### Minimizing Dependencies

Each module has minimal dependencies on other modules:

* Core modules (e.g., `core`, `samplers`) don't depend on higher-level modules
* Utility modules are designed to be used by all other modules
* CUDA implementations are separated to allow for CPU-only usage

### Consistent Error Handling

Error handling follows consistent patterns:

* Use descriptive error messages that suggest solutions
* Validate inputs early with helpful validation errors
* Provide debug information when operations fail

```python
def validate_dimensions(tensor: torch.Tensor, expected_dims: int) -> None:
    """Validate that tensor has the expected number of dimensions."""
    if tensor.dim() != expected_dims:
        raise ValueError(
            f"Expected tensor with {expected_dims} dimensions, "
            f"but got tensor with shape {tensor.shape}"
        )
```

### Consistent API Design

APIs are designed consistently across the library:

* Similar operations have similar interfaces
* Parameters follow consistent naming conventions
* Return types are consistent and well-documented

### Progressive Disclosure

Simple use cases are simple, while advanced functionality is available but not required:

```python
# Simple usage
sampler = LangevinDynamics(energy_fn)
samples = sampler.sample(n_samples=1000)

# Advanced usage
sampler = LangevinDynamics(
    energy_fn,
    step_size=0.01,
    noise_scale=1.0,
    step_size_schedule=LinearSchedule(0.01, 0.001),
    metropolis_correction=True
)
samples = sampler.sample(
    n_samples=1000,
    initial_samples=initial_x,
    callback=logging_callback
)
```

## Implementation Principles

### PyTorch First

TorchEBM is built on PyTorch and follows PyTorch patterns:

* Use PyTorch's tensor operations whenever possible
* Follow PyTorch's model design patterns (e.g., `nn.Module`)
* Leverage PyTorch's autograd for gradient computation
* Support both CPU and CUDA execution

### Vectorized Operations

Operations are vectorized where possible for efficiency:

```python
# Good: Vectorized operations
def compute_pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, y, p=2)

# Avoid: Explicit loops
def compute_pairwise_distances_slow(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(x.size(0), y.size(0))
    for i in range(x.size(0)):
        for j in range(y.size(0)):
            result[i, j] = torch.norm(x[i] - y[j])
    return result
```

### CUDA Optimization

Performance-critical operations are optimized with CUDA when appropriate:

* CPU implementations as fallback
* CUDA implementations for performance
* Automatic selection based on available hardware

### Type Annotations

Code uses type annotations for clarity and static analysis:

```python
def sample_chain(
    self,
    dim: int,
    n_steps: int,
    n_samples: int = 1,
    initial_samples: Optional[torch.Tensor] = None,
    return_trajectory: bool = False
) -> Union[torch.Tensor, Trajectory]:
    """Generate samples using a Markov chain."""
    # Implementation
```

## Testing Principles

* **Unit Testing**: Individual components are thoroughly tested
* **Integration Testing**: Component interactions are tested
* **Property Testing**: Properties of algorithms are tested
* **Numerical Testing**: Numerical algorithms are tested for stability and accuracy

## Documentation Principles

Documentation is comprehensive and includes:

* **API Documentation**: Clear documentation of all public APIs
* **Tutorials**: Step-by-step guides for common tasks
* **Examples**: Real-world examples of using the library
* **Theory**: Explanations of the underlying mathematical concepts

## Future Compatibility

TorchEBM is designed with future compatibility in mind:

* **API Stability**: Breaking changes are minimized and clearly documented
* **Feature Flags**: Experimental features are clearly marked
* **Deprecation Warnings**: Deprecated features emit warnings before removal

## Contributing Guidelines

When contributing to TorchEBM, adhere to these design principles:

* Make sure new components follow existing patterns
* Keep interfaces consistent with the rest of the library
* Write thorough tests for new functionality
* Document public APIs clearly
* Optimize for readability and maintainability

!!! example "Design Example: Adding a New Sampler"
    When adding a new sampler:
    
    1. Subclass the `Sampler` base class
    2. Implement required methods (`sample`, `sample_chain`)
    3. Follow the existing parameter naming conventions
    4. Add comprehensive documentation
    5. Write tests that verify the sampler's properties
    6. Optimize performance-critical sections

<div class="grid cards" markdown>

-   :material-file-tree:{ .lg .middle } __Project Structure__

    ---

    Understand how the project is organized.

    [:octicons-arrow-right-24: Project Structure](project_structure.md)

-   :material-function:{ .lg .middle } __Core Components__

    ---

    Learn about the core components in detail.

    [:octicons-arrow-right-24: Core Components](core_components.md)

-   :material-code-tags:{ .lg .middle } __Code Style__

    ---

    Follow the project's coding standards.

    [:octicons-arrow-right-24: Code Style](code_style.md)

</div> 