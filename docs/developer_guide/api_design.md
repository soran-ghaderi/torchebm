---
sidebar_position: 7
title: API Design
description: Design principles and patterns for TorchEBM's API
---

# API Design

This document outlines the design principles and patterns used in TorchEBM's API, providing guidelines for contributors and insights for users building on top of the library.

## API Design Philosophy

!!! info "Design Goals"
    TorchEBM's API is designed with these goals in mind:

    1. **Intuitive**: APIs should be easy to understand and use
    2. **Consistent**: Similar operations should have similar interfaces
    3. **Pythonic**: Follow Python conventions and best practices
    4. **Flexible**: Allow for customization and extension
    5. **Type-Safe**: Use type hints for better IDE support and error checking

## Core Abstractions

<div class="grid cards" markdown>

-   :material-function-variant:{ .lg .middle } __Energy Functions__

    ---

    Energy functions define the energy landscape that characterizes a probability distribution.

    ```python
    class BaseEnergyFunction(nn.Module):
        def forward(self, x):
            # Return energy values for inputs x
            pass
            
        def gradient(self, x):
            # Return energy gradients for inputs x
            pass
    ```

-   :material-sync:{ .lg .middle } __Samplers__

    ---

    Samplers generate samples from energy functions via various algorithms.

    ```python
    class BaseSampler:
        def __init__(self, energy_function, device="cpu"):
            self.energy_function = energy_function
            self.device = device
            
        def sample_chain(self, dim, n_steps, n_samples=1):
            # Generate samples
            pass
    ```

-   :material-scale-balance:{ .lg .middle } __Loss Functions__

    ---

    BaseLoss functions are used to train energy-based models, often using samplers.

    ```python
    class ContrastiveDivergence(nn.Module):
        def __init__(self, energy_fn, sampler, mcmc_steps=1):
            self.energy_fn = energy_fn
            self.sampler = sampler
            self.mcmc_steps = mcmc_steps
            
        def forward(self, data_samples):
            # Compute loss
            pass
    ```

-   :material-cube-outline:{ .lg .middle } __Models__

    ---

    Neural network models that can be used as energy functions.

    ```python
    class BaseModel(BaseEnergyFunction):
        def __init__(self):
            super().__init__()
            # Define model architecture
            
        def forward(self, x):
            # Compute energy
            pass
    ```

</div>

## Interface Design Patterns

### Method Naming Conventions

TorchEBM follows consistent naming patterns:

| Pattern | Example | Purpose |
|---------|---------|---------|
| `forward()` | `energy_fn.forward(x)` | Core computation (energy) |
| `gradient()` | `energy_fn.gradient(x)` | Compute gradients |
| `sample_chain()` | `sampler.sample_chain(dim, n_steps)` | Generate samples |
| `step()` | `sampler.step(x)` | Single sampling step |

### Parameter Ordering

Parameters follow a consistent ordering pattern:

1. Required parameters (e.g., input data, dimensions)
2. Algorithm-specific parameters (e.g., step size, number of steps)
3. Optional parameters with defaults (e.g., device, random seed)

### Return Types

Return values are consistently structured:

- Single values are returned directly
- Multiple return values use tuples
- Complex returns use dictionaries for named access
- Diagnostic information is returned in a separate dictionary

Example:
```python
# Return samples and diagnostics
samples, diagnostics = sampler.sample_chain(dim=2, n_steps=100)

# Access diagnostic information
acceptance_rate = diagnostics['acceptance_rate']
energy_trajectory = diagnostics['energy_trajectory']
```

## Extension Patterns

### Subclassing Base Classes

The primary extension pattern is to subclass the appropriate base class:

```python
class MyCustomSampler(BaseSampler):
    def __init__(self, energy_function, special_param, device="cpu"):
        super().__init__(energy_function, device)
        self.special_param = special_param
        
    def sample_chain(self, x, step_idx=None):
        # Custom sampling logic
        return x_new, diagnostics
```

### Composition Pattern

For more complex extensions, composition can be used:

```python
class HybridSampler(BaseSampler):
    def __init__(self, energy_function, sampler1, sampler2, switch_freq=10):
        super().__init__(energy_function)
        self.sampler1 = sampler1
        self.sampler2 = sampler2
        self.switch_freq = switch_freq
        
    def sample_chain(self, x, step_idx=None):
        # Choose sampler based on step index
        if step_idx % self.switch_freq < self.switch_freq // 2:
            return self.sampler1.step(x, step_idx)
        else:
            return self.sampler2.step(x, step_idx)
```

## Configuration and Customization

### Constructor Parameters

Features are enabled and configured primarily through constructor parameters:

```python
# Configure through constructor parameters
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.01,
    noise_scale=0.1,
    thinning=5,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

### Method Parameters

Runtime behavior is controlled through method parameters:

```python
# Control sampling behavior through method parameters
samples = sampler.sample_chain(
    dim=2,
    n_steps=1000,
    n_samples=100,
    initial_samples=None,  # If None, random initialization
    burn_in=100,
    verbose=True
)
```

## Handling Errors and Edge Cases

TorchEBM follows these practices for error handling:

1. **Input Validation**: Validate inputs early and raise descriptive exceptions
2. **Graceful Degradation**: Fall back to simpler algorithms when necessary
3. **Informative Exceptions**: Provide clear error messages with suggestions
4. **Default Safety**: Choose safe default values that work in most cases

Example:
```python
def sample_chain(self, dim, n_steps, n_samples=1):
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
        
    if dim <= 0:
        raise ValueError("dim must be positive")
        
    # Implementation
```

## API Evolution Guidelines

When evolving the API, we follow these guidelines:

1. **Backward Compatibility**: Avoid breaking changes when possible
2. **Deprecation Cycle**: Use deprecation warnings before removing features
3. **Default Arguments**: Add new parameters with sensible defaults
4. **Feature Flags**: Use boolean flags to enable/disable new features

Example of deprecation:
```python
def old_method(self, param):
    warnings.warn(
        "old_method is deprecated and will be removed in a future version. "
        "Use new_method instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return self.new_method(param)
```

## Documentation Standards

All APIs should include:

- Docstrings for all public classes and methods
- Type hints for parameters and return values
- Examples showing common usage patterns
- Notes on performance implications
- References to relevant papers or algorithms

Example:
```python
def sample_chain(
    self, 
    dim: int, 
    n_steps: int, 
    n_samples: int = 1
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Generate samples using Langevin dynamics.
    
    Args:
        dim: Dimensionality of samples
        n_steps: Number of sampling steps
        n_samples: Number of parallel chains
        
    Returns:
        tuple: (samples, diagnostics)
            - samples: Tensor of shape [n_samples, dim]
            - diagnostics: Dict with sampling statistics
            
    Example:
        >>> energy_fn = GaussianEnergy(torch.zeros(2), torch.eye(2))
        >>> sampler = LangevinDynamics(energy_fn, step_size=0.1)
        >>> samples, _ = sampler.sample_chain(dim=2, n_steps=100, n_samples=10)
    """
    # Implementation
```

By following these API design principles, TorchEBM maintains a consistent, intuitive, and extensible interface for energy-based modeling in PyTorch. 