---
title: Code Style Guide
description: Coding standards and style guidelines for TorchEBM contributions
icon: fontawesome/solid/code
---

# Code Style Guide

!!! abstract "Consistent Style"
    Following a consistent code style ensures our codebase remains readable and maintainable. This guide outlines the style conventions used in TorchEBM.

## Python Style Guidelines

TorchEBM follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some project-specific guidelines.

### Automatic Formatting

We use several tools to automatically format and check our code:

<div class="grid cards" markdown>

-   :material-format-paint:{ .lg .middle } __Black__

    ---

    Automatic code formatter with a focus on consistency.

    ```bash
    black torchebm/
    ```

-   :material-sort:{ .lg .middle } __isort__

    ---

    Sorts imports alphabetically and separates them into sections.

    ```bash
    isort torchebm/
    ```

-   :fontawesome-solid-broom:{ .lg .middle } __Flake8__

    ---

    Linter to catch logical and stylistic issues.

    ```bash
    flake8 torchebm/
    ```

</div>

### Code Structure

=== "Function Definitions"

    ```python
    def function_name(
        param1: type,
        param2: type,
        param3: Optional[type] = None
    ) -> ReturnType:
        """Short description of the function.
        
        More detailed explanation if needed.
        
        Args:
            param1: Description of parameter 1
            param2: Description of parameter 2
            param3: Description of parameter 3
        
        Returns:
            Description of the return value
            
        Raises:
            ExceptionType: When and why this exception is raised
        """
        # Function implementation
        pass
    ```

=== "Class Definitions"

    ```python
    class ClassName(BaseClass):
        """Short description of the class.
        
        More detailed explanation if needed.
        
        Args:
            attr1: Description of attribute 1
            attr2: Description of attribute 2
        """
        
        def __init__(
            self,
            attr1: type,
            attr2: type = default_value
        ):
            """Initialize the class.
            
            Args:
                attr1: Description of attribute 1
                attr2: Description of attribute 2
            """
            self.attr1 = attr1
            self.attr2 = attr2
            
        def method_name(self, param: type) -> ReturnType:
            """Short description of the method.
            
            Args:
                param: Description of parameter
                
            Returns:
                Description of the return value
            """
            # Method implementation
            pass
    ```

### Naming Conventions

<div class="grid" markdown>

<div markdown>
#### Classes

Use `CamelCase` for class names:

```python
class BaseEnergyFunction:
    pass

class LangevinDynamics:
    pass
```
</div>

<div markdown>
#### Functions and Variables

Use `snake_case` for functions and variables:

```python
def compute_energy(x):
    pass

sample_count = 1000
```
</div>

</div>

#### Constants

Use `UPPER_CASE` for constants:

```python
DEFAULT_STEP_SIZE = 0.01
MAX_ITERATIONS = 1000
```

## Documentation Style

TorchEBM uses [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for all code documentation.

!!! example "Docstring Example"

    ```python
    def sample_chain(
        self, 
        dim: int, 
        n_steps: int, 
        n_samples: int = 1
    ) -> torch.Tensor:
        """Generate samples using a Markov chain of specified length.
        
        Args:
            dim: Dimensionality of samples
            n_steps: Number of steps in the chain
            n_samples: Number of parallel chains to run
            
        Returns:
            Tensor of shape (n_samples, dim) containing final samples
            
        Examples:
            >>> energy_fn = GaussianEnergy(torch.zeros(2), torch.eye(2))
            >>> sampler = LangevinDynamics(energy_fn, step_size=0.01)
            >>> samples = sampler.sample_chain(dim=2, n_steps=100, n_samples=10)
        """
    ```

## Type Annotations

We use Python's type hints to improve code readability and enable static type checking:

```python
from typing import Optional, List, Union, Dict, Tuple, Callable

def function(
    tensor: torch.Tensor,
    scale: float = 1.0,
    use_cuda: bool = False,
    callback: Optional[Callable[[torch.Tensor], None]] = None
) -> Tuple[torch.Tensor, float]:
    # Implementation
    pass
```

## CUDA Code Style

For CUDA extensions, we follow these conventions:

=== "File Organization"

    ```
    torchebm/cuda/
    ├── kernels/
    │   ├── kernel_name.cu
    │   └── kernel_name.cuh
    ├── bindings.cpp
    └── __init__.py
    ```

=== "CUDA Naming Conventions"

    ```cpp
    // Function names in snake_case
    __global__ void compute_energy_kernel(float* input, float* output, int n) {
        // Implementation
    }
    
    // Constants in UPPER_CASE
    #define BLOCK_SIZE 256
    ```

## Imports Organization

Organize imports in the following order:

1. Standard library imports
2. Related third-party imports
3. Local application/library specific imports

```python
# Standard library
import os
import sys
from typing import Optional, Dict

# Third-party
import numpy as np
import torch
import torch.nn as nn

# Local application
from torchebm.core import BaseEnergyFunction
from torchebm.utils import get_device
```

## Comments

* Use comments sparingly - prefer self-documenting code with clear variable names
* Add comments for complex algorithms or non-obvious implementations
* Update comments when you change code

!!! tip "Good Comments Example"
    ```python
    # Correcting for numerical instability by adding a small epsilon
    normalized_weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)
    ```

## Pre-commit Hooks

TorchEBM uses pre-commit hooks to enforce code style. Make sure to install them as described in the [Development Setup](development_setup.md) guide.

## Recommended Editor Settings

=== "VS Code"
    ```json
    {
      "editor.formatOnSave": true,
      "editor.codeActionsOnSave": {
        "source.organizeImports": true
      },
      "python.linting.enabled": true,
      "python.linting.flake8Enabled": true,
      "python.formatting.provider": "black"
    }
    ```

=== "PyCharm"
    1. Install Black and isort plugins
    2. Configure Code Style for Python to match PEP 8
    3. Set Black as the external formatter
    4. Enable "Reformat code on save"
    5. Configure isort for import optimization

## Style Enforcement

Our CI pipeline checks for style compliance. Pull requests failing style checks will be automatically rejected.

!!! warning "CI Pipeline Failure"
    If your PR fails CI due to style issues, run the following commands locally to fix them:
    
    ```bash
    # Format code with Black
    black torchebm/
    
    # Sort imports
    isort torchebm/
    
    # Run flake8
    flake8 torchebm/
    
    # Run mypy for type checking
    mypy torchebm/
    ``` 