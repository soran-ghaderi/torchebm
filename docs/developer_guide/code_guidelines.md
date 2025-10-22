---
title: Code Guidelines
description: Standards for writing high-quality code in TorchEBM
icon: material/code-braces
---

# Code Guidelines

This document outlines the standards for writing code, designing APIs, and testing in TorchEBM. Following these guidelines ensures our codebase is consistent, readable, and maintainable.

## Code Style

TorchEBM follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) and uses automatic formatters to enforce a consistent style.

### Automatic Formatting and Linting

We use the following tools to maintain code quality. Please run them before committing your changes.

<div class="grid cards" markdown>

-   :material-format-paint:{ .lg .middle } __Black__

    ---

    Formats Python code automatically and uncompromisingly.

    ```bash
    black torchebm/ tests/
    ```

-   :material-sort:{ .lg .middle } __isort__

    ---

    Sorts imports alphabetically and separates them into sections.

    ```bash
    isort torchebm/ tests/
    ```

-   :fontawesome-solid-broom:{ .lg .middle } __Flake8__

    ---

    Linter to check for style and logical issues.

    ```bash
    flake8 torchebm/ tests/
    ```

</div>

### Naming Conventions

*   **Classes**: `CamelCase` (e.g., `LangevinDynamics`)
*   **Functions & Variables**: `snake_case` (e.g., `compute_energy`)
*   **Constants**: `UPPER_CASE` (e.g., `DEFAULT_STEP_SIZE`)

### Docstrings and Type Annotations

*   Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
*   Provide type hints for all function signatures.

```python
from typing import Optional, Tuple
import torch

def sample_chain(
    dim: int,
    n_steps: int,
    n_samples: int = 1
) -> Tuple[torch.Tensor, dict]:
    """Generate samples using a Markov chain.

    Args:
        dim: Dimensionality of the samples.
        n_steps: Number of steps in the chain.
        n_samples: Number of parallel chains to run.

    Returns:
        A tuple containing the final samples and a dictionary of diagnostics.
    """
    # ... implementation ...
```

## API Design Principles

Our API is designed to be intuitive, consistent, and flexible.

### Core Philosophy

*   **Simplicity**: Simple use cases should be simple. Advanced functionality should be available but not intrusive.
*   **Consistency**: Similar operations should have similar interfaces. Parameter names and ordering should be consistent across the library.
*   **Explicitness**: Configuration should be explicit, primarily through constructor arguments.

### Key Patterns

*   **Base Classes**: Core components like models, samplers, and losses inherit from base classes (`BaseModel`, `BaseSampler`, `BaseLoss`) that define a common interface.
*   **Composition**: Complex functionality is built by composing simpler components.
*   **Clear Return Values**: Functions return a single value, a tuple for multiple values, or a dictionary for complex outputs with named fields. Diagnostic information is often returned in a separate dictionary.

## Testing Guidelines

Comprehensive testing is crucial for the reliability of TorchEBM. We use `pytest` for all tests.

### Testing Philosophy

*   **Unit Tests**: Test individual components in isolation.
*   **Integration Tests**: Test how components interact.
*   **Numerical Tests**: Verify the correctness and stability of numerical algorithms.
*   **Property-Based Tests**: Use libraries like `hypothesis` to test that functions satisfy certain properties for a wide range of inputs.

### Writing Tests

*   Test files are located in the `tests/` directory and mirror the structure of the `torchebm/` package.
*   Test files must be named `test_*.py`.
*   Test functions must be named `test_*`.
*   Use `pytest.fixture` to create reusable test objects.
*   Use `pytest.mark.parametrize` to test a function with multiple different inputs.

### Running Tests

Run all tests from the root of the repository:

```bash
pytest
```

To get a coverage report:

```bash
pytest --cov=torchebm
```

We aim for high test coverage across the library. Pull requests that decrease coverage will not be merged.
