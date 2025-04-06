---
title: Testing Guide
description: Best practices for writing and running tests for TorchEBM
icon: fontawesome/solid/vial
---

# Testing Guide

!!! info "Quality Assurance"
    Comprehensive testing is essential for maintaining the reliability and stability of TorchEBM. This guide outlines our testing approach and best practices.

## Testing Philosophy

TorchEBM follows test-driven development principles where appropriate, especially for core functionality. Our testing strategy includes:

<div class="grid cards" markdown>

-   :fontawesome-solid-cube:{ .lg .middle } __Unit Tests__

    ---

    Test individual components in isolation to ensure they work correctly.

-   :fontawesome-solid-cubes:{ .lg .middle } __Integration Tests__

    ---

    Test combinations of components to ensure they work together seamlessly.

-   :material-chart-line:{ .lg .middle } __Performance Tests__

    ---

    Measure the speed and resource usage of critical operations.

-   :material-numeric:{ .lg .middle } __Numerical Tests__

    ---

    Verify numerical correctness of algorithms against known results.

</div>

## Test Directory Structure

```
tests/
├── unit/                # Unit tests
│   ├── core/            # Tests for core module
│   ├── samplers/        # Tests for samplers module
│   ├── losses/          # Tests for losses module
│   └── utils/           # Tests for utilities
├── integration/         # Integration tests
├── performance/         # Performance benchmarks
├── conftest.py          # Pytest configuration and fixtures
└── utils.py             # Test utilities
```

## Running Tests

=== "Basic Usage"
    ```bash
    # Run all tests
    pytest
    
    # Run specific tests
    pytest tests/unit/core/
    pytest tests/unit/samplers/test_langevin.py
    
    # Run specific test class
    pytest tests/unit/core/test_energy.py::TestGaussianEnergy
    
    # Run specific test method
    pytest tests/unit/core/test_energy.py::TestGaussianEnergy::test_energy_computation
    ```

=== "Coverage"
    ```bash
    # Run tests with coverage
    pytest --cov=torchebm
    
    # Generate HTML coverage report
    pytest --cov=torchebm --cov-report=html
    ```

=== "Parallel Execution"
    ```bash
    # Run tests in parallel (4 processes)
    pytest -n 4
    ```

## Writing Tests

We use [pytest](https://docs.pytest.org/) for all our tests. Here are guidelines for writing effective tests:

### Test Class Structure

```python
import pytest
import torch
from torchebm.core import GaussianEnergy

class TestGaussianEnergy:
    @pytest.fixture
    def energy_fn(self):
        """Fixture to create a standard Gaussian energy function."""
        return GaussianEnergy(
            mean=torch.zeros(2),
            cov=torch.eye(2)
        )
    
    def test_energy_computation(self, energy_fn):
        """Test that energy is correctly computed for known inputs."""
        x = torch.zeros(2)
        energy = energy_fn(x)
        assert energy.item() == 0.0
        
        x = torch.ones(2)
        energy = energy_fn(x)
        assert torch.isclose(energy, torch.tensor(1.0))
```

### Test Naming Conventions

* Test files should be named `test_*.py`
* Test classes should be named `Test*`
* Test methods should be named `test_*`
* Use descriptive names that indicate what's being tested

### Parametrized Tests

Use `pytest.mark.parametrize` for testing multiple inputs:

```python
import pytest
import torch
from torchebm.core import GaussianEnergy

class TestGaussianEnergy:
    @pytest.mark.parametrize("mean,cov,x,expected", [
        (torch.zeros(2), torch.eye(2), torch.zeros(2), 0.0),
        (torch.zeros(2), torch.eye(2), torch.ones(2), 1.0),
        (torch.ones(2), torch.eye(2), torch.zeros(2), 1.0),
    ])
    def test_energy_parametrized(self, mean, cov, x, expected):
        energy_fn = GaussianEnergy(mean=mean, cov=cov)
        energy = energy_fn(x)
        assert torch.isclose(energy, torch.tensor(expected))
```

### Fixtures

Use fixtures for common setup code:

```python
import pytest
import torch

@pytest.fixture
def device():
    """Return the default device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def precision():
    """Return the default precision for comparison."""
    return 1e-5
```

## Testing CUDA Code

When testing CUDA code, follow these guidelines:

```python
import pytest
import torch
from torchebm.cuda import cuda_function

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_function():
    # Prepare test data
    x = torch.randn(100, device="cuda")
    
    # Call function
    result = cuda_function(x)
    
    # Verify result
    expected = x * 2  # Hypothetical expected result
    assert torch.allclose(result, expected)
```

## Mocking

Use `unittest.mock` or `pytest-mock` for mocking dependencies:

```python
def test_with_mock(mocker):
    # Mock an expensive function
    mock_compute = mocker.patch("torchebm.utils.compute_expensive_function")
    mock_compute.return_value = torch.tensor(1.0)
    
    # Test code that uses the mocked function
    # ...
    
    # Verify the mock was called correctly
    mock_compute.assert_called_once_with(torch.tensor(0.0))
```

## Property-Based Testing

For complex functions, consider using property-based testing with [Hypothesis](https://hypothesis.readthedocs.io/):

```python
import hypothesis.strategies as st
from hypothesis import given
import torch
from torchebm.core import GaussianEnergy

@given(
    x=st.lists(st.floats(min_value=-10, max_value=10), min_size=2, max_size=2).map(torch.tensor)
)
def test_gaussian_energy_properties(x):
    """Test properties of Gaussian energy function."""
    energy_fn = GaussianEnergy(mean=torch.zeros(2), cov=torch.eye(2))
    
    # Property: energy is non-negative for standard Gaussian
    energy = energy_fn(x)
    assert energy >= 0
    
    # Property: energy is minimized at the mean
    energy_at_mean = energy_fn(torch.zeros(2))
    assert energy >= energy_at_mean
```

## Performance Testing

For critical components, include performance tests:

```python
import pytest
import time
import torch
from torchebm.samplers import LangevinDynamics
from torchebm.core import GaussianEnergy

@pytest.mark.performance
def test_langevin_performance():
    """Test the performance of Langevin dynamics sampling."""
    energy_fn = GaussianEnergy(mean=torch.zeros(10), cov=torch.eye(10))
    sampler = LangevinDynamics(energy_function=energy_fn, step_size=0.01)
    
    # Warm-up
    sampler.sample_chain(dim=10, n_steps=10, n_samples=100)
    
    # Timed test
    start_time = time.time()
    sampler.sample_chain(dim=10, n_steps=1000, n_samples=1000)
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"Sampling took {elapsed:.4f} seconds")
    
    # Ensure performance meets requirements
    assert elapsed < 2.0  # Adjust threshold as needed
```

## Test Coverage Requirements

TorchEBM aims for high test coverage:

* Core modules: 90%+ coverage
* Samplers and losses: 85%+ coverage
* Utilities: 80%+ coverage
* CUDA code: 75%+ coverage

Use `pytest-cov` to measure coverage:

```bash
pytest --cov=torchebm --cov-report=term-missing
```

## Continuous Integration

Our CI pipeline automatically runs tests on every pull request:

* All tests must pass before a PR can be merged
* Coverage should not decrease
* Performance tests should not show significant regressions

!!! tip "Local CI"
    Before submitting a PR, run the full test suite locally to ensure it passes:
    
    ```bash
    # Install test dependencies
    pip install -e ".[test]"
    
    # Run all tests
    pytest
    
    # Check coverage
    pytest --cov=torchebm
    ```

## Resources

<div class="grid cards" markdown>

-   :fontawesome-brands-python:{ .lg .middle } __pytest Documentation__

    ---

    Comprehensive guide to pytest features.

    [:octicons-arrow-right-24: pytest Docs](https://docs.pytest.org/)

-   :material-chart-bar:{ .lg .middle } __pytest-cov__

    ---

    Coverage plugin for pytest.

    [:octicons-arrow-right-24: pytest-cov Docs](https://pytest-cov.readthedocs.io/)

-   :material-bug:{ .lg .middle } __Hypothesis__

    ---

    Property-based testing for Python.

    [:octicons-arrow-right-24: Hypothesis Docs](https://hypothesis.readthedocs.io/)

</div> 