---
sidebar_position: 1
title: Contributing Guidelines
description: How to contribute to TorchEBM
---

# Contributing to TorchEBM

Thank you for your interest in contributing to TorchEBM! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

We expect all contributors to follow our Code of Conduct. Please be respectful and inclusive in your interactions with others.

## Ways to Contribute

There are many ways to contribute to TorchEBM:

1. **Report bugs**: Report bugs or issues by opening an issue on GitHub
2. **Suggest features**: Suggest new features or improvements
3. **Improve documentation**: Fix typos, clarify explanations, or add examples
4. **Write code**: Implement new features, fix bugs, or improve performance
5. **Review pull requests**: Help review code from other contributors

## Development Workflow

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/torchebm.git
   cd torchebm
   ```
3. Set the original repository as an upstream remote:
   ```bash
   git remote add upstream https://github.com/soran-ghaderi/torchebm.git
   ```
4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run tests to make sure your changes don't break existing functionality:
   ```bash
   pytest
   ```
4. Add and commit your changes using our [commit message conventions](#commit-message-conventions)

### Code Style

We follow PEP 8 for Python code style with some modifications:

- Line length limit: 88 characters
- Use double quotes for strings
- Follow naming conventions:
  - Classes: `CamelCase`
  - Functions and variables: `snake_case`
  - Constants: `UPPER_CASE`

We use `black` for code formatting and `isort` for sorting imports.

### Commit Message Conventions

We follow a specific format for commit messages to make the project history clear and generate meaningful changelogs. Each commit message should have a specific format:

The first line should be max 50-60 chars. Any further details should be in the next lines separated by an empty line.

- **✨ feat**: Introduces a new feature
- **🐛 fix**: Patches a bug in the codebase
- **📖 docs**: Changes related to documentation
- **💎 style**: Changes that do not affect the meaning of the code (formatting)
- **📦 refactor**: Code changes that neither fix a bug nor add a feature
- **🚀 perf**: Improvements to performance
- **🚨 test**: Adding or correcting tests
- **👷 build**: Changes affecting the build system or external dependencies
- **💻 ci**: Changes to Continuous Integration configuration
- **🎫 chore**: Miscellaneous changes that don't modify source or test files
- **🔙 revert**: Reverts a previous commit

Example:
```
✨ feat: new feature implemented

The details of the commit (if any) go here.
```

For version bumping, include one of these tags in your commit message:
- Use `#major` for breaking changes
- Use `#minor` for new features
- Default is patch level for bug fixes

For releasing to PyPI, include `#release` in your commit message.

For more detailed information about our commit message conventions, please see our [Commit Message Conventions](commit_conventions.md) guide.

### Submitting Changes

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Create a pull request on GitHub
3. In your pull request description, explain the changes and link to any related issues
4. Wait for a review and address any feedback

## Pull Request Guidelines

- Keep pull requests focused on a single task
- Add tests for new features or bug fixes
- Update documentation as needed
- Ensure all tests pass
- Follow the code style guidelines

## Issue Guidelines

When opening an issue, please provide:

- A clear and descriptive title
- A detailed description of the issue
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Version information (TorchEBM, PyTorch, Python, CUDA)
- Any relevant code snippets or error messages

## Implementing New Features

### Samplers

When implementing a new sampler:

1. Create a new file in `torchebm/samplers/`
2. Extend the `BaseSampler` class
3. Implement the required methods:
   - `__init__`: Initialize the sampler with appropriate parameters
   - `step`: Implement the sampling step
   - `sample_chain`: Implement a full sampling chain (or use the default implementation)
4. Add tests in `tests/samplers/`
5. Update documentation

Example:

```python
from torchebm.core import BaseSampler
import torch

class MySampler(BaseSampler):
    def __init__(self, energy_function, param1, param2, device="cpu"):
        super().__init__(energy_function, device)
        self.param1 = param1
        self.param2 = param2
    
    def sample_chain(self, x, step_idx=None):
        # Implement your sampling algorithm here
        # x shape: [n_samples, dim]
        
        # Your sampler logic
        x_new = ...
        
        # Return updated samples and any diagnostics
        return x_new, {"diagnostic1": value1, "diagnostic2": value2}
```

### Energy Functions

When implementing a new energy function:

1. Create a new class in `torchebm/core/energy_function.py` or a new file in `torchebm/core/`
2. Extend the `BaseEnergyFunction` class
3. Implement the required methods:
   - `__init__`: Initialize the energy function with appropriate parameters
   - `forward`: Compute the energy value for a given input
4. Add tests in `tests/core/`
5. Update documentation

Example:

```python
from torchebm.core import BaseEnergyFunction
import torch


class MyEnergyFunction(BaseEnergyFunction):
    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def forward(self, x):
        # Implement your energy function here
        # x shape: [batch_size, dimension]
        # Return shape: [batch_size]
        return torch.sum(self.param1 * x ** 2 + self.param2 * torch.sin(x), dim=-1)
```

### BaseLoss Functions

When implementing a new loss function:

1. Create a new class in `torchebm/losses/`
2. Implement the required methods:
   - `__init__`: Initialize the loss function with appropriate parameters
   - `forward`: Compute the loss value
3. Add tests in `tests/losses/`
4. Update documentation

Example:

```python
import torch
import torch.nn as nn

class MyLossFunction(nn.Module):
    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def forward(self, model, data_samples):
        # Implement your loss function here
        # Return a scalar loss value
        return loss
```

## Documentation Guidelines

- Use clear, concise language
- Include examples for complex functionality
- Document parameters, return values, and exceptions
- Add docstrings to classes and functions
- Update the roadmap when implementing new features

## Getting Help

If you need help or have questions:

- Check existing documentation
- Search for similar issues on GitHub
- Ask for help in your pull request or issue

## Thank You!

Thank you for contributing to TorchEBM! Your help is greatly appreciated and makes the library better for everyone. 