---
title: Developer Guide
description: Comprehensive guide for TorchEBM developers
icon: material/book-open-page-variant
---

# TorchEBM Developer Guide

Welcome to the TorchEBM developer guide! This comprehensive resource is designed to help you understand the project's architecture, contribute effectively, and follow best practices when working with the codebase.

## Getting Started with Development

<div class="grid cards" markdown>

-   :material-cog-outline:{ .lg .middle } __Development Setup__

    ---

    Set up your development environment and prepare for contribution.

    [:octicons-arrow-right-24: Development Setup](development_setup.md)

-   :material-file-code-outline:{ .lg .middle } __Code Style__

    ---

    Learn about the coding standards and style guidelines.

    [:octicons-arrow-right-24: Code Style](code_style.md)

-   :material-test-tube:{ .lg .middle } __Testing Guide__

    ---

    Discover how to write and run tests for TorchEBM.

    [:octicons-arrow-right-24: Testing Guide](testing_guide.md)

-   :material-git-commit:{ .lg .middle } __Commit Conventions__

    ---

    Understand the commit message format and conventions.

    [:octicons-arrow-right-24: Commit Conventions](commit_conventions.md)

</div>

## Understanding the Architecture

<div class="grid cards" markdown>

-   :material-folder-outline:{ .lg .middle } __Project Structure__

    ---

    Explore the organization of the TorchEBM codebase.

    [:octicons-arrow-right-24: Project Structure](project_structure.md)

-   :material-lightbulb-outline:{ .lg .middle } __Design Principles__

    ---

    Learn about the guiding principles behind TorchEBM.

    [:octicons-arrow-right-24: Design Principles](design_principles.md)

-   :material-puzzle-outline:{ .lg .middle } __Core Components__

    ---

    Understand the core components and their interactions.

    [:octicons-arrow-right-24: Core Components](core_components.md)

</div>

## Implementation Details

<div class="grid cards" markdown>

-   :material-function:{ .lg .middle } __Energy Functions__

    ---

    Detailed information about energy function implementations.

    [:octicons-arrow-right-24: Energy Functions](implementation_energy.md)

-   :material-chart-bell-curve:{ .lg .middle } __Samplers__

    ---

    Understand how samplers are implemented in TorchEBM.

    [:octicons-arrow-right-24: Samplers](implementation_samplers.md)

-   :material-function-variant:{ .lg .middle } __Loss Functions__

    ---

    Learn about the implementation of various loss functions.

    [:octicons-arrow-right-24: BaseLoss Functions](implementation_losses.md)

-   :material-vector-square:{ .lg .middle } __Model Architecture__

    ---

    Explore the implementation of neural network models.

    [:octicons-arrow-right-24: Model Architecture](implementation_models.md)

-   :material-gpu:{ .lg .middle } __CUDA Optimizations__

    ---

    Discover performance optimizations using CUDA.

    [:octicons-arrow-right-24: CUDA Optimizations](cuda_optimizations.md)

</div>

## Contributing

<div class="grid cards" markdown>

-   :material-source-pull:{ .lg .middle } __Contributing__

    ---

    Guidelines for contributing to TorchEBM.

    [:octicons-arrow-right-24: Contributing](contributing.md)

-   :material-api:{ .lg .middle } __API Design__

    ---

    Learn about API design principles in TorchEBM.

    [:octicons-arrow-right-24: API Design](api_design.md)

-   :material-file-document-outline:{ .lg .middle } __API Generation__

    ---

    Understand how API documentation is generated.

    [:octicons-arrow-right-24: API Generation](api_generation.md)

-   :material-speedometer:{ .lg .middle } __Performance__

    ---

    Best practices for optimizing performance.

    [:octicons-arrow-right-24: Performance](performance.md)

</div>

## Contributing Process

The general process for contributing to TorchEBM involves:

1. **Set up the development environment**: Follow the [development setup guide](development_setup.md) to prepare your workspace.

2. **Understand the codebase**: Familiarize yourself with the [project structure](project_structure.md), [design principles](design_principles.md), and [core components](core_components.md).

3. **Make changes**: Implement your feature or bug fix, following the [code style](code_style.md) guidelines.

4. **Write tests**: Add tests for your changes as described in the [testing guide](testing_guide.md).

5. **Submit a pull request**: Follow the [contributing guidelines](contributing.md) to submit your changes.

We welcome contributions from the community and are grateful for your help in improving TorchEBM!

## Development Philosophy

TorchEBM aims to be:

* **Modular**: Components should be easy to combine and extend
* **Performant**: Critical operations should be optimized for speed
* **User-friendly**: APIs should be intuitive and well-documented
* **Well-tested**: Code should be thoroughly tested to ensure reliability

Learn more about our [Design Principles](design_principles.md).

## Quick Reference

* **Installation for development**: `pip install -e ".[dev]"`
* **Run tests**: `pytest`
* **Check code style**: `black torchebm/ && isort torchebm/ && flake8 torchebm/`
* **Build documentation**: `mkdocs serve`
* **Pre-commit hooks**: `pre-commit install`

## Getting Help

If you encounter issues during development, you can:

* **Open an issue** on GitHub
* **Ask questions** in the GitHub Discussions
* **Reach out to maintainers** via email or GitHub 