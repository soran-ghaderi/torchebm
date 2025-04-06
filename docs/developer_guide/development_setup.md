---
title: Development Setup
description: Set up your local environment for TorchEBM development
icon: material/tools
---

# Development Setup

!!! tip "Quick Start"
    If you're just getting started with TorchEBM development, this guide will help you set up your environment properly.

## Prerequisites

Before setting up the TorchEBM development environment, make sure you have the following:

<div class="grid cards" markdown>

-   :material-language-python:{ .lg .middle } __Python 3.9+__

    ---

    TorchEBM requires Python 3.9 or higher.

    [:octicons-arrow-right-24: Install Python](https://www.python.org/downloads/)

-   :material-git:{ .lg .middle } __Git__

    ---

    You'll need Git for version control.

    [:octicons-arrow-right-24: Install Git](https://git-scm.com/downloads)

-   :fontawesome-brands-github:{ .lg .middle } __GitHub Account__

    ---

    For contributing to the repository.

    [:octicons-arrow-right-24: Create GitHub Account](https://github.com/join)

</div>

## Setting Up Your Environment

### 1. Fork and Clone the Repository

=== "GitHub UI"
    1. Navigate to [TorchEBM repository](https://github.com/soran-ghaderi/torchebm)
    2. Click the **Fork** button in the top-right corner
    3. Clone your fork to your local machine:
       ```bash
       git clone https://github.com/YOUR-USERNAME/torchebm.git
       cd torchebm
       ```

=== "GitHub CLI"
    ```bash
    gh repo fork soran-ghaderi/torchebm --clone=true
    cd torchebm
    ```

### 2. Set Up Virtual Environment

It's recommended to use a virtual environment to manage dependencies:

=== "venv"
    ```bash
    python -m venv venv
    # Activate on Windows
    venv\Scripts\activate
    # Activate on macOS/Linux
    source venv/bin/activate
    ```

=== "conda"
    ```bash
    conda create -n torchebm python=3.9
    conda activate torchebm
    ```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This will install TorchEBM in development mode along with all development dependencies.

## Development Workflow

<div class="grid" markdown>

<div markdown>
### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```
</div>

<div markdown>
### 2. Make Changes

Make your changes to the codebase.
</div>

<div markdown>
### 3. Run Tests

```bash
pytest
```
</div>

<div markdown>
### 4. Commit Changes

Follow our [commit conventions](commit_conventions.md).

```bash
git commit -m "feat: add new feature"
```
</div>

</div>

### 5. Push Changes and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then go to GitHub and create a pull request.

## Documentation Development

To preview documentation locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

This will start a local server at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

## Common Issues

!!! question "Missing CUDA?"
    If you're developing CUDA extensions, ensure you have the right CUDA toolkit installed:
    
    ```bash
    pip install torch==1.12.0+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html
    ```

## Next Steps

<div class="grid cards" markdown>

-   :fontawesome-solid-code:{ .lg .middle } __Code Style__

    ---

    Learn about our coding standards.

    [:octicons-arrow-right-24: Code Style](code_style.md)

-   :fontawesome-solid-vial:{ .lg .middle } __Testing Guide__

    ---

    Learn how to write effective tests.

    [:octicons-arrow-right-24: Testing Guide](testing_guide.md)

-   :fontawesome-solid-book:{ .lg .middle } __API Design__

    ---

    Understand our API design principles.

    [:octicons-arrow-right-24: API Design](api_design.md)

</div> 