---
title: Getting Started
description: Your first steps to contributing to TorchEBM
icon: material/rocket-launch
---

# Getting Started with TorchEBM Development

Welcome to the TorchEBM development community! This guide provides everything you need to set up your environment, understand our workflow, and make your first contribution.

## Ways to Contribute

We welcome contributions of all kinds:

<div class="grid cards" markdown>

-   :material-bug:{ .lg .middle } __Report Bugs__

    ---

    Find a bug? Report it on our [issue tracker](https://github.com/soran-ghaderi/torchebm/issues) with a clear description and steps to reproduce.

-   :material-lightbulb-on:{ .lg .middle } __Suggest Features__

    ---

    Have an idea for a new feature or an improvement? Share it in the [discussions](https://github.com/soran-ghaderi/torchebm/discussions).

-   :material-file-document-edit:{ .lg .middle } __Improve Documentation__

    ---

    Help us make our documentation better by fixing typos, clarifying explanations, or adding new examples.

-   :material-source-pull:{ .lg .middle } __Write Code__

    ---

    Contribute directly to the codebase by fixing bugs, implementing new features, or improving performance.

</div>

## Development Setup

Follow these steps to set up your local development environment.

### 1. Prerequisites

Make sure you have the following installed:

*   **Python 3.9+**
*   **Git**
*   A **GitHub Account**

### 2. Fork and Clone

First, fork the [TorchEBM repository](https://github.com/soran-ghaderi/torchebm) on GitHub. Then, clone your fork locally:

```bash
git clone https://github.com/YOUR-USERNAME/torchebm.git
cd torchebm
```

### 3. Set Up Virtual Environment

It's highly recommended to use a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Or on Windows
venv\Scripts\activate
```

### 4. Install Dependencies

Install TorchEBM in editable mode along with all development dependencies:

```bash
pip install -e ".[dev]"
```

## Contribution Workflow

### 1. Create a Branch

Create a new branch for your changes. Use a descriptive name, like `feature/new-sampler` or `fix/gradient-bug`.

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Make your changes to the codebase. Be sure to follow our [Code Guidelines](code_guidelines.md).

### 3. Run Tests

Before committing, run the tests to ensure your changes haven't broken anything:

```bash
pytest
```

### 4. Commit Your Changes

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. Your commit messages should be structured as follows:

```
<type>: <description>

[optional body]

[optional footer]
```

**Common types:**

| Type       | Description                                                 |
|------------|-------------------------------------------------------------|
| **feat**   | A new feature                                               |
| **fix**    | A bug fix                                                   |
| **docs**   | Documentation only changes                                  |
| **style**  | Changes that do not affect the meaning of the code          |
| **refactor**| A code change that neither fixes a bug nor adds a feature   |
| **perf**   | A code change that improves performance                     |
| **test**   | Adding missing tests or correcting existing tests           |
| **chore**  | Changes to the build process or auxiliary tools             |


**Example:**

```bash
git commit -m "feat: add support for adaptive step sizes in LangevinDynamics"
```

### 5. Create a Pull Request

Push your branch to your fork and open a pull request to the `main` branch of the TorchEBM repository.

```bash
git push origin feature/your-feature-name
```

In your pull request description, clearly explain the changes you've made and reference any relevant issues.

## Documentation Development

To work on the documentation locally, install the docs dependencies and serve the site:

```bash
pip install -e ".[docs]"
mkdocs serve
```

This will start a live-reloading server at `http://127.0.0.1:8000`.
