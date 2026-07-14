# Contributing to TorchEBM

Thank you for your interest in contributing to TorchEBM! We welcome contributions of all kinds, including bug fixes, documentation improvements, new examples, performance enhancements, and new features. Whether you're fixing a bug, improving documentation, or adding a new feature, we're happy to have your contribution.

## Before You Start

Please read the Developer Guide before making any changes:

[Developer Guide](https://soran-ghaderi.github.io/torchebm/latest/developer_guide/)

The guide covers:

- Development setup
- Project architecture
- Code guidelines
- Performance and benchmarking
- Testing
- Pull request workflow

## Quick Start

Clone your fork and install the development dependencies:

```bash
git clone https://github.com/<your-github-username>/torchebm.git
cd torchebm
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -e ".[dev]"
```

For documentation work:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Development Workflow

1. Create a new branch from `master`.
2. Make your changes.
3. Format your code.
4. Run the test suite (`pytest tests/ -v`).
5. Commit using Conventional Commits.
6. Open a Pull Request linked to the relevant issue.

## Pull Request Checklist

Before opening a Pull Request, ensure that:

- Tests pass (`pytest tests/ -v`)
- Code is formatted with `black` and `isort`
- Commit messages follow the Conventional Commits format
- Related issues are linked when applicable.

For complete contribution instructions, please refer to the [Developer Guide](https://soran-ghaderi.github.io/torchebm/latest/developer_guide/).