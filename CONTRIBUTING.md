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

## Dependencies

`torch` is the only runtime dependency, and `torchebm/` imports nothing else. Data
loaders, pretrained checkpoints, trainer loops and plotting belong to the
application, not to the library.

The optional extras (`examples`, `dev`, `docs`) serve the repository, never the
library at runtime. A new runtime extra is justified only when a module under
`torchebm/` needs it, and then it must:

- import the package lazily at the call site, never at module import time;
- raise an `ImportError` naming the extra to install;
- be named after the capability it unlocks (for example `vision`), not after a
  paper or an algorithm.

## Development Workflow

1. Claim the issue. Check that nobody is assigned and no open Pull Request
   already links it, then comment on the issue to say you are taking it.
   Pull Requests that duplicate an earlier one for the same issue are closed.
2. Create a new branch from `master`.
3. Make your changes.
4. Format your code.
5. Run the test suite (`pytest tests/ -v`).
6. Commit using Conventional Commits.
7. Open a Pull Request linked to the relevant issue.

## Pull Request Checklist

Before opening a Pull Request, ensure that:

- No earlier open Pull Request targets the same issue
- Tests pass (`pytest tests/ -v`)
- Code is formatted with `black` and `isort`
- Commit messages follow the Conventional Commits format
- Related issues are linked when applicable.

For complete contribution instructions, please refer to the [Developer Guide](https://soran-ghaderi.github.io/torchebm/latest/developer_guide/).