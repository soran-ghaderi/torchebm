# Contributing to TorchEBM

Thank you for your interest in contributing to TorchEBM! We welcome contributions of all kinds, including bug fixes, documentation improvements, new examples, performance enhancements, and new features. Whether you're fixing a bug, improving documentation, or adding a new feature, we're happy to have your contribution.

> ⭐ If TorchEBM is useful for your work, please consider starring the repository. It is the main way people discover the project.

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

## Coding Style and Rules

- Surgical diffs only. Do not reformat, reorder imports, or touch   lines unrelated to the issue you are closing.
- Docs are MkDocs, not Sphinx. Google-style docstrings with `r"""`.  LaTeX is \( \) inline and \[ \] block. No `.. math::`, no `:class:`.
- Type annotations go in the signature, not the docstring.
- Never modify base classes unless the issue names one.
- Every tensor op in its maximally optimized form. No host syncs (`.item()`, `.cpu()`, Python `and`/`or` on tensors) in hot paths.
- If you are adding something new, avoid parallel implementation, reuse existing code when available. avoid redundancy and non-necessary verbosity with codes and docstrings.

# Note

- If you notice something adjacent while working (a bug, a missing test, an unclear docstring), do not expand this PR. Open a separate issue using the same format: terse scope, then a `Done when:` line. Link it from your PR. Very brief message and use appropriate existing labels on the repo.


## Pull Request Checklist

Before opening a Pull Request, ensure that:

- No earlier open Pull Request targets the same issue (Please check PR messages and ensure others haven't targeted this issue, unless you find a mistake in their PR and you fix it which you should do so via the same opened PR. https://github.com/soran-ghaderi/torchebm/pulls)
- Tests pass (`pytest tests/ -v`)
- Code is formatted with `black` and `isort`
- Commit messages follow the Conventional Commits format
- No AI attribution in commits, PR title or PR description: no
  `Co-authored-by` trailers for AI tools, no "generated with" footers, no
  mentions of assistants. Trailers land in the contributors graph and PR text
  lands in the changelog; both must name people only.
- Related issues are linked when applicable.


For complete contribution instructions, please refer to the [Developer Guide](https://soran-ghaderi.github.io/torchebm/latest/developer_guide/).


## Support the project

Beyond code, the most useful things you can do are to star the repository, open an issue when something is unclear or broken, and cite TorchEBM if it supports published work (see `CITATION.cff`).
