---
title: Developer Guide
description: Contributor hub for TorchEBM, from fresh clone to merged PR
icon: material/book-open-page-variant
---

# Developer Guide

Everything you need to contribute to TorchEBM. This page takes you from a
fresh clone to a merged pull request; the other three pages cover the code
itself:

- [Architecture](architecture.md): package layout and how the core abstractions compose.
- [Code Guidelines](code_guidelines.md): style, API design, performance rules, testing.
- [Performance and Benchmarking](performance.md): writing fast code, and measuring it.

## Setup

Requirements: Python 3.9+, Git, a GitHub account.

```bash
# Fork on GitHub, then:
git clone https://github.com/<your-user>/torchebm.git
cd torchebm
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

For docs work, additionally:

```bash
pip install -e ".[docs]"
mkdocs serve                     # live preview at http://127.0.0.1:8000
```

## Workflow

1. **Claim the issue**. Check that nobody is assigned and no open PR already
   links it, then comment on the issue to say you are taking it. A PR that
   duplicates an earlier one for the same issue is closed in favour of the first.
2. **Branch** from `master` with a descriptive name.
   ```bash
   git checkout -b feat/adaptive-step-size
   ```
3. **Code**: follow the [Code Guidelines](code_guidelines.md). Mirror the package layout under `tests/`.
4. **Test and format** before every commit.
   ```bash
   black torchebm/ tests/
   isort torchebm/ tests/
   pytest tests/ -v
   ```
5. **Commit** using Conventional Commits (below).
6. **Push and open a PR** against `master`. Link any related issue.

## Commit conventions (mandatory)

TorchEBM uses [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <summary>
```

| Type       | Use for                                          |
|------------|--------------------------------------------------|
| `feat`     | A new user-facing feature                        |
| `fix`      | A bug fix                                        |
| `perf`     | A change that improves performance               |
| `refactor` | Code change that is neither a feature nor a fix  |
| `test`     | Adding or fixing tests                           |
| `docs`     | Documentation only                               |
| `style`    | Formatting, whitespace, no logic change          |
| `build`    | Build system, packaging                          |
| `ci`       | CI configuration                                 |
| `chore`    | Other maintenance (deps, tooling)                |

**Examples**

```text
feat(samplers): add adaptive step size to LangevinDynamics
fix(losses): correct gradient sign in EquilibriumMatching
perf(integrators): cache RK buffers on device once per integrate()
```

Breaking changes add a `!` after the type and a `BREAKING CHANGE:` footer.
Keep the summary under 72 chars, imperative mood, no trailing period.

No AI attribution anywhere: no `Co-authored-by` trailers for AI tools, no
"generated with" footers, no mentions of assistants in commit messages, PR
titles or PR descriptions. Attribution trailers land in the contributors graph;
PR text lands in the changelog. Both must name people only.

## Before opening the PR

- [ ] No earlier open PR targets the same issue
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Formatting applied: `black` and `isort`
- [ ] Public API changes documented in the relevant docstring
- [ ] If you touched an example: `pytest -m examples tests/examples` passes
- [ ] If performance-sensitive: benchmarked per [Performance and Benchmarking](performance.md)
- [ ] Commit messages follow Conventional Commits
- [ ] No AI attribution in commits, PR title or PR description

The PR description should say **what** changed and **why**, and reference any
issue it closes (`Closes #123`).
