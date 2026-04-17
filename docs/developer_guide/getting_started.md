---
title: Getting Started
description: Set up, make changes, and open a PR against TorchEBM
icon: material/rocket-launch
---

# Getting Started

This page takes you from a fresh clone to a merged pull request.

---

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

---

## Workflow

1. **Branch** from `main` with a descriptive name.
   ```bash
   git checkout -b feat/adaptive-step-size
   ```
2. **Code**: follow the [Code Guidelines](code_guidelines.md). Mirror the package layout under `tests/`.
3. **Test & format** before every commit.
   ```bash
   black torchebm/ tests/
   isort torchebm/ tests/
   pytest tests/ -v
   ```
4. **Commit** using Conventional Commits (see below).
5. **Push & open a PR** against `main`. Link any related issue.

---

## Commit conventions (mandatory)

TorchEBM uses [Conventional Commits](https://www.conventionalcommits.org/). The format is:

```
<type>(<optional scope>): <summary>

<optional body>

<optional footer>
```

| Type       | Use for                                                   |
|------------|-----------------------------------------------------------|
| `feat`     | A new user-facing feature                                 |
| `fix`      | A bug fix                                                 |
| `perf`     | A change that improves performance                        |
| `refactor` | Code change that is neither a feature nor a fix           |
| `test`     | Adding or fixing tests                                    |
| `docs`     | Documentation only                                        |
| `style`    | Formatting, whitespace, no logic change                   |
| `build`    | Build system, packaging                                   |
| `ci`       | CI configuration                                          |
| `chore`    | Other maintenance (deps, tooling)                         |

**Examples**

```text
feat(samplers): add adaptive step size to LangevinDynamics
fix(losses): correct gradient sign in EquilibriumMatching
perf(integrators): cache RK buffers on device once per integrate()
docs(developer_guide): tighten profiling page
```

Breaking changes add a `!` after the type and a `BREAKING CHANGE:` footer:

```text
refactor(core)!: rename BaseSampler.step -> BaseSampler.step_one

BREAKING CHANGE: external subclasses must rename the overridden method.
```

Keep the summary under 72 chars, imperative mood, no trailing period.

---

## Before opening the PR

- [ ] Tests pass: `pytest tests/ -v`
- [ ] Formatting applied: `black` and `isort`
- [ ] Public API changes documented in the relevant docstring
- [ ] If performance-sensitive: benchmarked per [Benchmarking](benchmarking.md); profiled only if the change is non-trivial (see [Profiling](profiling.md))
- [ ] Commit messages follow Conventional Commits

The PR description should say **what** changed and **why**, and reference any issue it closes (`Closes #123`).
