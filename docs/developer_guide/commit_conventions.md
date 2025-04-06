---
sidebar_position: 2
title: Commit Message Conventions
description: Guidelines for writing standardized commit messages
---

# Commit Message Conventions

TorchEBM follows a specific format for commit messages to maintain a clear project history and generate meaningful changelogs. This document outlines the conventions that all contributors should follow when making commits to the project.

## Format

Each commit message should have a specific format:

1. The first line should be a maximum of 50-60 characters
2. It should begin with an emoji followed by a type, then a colon and a brief description
3. Any further details should be in the subsequent lines, separated by an empty line

## Types and Emojis

We use the following types and emojis to categorize commits:

| Type | Emoji | Description |
|------|-------|-------------|
| **feat** | ✨ | Introduces a new feature |
| **fix** | 🐛 | Patches a bug in the codebase |
| **docs** | 📖 | Changes related to documentation |
| **style** | 💎 | Changes that do not affect the meaning of the code (formatting) |
| **refactor** | 📦 | Code changes that neither fix a bug nor add a feature |
| **perf** | 🚀 | Improvements to performance |
| **test** | 🚨 | Adding or correcting tests |
| **build** | 👷 | Changes affecting the build system or external dependencies |
| **ci** | 💻 | Changes to Continuous Integration configuration |
| **chore** | 🎫 | Miscellaneous changes that don't modify source or test files |
| **revert** | 🔙 | Reverts a previous commit |

## Examples

```
✨ feat: add Hamiltonian Monte Carlo sampler
```

```
🐛 fix: correct gradient calculation in Langevin dynamics

This fixes an issue where gradients were not being properly scaled
by the step size, leading to instability in long sampling chains.
```

```
📖 docs: improve installation instructions

Update pip installation command and add conda installation option.
```

## Version Bumping and Releasing

!!! note "For Maintainers"
    Version bumping and release tags are primarily for project maintainers. As a contributor, you don't need to worry about these when submitting pull requests. Project maintainers will handle versioning and releases.

For project maintainers, our CI/CD workflow supports the following tags:

- Use `#major` for breaking changes requiring a major version bump (e.g., 1.0.0 to 2.0.0)
- Use `#minor` for new features requiring a minor version bump (e.g., 1.0.0 to 1.1.0)
- Default is patch level for bug fixes (e.g., 1.0.0 to 1.0.1)
- Include `#release` to trigger a release to PyPI

Example (for maintainers):
```
✨ feat: add comprehensive API for custom energy functions #minor #release
```

## Best Practices

1. **Be descriptive** but concise in your commit message
2. **Focus on the why**, not just the what
3. **Use present tense** ("add feature" not "added feature")
4. **Separate commits logically** - one commit per logical change
5. **Reference issues** in commit messages when appropriate (e.g., "Fixes #123")

Following these conventions helps maintain a clean project history and facilitates automated changelog generation. 