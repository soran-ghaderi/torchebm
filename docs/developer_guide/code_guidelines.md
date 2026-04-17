---
title: Code Guidelines
description: Style, API design, and testing conventions for TorchEBM
icon: material/code-braces
---

# Code Guidelines

Short version: **simple, vectorised, consistent with PyTorch**. Full rules below.

---

## Style

- **PEP 8** enforced by `black` (line length 88) and `isort` (profile `black`).
- Run both before every commit:
  ```bash
  black torchebm/ tests/
  isort torchebm/ tests/
  ```
- **Type hints** on every public signature.
- **Google-style docstrings** with raw strings (`r"""..."""`). LaTeX with `\( \)` inline and `\[ \]` for blocks. No Sphinx directives (`:meth:`, `:attr:`).

```python
def sample_chain(
    dim: int,
    n_steps: int,
    n_samples: int = 1,
) -> tuple[torch.Tensor, dict]:
    r"""Run a Markov chain.

    Args:
        dim: Sample dimensionality.
        n_steps: Number of MCMC steps.
        n_samples: Number of parallel chains.

    Returns:
        Final samples and a diagnostics dict.
    """
```

### Naming

- Classes: `CamelCase` (`LangevinDynamics`)
- Functions / variables: `snake_case` (`compute_energy`)
- Constants: `UPPER_CASE` (`DEFAULT_STEP_SIZE`)

---

## API design

- **Inherit** from the matching base class (`BaseLoss`, `BaseSampler`, `BaseModel`, `BaseIntegrator`, `BaseInterpolant`). Do not invent parallel hierarchies.
- **Argument order is part of the API.** Keep it consistent across samplers, losses, and integrators. When in doubt, match the closest existing component.
- **Explicit over implicit**: configuration goes through constructor kwargs, not globals.
- **Compose, don't extend deeply.** Two levels of inheritance is usually the limit.

---

## Performance rules

These are non-negotiable for any code in `torchebm/`:

- Vectorise. No Python loops over batch elements.
- No `.item()`, `.cpu()`, or `.tolist()` inside a hot loop.
- Use `self.device` / `self.dtype` from `DeviceMixin`. Never hard-code `"cuda"`.
- Inputs moved to device in `BaseLoss.__call__`. subclass `forward()` must not re-do it.
- Wrap grad-free regions with `torch.no_grad()`.
- Use `self.autocast_context()` for mixed precision, not bare `torch.autocast`.

See [Performance](performance.md) for patterns and [Profiling](profiling.md) for when to measure.

---

## Testing

- `pytest` for everything. Tests live under `tests/` and mirror the `torchebm/` layout.
- File names: `test_<module>.py`. Function names: `test_<behaviour>`.
- Use `@pytest.fixture` for shared setup and `@pytest.mark.parametrize` for table-driven tests (device, dtype, scale).
- For stochastic code, seed once at the top of the test; mock `torch.randn_like` / `torch.rand` when you need exact equality.
- Verify both **correctness** (finite output, shape, known analytical answer) and **gradient flow** (trainable params receive non-zero grads).

Run the suite:

```bash
pytest tests/ -v                 # all tests
pytest tests/losses -v           # one package
pytest --cov=torchebm            # with coverage
```

Benchmark and profile tests live under `benchmarks/` and are disabled by default. see [Benchmarking](benchmarking.md).

---

## What not to do

- Do not add docstrings, comments, or type hints to code you did not change.
- Do not create one-shot helper modules. inline the operation.
- Do not write markdown files (README, tutorial, guide) unless explicitly asked.
- Do not edit files with shell redirection (`cat >`, `echo >>`, heredocs). Use the editor / file tools.
- Do not introduce flake8 / pylint configs; black + isort + mypy are the configured tools.
