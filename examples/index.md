# Examples

A runnable curriculum. Every example is one folder teaching one idea, it runs
with nothing but `pip install torchebm`, and CI executes all of them on every
commit, so nothing here can drift from the library.

Each page in this section is generated from the folder it documents: the code
you read is the code that runs.

## The four tiers

Work through them in order, or jump to the one that matches your goal. Each tier
has a companion chapter in [Concepts](../concepts/index.md) explaining the theory
the examples exercise.

| Tier | You learn to | Theory |
| --- | --- | --- |
| [Foundations](00-foundations/index.md) | build the static objects: energies, datasets, schedulers, interpolants | [The Energy-Based View](../concepts/energy_view.md) |
| [Sampling](10-sampling/index.md) | draw samples from a *fixed* target: MCMC, parallel chains, integrators, flows | [Sampling and Integration](../concepts/sampling.md) |
| [Training](20-training/index.md) | *learn* a target from data: CD, score matching, equilibrium and energy matching, couplings | [Learning Objectives](../concepts/objectives.md) and [Interpolants and Couplings](../concepts/transport.md) |
| [Showcase](90-showcase/index.md) | see the components pushed end to end | [Design and Scope](../concepts/design.md) |

## Where to start

| If you want to | Start with |
| --- | --- |
| understand what an energy *is* | Foundations: Energy Landscapes |
| sample a distribution you already have | Sampling: Langevin Dynamics 101 |
| train an energy-based model on your data | Training: CD-k on Two Moons |
| build a fast generative model | Training: Equilibrium Matching in 2D |
| see why the pairing of noise and data matters | Training: Coupling Comparison |
| choose a numerical integrator | Sampling: Integrator Comparison |

## Anatomy of an example

```
<slug>/
  main.py     THE lesson: torchebm + the standard library, linear, top to bottom
  meta.yaml   title / summary / order / difficulty / tags / ci
  index.md    OPTIONAL: prose shown above the code on the website
  plot.py     OPTIONAL: renders figures locally (needs matplotlib)
```

**`main.py` is the lesson.** It imports only `torchebm` and the standard library,
so it runs on a bare install. No `argparse`, no `def main()`, no device flags: a
few constants at the top, then a linear body that ends by printing the key result
(a shape, a metric, a recovered statistic). It should read like a docstring you
can run.

**Smoke mode.** Every `main.py` honours `TORCHEBM_SMOKE=1` by shrinking its
iteration counts so CI can execute it in seconds:

```python
SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 500
```

**Visualization is decoupled.** Pages here are text: prose and code. Plotting is
not part of the lesson, so where a picture genuinely helps, the folder carries a
`plot.py` that writes into a local, gitignored `assets/` folder. Nothing is
committed or embedded.

## `meta.yaml`

The single source of an example's metadata, and the discovery signal: a folder
with a `meta.yaml` *is* an example.

```yaml
title: "Langevin Dynamics 101"
summary: "Sample a 2D energy with Langevin; trade step size against noise."
order: 400            # position in the global learning path; bands:
                      #   0-399 Foundations, 400-799 Sampling,
                      #   800-1199 Training, 1200+ Showcase
difficulty: intro     # intro | intermediate | advanced
tags: [langevin, sampling, mcmc, 2d-energy]
entrypoint: main.py
ci:                   # optional
  timeout: 120        # seconds (default 120)
  # skip: true        # heavy showcases only
```

## Adding an example

1. Create a folder under the right tier, with a numeric prefix.
2. Write `main.py`: pure torchebm, prints its result, honours `TORCHEBM_SMOKE`.
3. Add `meta.yaml`, choosing an `order` inside the tier's band.
4. Optionally add `index.md` (prose) and `plot.py` (figures).
5. Verify:
   ```bash
   TORCHEBM_SMOKE=1 python examples/<tier>/<section>/<slug>/main.py
   pytest -m examples tests/examples
   mkdocs serve
   ```

You never write a docs page: the nav entry, the tier table, and the page itself
are all generated from the folder.

## Tags

- capability: `langevin`, `hmc`, `score-matching`, `cd`, `eqm`, `energy-matching`, `flow`, `coupling`
- surface: `2d-energy`, `image`, `interpolant`, `integrator`, `scheduler`, `dataset`
- trait: `gpu`, `animation`, `comparison`, `showcase`
