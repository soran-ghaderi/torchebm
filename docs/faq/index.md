---
title: Frequently Asked Questions
description: Common questions about TorchEBM's scope, components, and usage.
icon: material/help-circle-outline
---

# Frequently Asked Questions

## What is TorchEBM, and what is it not?

TorchEBM is a PyTorch library of composable primitives for generative models
that learn dynamics on data space: energies and fields, samplers, losses,
interpolants, couplings, integrators, and schedulers. It covers EBMs,
diffusion, flow matching, and equilibrium and energy matching; Schrödinger
bridges, flow maps, and discrete or manifold state spaces are on the roadmap.
It ships no
pretrained checkpoints and no dataset loaders beyond the 2D synthetic
benchmarks, and `BaseTrainer` is a thin wrapper around the training loop. See
[Design and Scope](../concepts/design.md).

## Is every model in TorchEBM an energy-based model?

No. Energy-based models are the case where the model is a scalar potential and
samples come from its gradient. Flow-matching velocities, diffusion scores and
the implicit equilibrium-matching field are vector fields that need not be
gradients of any energy. The library takes its name from EBMs; the
[Design and Scope](../concepts/design.md) page places both kinds of model.

## How does it relate to diffusers, torchdiffeq, or POT?

`diffusers` ships production diffusion pipelines around pretrained
checkpoints; TorchEBM ships the mathematical components those methods are made
of, for research on the methods themselves. `torchdiffeq` provides general ODE
solvers; TorchEBM implements its own integrator layer (fixed-step, adaptive,
implicit, symplectic) shared by MCMC and flow sampling, with no external
solver dependency. `POT` is a general OT toolbox; TorchEBM likewise implements
its own coupling layer, the pairing rules its transport-based objectives
consume, shared across the library with no external OT dependency.

## Which training objective should I use?

The decision table in [Learning Objectives](../concepts/objectives.md) is the
full answer. In brief: contrastive divergence when you need a calibrated
energy and can afford MCMC inside the loss; denoising score matching for
sampling-free EBM training; equilibrium matching for generative quality with
few integration steps; energy matching when one time-independent potential
should serve both transport and Boltzmann sampling.

## How do samplers and integrators compose?

Every sampler accepts `integrator=` as a registry name or instance:
`LangevinDynamics` and HMC use it for their dynamics, `FlowSampler` for
ODE/SDE generation. `get_integrator("rk4")` constructs one directly. The
[integrator comparison](../examples/10-sampling/02-integrators/01-integrator-comparison.md)
example measures the accuracy orders.

## Why do I see deprecation warnings about model-bound losses and samplers?

TorchEBM moved to the standard PyTorch loop: losses consume model outputs
and samplers take the model at call time. The old forms keep working and
warn once per process.

| Deprecated | Use instead |
| --- | --- |
| `ContrastiveDivergence(model, sampler)`, `PersistentContrastiveDivergence` | `ContrastiveDivergenceLoss()(model(x), model(x_neg))` with `x_neg = sampler(model, x0, n_steps=k)`; a `ReplayBuffer` holds persistent chains |
| `FlowMatchingLoss(model=...)`, `EquilibriumMatchingLoss(model=...)`, `EnergyMatchingLoss(model=...)` | the same classes without `model`: `x_t, u_t = path(x0, x1, t)` then `loss_fn(model(x_t, t), u_t)` |
| `ScoreMatching`, `DenoisingScoreMatching`, `SlicedScoreMatching` | `ScoreMatchingLoss`, `DenoisingScoreMatchingLoss`, `SlicedScoreMatchingLoss` on the output of a `prediction_type="score"` model |
| `LangevinDynamics(model=...)`, `HamiltonianMonteCarlo(model=...)`, `FlowSampler(model=...)` and `sampler.sample(x=...)` | `sampler(model, x, n_steps=...)` |
| `FlowSampler(prediction=, negate_velocity=, train_eps=)` | `prediction_type=` on the model or the sampler; trained fields point from noise to data |
| `EqMEnergy` | a `prediction_type="score"` model sampled directly |
| `model.gradient(x)` | `model(x)` on a model built with `prediction_type="score"`, which returns \(-\nabla_x E\); a closed-form gradient goes in `AnalyticModel._score` |
| `ContrastiveDivergenceTrainer` | `BaseTrainer(model, optimizer, ContrastiveDivergenceLoss(), sampler=..., buffer=...)` |
| `ReflowCoupling(flow)` | `x1 = sampler(model, x0, n_steps=...)` in the training loop |
| `EMA(model, decay=d)` | `torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(d))` with `ema.update_parameters(model)`; sample from `ema.module.eval()` |

The deprecated paths are removed in release 0.12.0, and not before
2027-01-29; `tests/test_deprecations.py` enforces the window. The scripts in
`examples/20-training` show each loop in full.

## How do I define a custom energy?

Subclass `BaseModel` with any differentiable `forward` mapping `(N, d)` to
`(N,)`; built with `prediction_type="score"` or `("energy", "score")`, the
call returns the autograd score. See
[The Energy-Based View](../concepts/energy_view.md) and the
[custom energies example](../examples/00-foundations/01-energy/02-custom-energy.md).

## Do I need a GPU?

No. Every example and test runs on CPU; the library is device-agnostic and
components accept a `device=` argument. A GPU matters for training neural
energies and for large chain counts, where vectorized sampling scales
linearly.

## Why do my Langevin samples diverge or collapse?

The usual causes are a step size too large for the energy's curvature
(diverges) or a noise scale of zero (collapses to a mode; that is gradient
descent). Anneal with a scheduler, as in the
[scheduler anatomy example](../examples/00-foundations/03-schedulers/01-scheduler-anatomy.md),
and prefer smooth activations (SiLU, GELU) in neural energies so the force
field stays well behaved.

## How are the examples kept correct?

Every example folder is executed by CI as a smoke test
(`pytest -m examples tests/examples`) with `TORCHEBM_SMOKE=1` shrinking
iteration counts. If an API change breaks an example, the build fails.

## Where do benchmarks live?

On the external dashboard at
[soran-ghaderi.github.io/torchebm-benchmarks](https://soran-ghaderi.github.io/torchebm-benchmarks/),
generated from `benchmarks/` runs in this repository. The
[performance guide](../developer_guide/performance.md) describes the workflow.

## Are Schrödinger bridges supported?

Not yet. Bridge matching needs a probability path with a latent term
\(\gamma(t) z\), which the interpolants do not have yet, and an SDE sampler that
consumes a learned velocity and score together; iterative proportional
fitting also needs simulation inside the training loop. The couplings
(including entropic OT) and the SDE integrators already exist, and the
[design page](../concepts/design.md#scope-boundaries) lists the gap.

## How do I cite TorchEBM?

Use the BibTeX entry on the [home page](../index.md#citation). GitHub also
serves it from `CITATION.cff` through the "Cite this repository" button, which
exports BibTeX and APA.

## How do I contribute?

Start with the [Developer Guide](../developer_guide/index.md): environment setup,
code guidelines, and the PR workflow. Adding an example is a good first
contribution; the recipe is in the
[examples overview](../examples/index.md).

Questions not covered here:
[open an issue](https://github.com/soran-ghaderi/torchebm/issues).
