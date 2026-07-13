---
title: Frequently Asked Questions
description: Common questions about TorchEBM's scope, components, and usage.
icon: material/help-circle-outline
---

# Frequently Asked Questions

## What is TorchEBM, and what is it not?

TorchEBM is a PyTorch library of composable primitives for energy-based
generative modeling: energies, samplers, losses, interpolants, couplings,
integrators, and schedulers, spanning EBMs, diffusion, flow matching,
stochastic interpolants, and (roadmap) Schrödinger bridges. It is not a model
zoo or a training framework: there are no pretrained checkpoints, no dataset loaders beyond 2D
synthetic benchmarks, and no `Trainer` class. See
[Design and Scope](../concepts/design.md).

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

## How do I define a custom energy?

Subclass `BaseModel` with any differentiable `forward` mapping `(N, d)` to
`(N,)`; gradients come from autograd via `.gradient()`. See
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

Not yet. Bridges are on the roadmap; the components they require (couplings,
interpolants, SDE integrators) are already maintained, and the design page
records the gap explicitly.

## How do I cite TorchEBM?

```bibtex
@misc{torchebm_library_2025,
  author       = {Ghaderi, Soran and Contributors},
  title        = {TorchEBM: A PyTorch Library for Training Energy-Based Models},
  year         = {2025},
  url          = {https://github.com/soran-ghaderi/torchebm},
}
```

## How do I contribute?

Start with the [Developer Guide](../developer_guide/index.md): environment setup,
code guidelines, and the PR workflow. Adding an example is a good first
contribution; the recipe is in the
[examples overview](../examples/index.md).

Questions not covered here:
[open an issue](https://github.com/soran-ghaderi/torchebm/issues).
