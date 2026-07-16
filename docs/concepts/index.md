---
title: Concepts
description: The conceptual model behind TorchEBM, how energies, paths, couplings, objectives, samplers, and integrators compose.
icon: material/lightbulb-outline
---

# Concepts

TorchEBM treats generative modeling as the composition of a small number of
mathematical objects. Each has one subpackage, one base class, and one page
here:

| Axis | Question it answers | Package | Page |
| --- | --- | --- | --- |
| Energy | What is the model? | `torchebm.core`, `torchebm.models` | [The Energy-Based View](energy_view.md) |
| Dynamics | How are samples drawn? | `torchebm.samplers`, `torchebm.integrators` | [Sampling and Integration](sampling.md) |
| Objective | How is the model fit to data, with or without sampling in the loop? | `torchebm.losses` | [Learning Objectives](objectives.md) |
| Transport | Which path and pairing connect noise and data? | `torchebm.interpolants`, `torchebm.couplings` | [Interpolants and Couplings](transport.md) |

[Design and Scope](design.md) states the unifying abstraction precisely,
separates simulation-free from simulation-based training, and places EBMs,
score-based and diffusion models, flow matching, stochastic interpolants, and
Schrödinger bridges in one taxonomy, with references.

Read in any order; each page links to the runnable
[examples](../examples/index.md) that exercise it.
