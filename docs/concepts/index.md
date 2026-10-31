---
title: Concepts
description: The conceptual model behind TorchEBM, how models (potentials and fields), paths, couplings, objectives, samplers, and integrators compose.
icon: material/lightbulb-outline
---

# Concepts

TorchEBM treats generative modeling as the composition of a small number of
mathematical objects. Each has one subpackage, one base class, and one page
here:

| Axis | Question it answers | Package | Page |
| --- | --- | --- | --- |
| Model | Which potential or field defines the dynamics? | `torchebm.core`, `torchebm.models` | [The Energy-Based View](energy_view.md) for potentials, [Learning Objectives](objectives.md) for fields |
| Dynamics | How are samples drawn? | `torchebm.samplers`, `torchebm.integrators` | [Sampling and Integration](sampling.md) |
| Objective | How is the model fit to data, with or without sampling in the loop? | `torchebm.losses` | [Learning Objectives](objectives.md) |
| Transport | Which path and pairing connect the source and the data? | `torchebm.interpolants`, `torchebm.couplings` | [Interpolants and Couplings](transport.md) |

[Design and Scope](design.md) states the unifying object, a learned dynamical
system on data space, and the axes that place every method family; it
separates simulation-free from simulation-based training and marks which
families are supported and which are on the roadmap, with references.

Read in any order; each page links to the runnable
[examples](../examples/index.md) that exercise it.
