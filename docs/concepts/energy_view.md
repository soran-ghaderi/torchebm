---
title: The Energy-Based View
description: Energies define densities; forces and scores follow by differentiation; analytic and neural energies share one contract.
icon: material/terrain
---

# The Energy-Based View

An energy function \(E_\theta : \mathbb{R}^d \to \mathbb{R}\) defines a
probability density

\[
p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z_\theta},
\]

where lower energy means higher probability. The normalizer \(Z_\theta\) is
intractable in general, but differences of energies, gradients of energies,
and ratios of densities do not depend on it.

For a model parameterized by an energy, two derived quantities do most of the
work:

- the **score** \(s_\theta(x) = \nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)\),
  which is what score-matching objectives estimate, and
- the **force** \(-\nabla_x E_\theta(x)\) (the same vector), which is the drift
  every gradient-based sampler follows.

Models that output a vector field directly, such as flow-matching velocities
and the implicit equilibrium-matching field, have no energy in general;
[Design and Scope](design.md) places both kinds of model.

## The model contract

Any differentiable map from a batch `(N, d)` to energies `(N,)` is a valid
model. `forward` returns the energy; the call returns what `prediction_type`
names: the energy (default), the score \(-\nabla_x E\) through autograd, or
both from one pass. `BaseModel` adds device and dtype handling and the
integration with every sampler and loss:

```python
import torch
from torch import nn
from torchebm.core import BaseModel

class MLP(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = MLP(prediction_type=("energy", "score"))
energy, score = model(torch.randn(64, 2))
```

A model declares only what its training loss consumes; samplers read the
attribute and differentiate an energy themselves.

Architecture guidance is the usual for EBMs: smooth activations (SiLU, GELU)
keep the gradient field well behaved, and the output must remain a raw scalar
per point (no final nonlinearity). For image-scale conditional models the
library ships `DiT` (with the standard size presets `dit_s_2` through
`dit_xl_8`) and its components in `torchebm.models`.

## Analytic energies

`torchebm.core` provides closed-form landscapes used throughout the examples,
tests, and benchmarks (e.g. `GaussianModel`, `DoubleWellModel`). They make
sampler behavior measurable, since means, covariances, and mode structure are
known exactly. The shipped set, generated from the installed package:

<!-- torchebm:diagram energies -->

```python
from torchebm.core import GaussianModel
model = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))
```

They subclass `AnalyticModel`, so `prediction_type="score"` returns the
closed-form score without autograd.

## Scheduled parameters

Hyperparameters that should vary over a run (a step size annealed during
sampling, a temperature ramp inside a loss) are `Schedulable`: any such
parameter accepts a float or a `BaseScheduler` (linear, cosine,
exponential-decay, warmup, and temperature schedules among the shipped ones),
and the consuming component advances it once per step.

```python
from torchebm.core import CosineScheduler
from torchebm.samplers import LangevinDynamics

sampler = LangevinDynamics(
    step_size=CosineScheduler(start_value=0.05, end_value=0.005, n_steps=500),
)
samples = sampler(model, torch.randn(2000, 2), n_steps=500)
```

## Runnable counterparts

- [Energy Landscapes](../examples/00-foundations/01-energy/01-energy-landscapes.md)
- [Custom Energies](../examples/00-foundations/01-energy/02-custom-energy.md)
- [Scheduler Anatomy](../examples/00-foundations/03-schedulers/01-scheduler-anatomy.md)
