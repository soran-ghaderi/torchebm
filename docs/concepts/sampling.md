---
title: Sampling and Integration
description: From MCMC on a fixed energy to ODE/SDE integration of learned fields, and the shared numerical layer underneath.
icon: material/chart-scatter-plot
---

# Sampling and Integration

A sampler turns a model into samples. The samplers range from Markov chain
methods on a fixed energy to deterministic integration of a learned velocity
field, and all of them step through a shared integrator layer. The current
family, generated from the installed package at build time:

<!-- torchebm:diagram samplers -->

## MCMC on an energy

**Langevin dynamics** descends the energy gradient (equivalently, ascends the
model score \(-\nabla_x E\)) with injected noise,

\[
x_{k+1} = x_k - \eta\, \nabla_x E(x_k) + \sqrt{2\eta}\; \varepsilon_k,
\qquad \varepsilon_k \sim \mathcal{N}(0, I),
\]

and is the workhorse behind contrastive-divergence negatives and annealed
generation:

```python
from torchebm.samplers import LangevinDynamics
sampler = LangevinDynamics(step_size=0.02, noise_scale=1.0)
samples = sampler(model, torch.randn(2000, 2), n_steps=500)
```

**Hamiltonian Monte Carlo** augments the state with momentum and integrates
Hamiltonian dynamics with a symplectic leapfrog before a Metropolis
accept/reject, trading gradient evaluations for long, decorrelated proposals.
**Riemannian manifold HMC** replaces the flat metric with a position-dependent
one and integrates the resulting non-separable Hamiltonian with a generalised
(implicit) leapfrog; it is the geometry-aware option for ill-conditioned
targets.

```python
from torchebm.samplers import HamiltonianMonteCarlo
sampler = HamiltonianMonteCarlo(step_size=0.1, n_leapfrog_steps=10)
samples = sampler(model, torch.randn(1000, 2), n_steps=300)
```

Optimization-based samplers (plain gradient descent and momentum variants such
as `NesterovSampler`) are the noise-free members of the family. On an energy
they descend to its minima; on a field that is not a gradient, such as an
implicit equilibrium-matching model, the same update
\(x \leftarrow x + \eta f(x)\) is a fixed-point iteration toward the field's
attractors.

Four conventions hold across all of them:

- **The model is an argument of the call.** `sampler(model, x, n_steps=...)`,
  as the function is for `torchdiffeq`; the sampler reads
  `model.prediction_type` and differentiates an energy into the score it
  consumes, so one sampler serves analytic energies, trained energies and
  score models alike.
- **Chains are a batch dimension.** The rows of `x` advance as one tensor
  program, so the number of chains is the batch size.
- **Trajectories are tensors.** `return_trajectory=True` returns the state
  after every step, shape `(batch, n_steps, ...)`, for diagnostics and
  animation.
- **Hyperparameters are schedulable.** Step sizes and noise scales accept
  schedulers, which is how annealed Langevin and temperature sweeps are
  expressed; a `Schedulable` model such as `InteractionModel` is stepped in
  lockstep.

## Continuous-time generation

For trained flow, diffusion, and equilibrium models, `FlowSampler` integrates
the learned field from noise to data:

```python
from torchebm.samplers import FlowSampler
sampler = FlowSampler(interpolant="linear", prediction_type="velocity",
                      integrator="heun")
samples = sampler(model, torch.randn(4096, 2), n_steps=50)
```

`prediction_type` names what the network outputs (`"velocity"`, `"score"`,
or `"epsilon"`); the interpolant converts between them internally, so one
sampler serves flow matching, score-based diffusion, and noise-prediction
models. The process is chosen at construction: `mode="ode"` (default)
integrates the probability-flow ODE, `mode="sde"` adds a diffusion term with
selectable form (`diffusion_form=`, `last_step=`). The call follows the same
contract as every other sampler and also accepts a custom time grid, `t=`,
in place of `n_steps`; with a fixed-step integrator it supports
`return_trajectory`, while adaptive integrators (`dopri5`, ...) return the
final state only. An equilibrium-matching field declares no
`prediction_type` and is integrated as the velocity directly.

## The integrator layer

Both worlds sit on `torchebm.integrators`, which covers the standard solver
families: explicit Runge-Kutta schemes of increasing order (e.g. `heun`,
`rk4`), adaptive step-size solvers (e.g. `dopri5`), implicit schemes for stiff
dynamics, and symplectic integrators for Hamiltonian ones (e.g. `leapfrog`).
The shipped set, generated from the installed package:

<!-- torchebm:diagram integrators -->

Every sampler accepts `integrator=` as a registry name or an instance, and
`get_integrator` constructs one directly:

```python
from torchebm.integrators import get_integrator
integ = get_integrator("rk4")
```

Higher-order solvers are more accurate per step, adaptive solvers choose the
step size from an error estimate, implicit solvers stay stable on stiff
dynamics, and symplectic solvers preserve the phase-space volume that HMC's
correctness depends on. The
[integrator comparison example](../examples/10-sampling/02-integrators/01-integrator-comparison.md)
measures the orders directly against an exact solution.

## Runnable counterparts

- [Langevin Dynamics 101](../examples/10-sampling/01-mcmc/01-langevin-101.md)
- [HMC 101](../examples/10-sampling/01-mcmc/02-hmc-101.md)
- [Parallel Chains](../examples/10-sampling/01-mcmc/03-parallel-chains.md)
- [FlowSampler ODE 101](../examples/10-sampling/03-flow/01-flow-sampler-ode.md)
