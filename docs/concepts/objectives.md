---
title: Learning Objectives
description: Contrastive divergence, score matching, and the simulation-free transport objectives, and how to choose among them.
icon: material/target
---

# Learning Objectives

The objectives in `torchebm.losses` fall into two groups. Energy objectives
fit a potential \(E_\theta\) with \(p_\theta(x) \propto e^{-E_\theta(x)}\);
maximum likelihood gives the gradient

\[
\nabla_\theta \, \mathbb{E}_{x \sim p_{\text{data}}}[-\log p_\theta(x)]
= \mathbb{E}_{x \sim p_{\text{data}}}[\nabla_\theta E_\theta(x)]
- \mathbb{E}_{x \sim p_\theta}[\nabla_\theta E_\theta(x)] ,
\]

a push-down on data and a push-up on model samples, and the energy objectives
differ in how they treat the expectation under the model: contrastive
divergence estimates it with MCMC, score matching avoids it by fitting
\(-\nabla_x E_\theta\). Transport objectives fit a model to a probability path
by regression and need no model samples; the model is a velocity, a
time-invariant field, or a potential trained through its gradient (EqM-E,
energy matching). The shipped family, generated from the installed package at
build time:

<!-- torchebm:diagram losses -->

## MCMC-based: contrastive divergence

CD-k approximates model samples with k steps of MCMC started at the data[^1]:

```python
from torchebm.losses import ContrastiveDivergenceLoss
from torchebm.samplers import LangevinDynamics

loss_fn = ContrastiveDivergenceLoss()
sampler = LangevinDynamics(step_size=0.1)
negatives = sampler(model, batch, n_steps=10)
loss = loss_fn(model(batch), model(negatives))
```

Starting the chains from a `ReplayBuffer` instead (`sampler(model,
buffer.sample(n), n_steps=10)`, then `buffer.add(negatives)`) gives PCD:
negatives resume across updates, so chains explore the model distribution
beyond k steps. CD pays for an inner MCMC loop at every update and in exchange
fits the energy by approximate maximum likelihood.

## Simulation-free: score matching

Score matching sidesteps model samples entirely by fitting the score[^2].
The exact objective needs the Hessian trace; the practical members are
denoising SM[^3], which matches the score of noise-perturbed data (the model
is built with `prediction_type="score"`, so its call returns \(-\nabla_x E\)),

```python
from torchebm.losses import DenoisingScoreMatchingLoss

model = MLP(prediction_type="score")
noise = torch.randn_like(batch)
loss = DenoisingScoreMatchingLoss()(model(batch + 0.1 * noise), -noise / 0.1)
```

and sliced SM, which estimates the trace with random projections. DSM at a
ladder of noise scales is the training principle underlying score-based
diffusion[^4].

## Transport-based: flow, equilibrium and energy matching

Transport objectives are simulation-free: the regression target comes from an
interpolant path, so no sampler runs inside the loss (see
[Interpolants and Couplings](transport.md)).

**Flow matching** regresses a time-conditioned velocity field `v(x, t)` onto
the interpolant velocity `u_t` and generates by integrating it forward with
`FlowSampler`. The path objects do the data work in the loop: an interpolant
builds `x_t` and the target `u_t`, a coupling pairs noise with data,
`compute_density_for_timestep_sampling` skews `t`, and a per-sample
`weight=` reweights the loss.

```python
from torchebm.interpolants import LinearInterpolant
from torchebm.losses import FlowMatchingLoss
from torchebm.samplers import FlowSampler

path, loss_fn = LinearInterpolant(), FlowMatchingLoss()
x_t, u_t = path(torch.randn_like(x1), x1, t)
loss = loss_fn(model(x_t, t), u_t)
samples = FlowSampler()(model, torch.randn(64, 2), n_steps=50)
```

**Equilibrium matching** trains a time-invariant model toward `c(t) u_t`, the
path velocity from noise to data scaled by `c(t)`, the truncated decay of the
paper, which vanishes at the data. `FlowSampler` integrates the learned field
over a fixed horizon, and the gradient-descent sampler iterates
\(x \leftarrow x + \eta f(x)\) toward its attractors; both apply the same
update with different step sizes and horizons. With the implicit formulation
the model is a time-invariant velocity field (a model that declares
`prediction_type="velocity"`, such as `DiT`, needs the same on the gradient
samplers), and it is not
constrained to be the gradient of an energy; its population optimum is not
one in general, because the target averages noise levels with weights that
depend on \(x\). With the explicit one (EqM-E) the model is a scalar potential
called with `prediction_type="score"`, so its field is a gradient by
construction; `InteractionModel` wraps its energy for diverse sampling, and
the energy scores OOD points directly.

```python
from torchebm.losses import EquilibriumMatchingLoss
from torchebm.samplers import FlowSampler, GradientDescentSampler

x_t, u_t = path(torch.randn_like(x1), x1, t)
loss = EquilibriumMatchingLoss()(model(x_t), u_t, t)
samples_ode = FlowSampler(integrator="euler")(model, x0, n_steps=100)
samples_gd = GradientDescentSampler(step_size=0.05)(model, x0, n_steps=200)
```

The two losses differ only in the scaling of the regression target and in the
clock the model sees:

| Loss | Target | Clock shown to the model | Sample with |
| --- | --- | --- | --- |
| `FlowMatchingLoss` | `u_t` (noise -> data) | `t` | `FlowSampler` |
| `EquilibriumMatchingLoss` | `c(t) u_t` | none (time-invariant) | `FlowSampler`, fixed-point iteration with `GradientDescentSampler` |

With `ct="constant"` and `ct_multiplier=1` the EqM objective is bit-identical
to `FlowMatchingLoss`.

**Energy matching** (arXiv:2504.10612) keeps a single time-independent scalar
potential: an OT flow-matching warm-up shapes it as transport, then a
contrastive phase with temperature-scheduled Langevin negatives sharpens its
Boltzmann density near the data. The potential returns its energy and score
from one call; the warm-up calls `loss_fn(score, u_t, t)` alone, the joint
phase adds the energies of data and Langevin negatives. The coupling pairs
the batch in the loop and `weight=` takes its per-pair weights:

```python
from torchebm.couplings import SinkhornCoupling
from torchebm.losses import EnergyMatchingLoss

model = MLP(prediction_type=("energy", "score"))
loss_fn = EnergyMatchingLoss(lambda_cd=2.0)
x0, x1 = SinkhornCoupling(reg=0.01)(torch.randn_like(x1), x1)
x_t, u_t = path(x0, x1, t)
_, score = model(x_t)
energy_pos, _ = model(x1)
energy_neg, _ = model(negatives)
loss = loss_fn(score, u_t, t, energy_pos, energy_neg)
```

## Choosing an objective

| Objective | Inner sampling | Trains | Generate with | Reach for it when |
| --- | --- | --- | --- | --- |
| CD / PCD / PT-CD | yes (k MCMC steps) | energy | MCMC | you need a calibrated energy and can afford MCMC per step |
| Exact / sliced SM | no (Hessian term) | energy | Langevin | low dimension, no noise tolerance |
| Denoising SM | no | energy (smoothed) | annealed Langevin | fast sampling-free training, noise scale acceptable |
| Flow matching | no | velocity field | `FlowSampler` ODE/SDE | pure generative transport, standard diffusion-style recipes |
| Equilibrium matching | no | field (implicit) or energy (EqM-E) | `FlowSampler` ODE, or iteration of the field | generative quality with few integration steps |
| Energy matching | phase 2 only | energy | one Langevin sweep | one potential for both transport and Boltzmann sampling |

Objectives with inner sampling fit an energy by approximate maximum likelihood
at the cost of MCMC at every update. Transport objectives train by regression
and generate in a fixed number of steps; equilibrium and energy matching
apply that regime to time-invariant models, and EqM-E and energy matching keep
a scalar energy.

## Runnable counterparts

- [CD-k on Two Moons](../examples/20-training/01-mcmc-losses/01-cd-k.md)
- [Persistent CD](../examples/20-training/01-mcmc-losses/02-persistent-cd.md)
- [Denoising Score Matching](../examples/20-training/02-score-matching/01-denoising-score-matching.md)
- [Equilibrium Matching in 2D](../examples/20-training/03-equilibrium-matching/01-equilibrium-matching-2d.md)
- [Energy Matching in 2D](../examples/20-training/04-energy-matching/01-energy-matching-2d.md)

[^1]: G. E. Hinton. Training products of experts by minimizing contrastive divergence. *Neural Computation*, 14(8), 2002.
[^2]: A. Hyvärinen. Estimation of non-normalized statistical models by score matching. *JMLR*, 6, 2005.
[^3]: P. Vincent. A connection between score matching and denoising autoencoders. *Neural Computation*, 23(7), 2011.
[^4]: Y. Song and S. Ermon. Generative modeling by estimating gradients of the data distribution. *NeurIPS*, 2019.
