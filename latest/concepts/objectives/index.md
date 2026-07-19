# Learning Objectives

Fitting (p\_\\theta(x) \\propto e^{-E\_\\theta(x)}) by maximum likelihood gives the gradient

\[ \\nabla\_\\theta , \\mathbb{E}_{x \\sim p_{\\text{data}}}[-\\log p\_\\theta(x)] = \\mathbb{E}_{x \\sim p_{\\text{data}}}[\\nabla\_\\theta E\_\\theta(x)]

- \\mathbb{E}_{x \\sim p_\\theta}[\\nabla\_\\theta E\_\\theta(x)] , \]

a push-down on data and a push-up on model samples. Every objective in `torchebm.losses` is a different answer to the hard part, the expectation under the model itself. The shipped family, generated from the installed package at build time:

```
graph TD
    BaseContrastiveDivergence(["BaseContrastiveDivergence"])
    BaseLoss(["BaseLoss"])
    BaseScoreMatching(["BaseScoreMatching"])
    BaseContrastiveDivergence --> ContrastiveDivergence
    BaseLoss --> BaseContrastiveDivergence
    BaseScoreMatching --> DenoisingScoreMatching
    BaseLoss --> BaseScoreMatching
    BaseLoss --> EnergyMatchingLoss
    BaseLoss --> EquilibriumMatchingLoss
    BaseContrastiveDivergence --> ParallelTemperingCD
    BaseContrastiveDivergence --> PersistentContrastiveDivergence
    BaseScoreMatching --> ScoreMatching
    BaseScoreMatching --> SlicedScoreMatching
```

## MCMC-based: contrastive divergence

CD-k approximates model samples with k steps of MCMC started at the data[1](#fn:1):

```python
from torchebm.losses import ContrastiveDivergence
cd = ContrastiveDivergence(model=energy, sampler=langevin, k_steps=10)
loss, negatives = cd(batch)
```

`persistent=True` switches to PCD: negatives resume from a replay buffer instead of restarting at the data, so chains explore the model distribution across updates. `ParallelTemperingCD` runs chains at several temperatures and swaps them, for multimodal targets where single-temperature chains get stuck. CD trains slowly per step (an inner MCMC loop) but yields a genuine energy with meaningful level sets.

## Simulation-free: score matching

Score matching sidesteps model samples entirely by fitting the score[2](#fn:2). The exact objective needs the Hessian trace; the practical members are denoising SM[3](#fn:3), which matches the score of noise-perturbed data,

```python
from torchebm.losses import DenoisingScoreMatching
dsm = DenoisingScoreMatching(model=energy, noise_scale=0.1)
```

and sliced SM, which estimates the trace with random projections. DSM at a ladder of noise scales is the training principle underlying score-based diffusion[4](#fn:4).

## Transport-based: equilibrium and energy matching

The modern objectives are simulation-free: they replace the inner sampling loop with regression along an interpolant path (see [Interpolants and Couplings](https://soran-ghaderi.github.io/torchebm/latest/concepts/transport/index.md)).

**Equilibrium matching** trains a time-invariant field `f(x)` toward the noise direction along the path (`f` points data -> noise), so every route transports noise -> data by moving along `-f`. With the implicit formulation (`energy_type="none"`) `f` *is* the gradient field: integrate it with `FlowSampler(negate_velocity=True)`, or descend it with the `EqMEnergy` adapter, which turns the field into the scalar `BaseModel` the gradient-based samplers and `InteractionModel` consume. The explicit formulation (`energy_type="dot"`) trains the scalar energy `g(x) = x . f(x)` for gradient-descent sampling and OOD scoring. It also accepts a `coupling=` (default identity) and, like energy matching, honors per-pair coupling weights.

```python
from torchebm.losses import EquilibriumMatchingLoss
from torchebm.models import EqMEnergy
from torchebm.samplers import FlowSampler, GradientDescentSampler

eqm = EquilibriumMatchingLoss(model=field, interpolant="linear", energy_type="none")
# ... train ...
ode = FlowSampler(field, negate_velocity=True, integrator="euler")   # integrate -f
gd = GradientDescentSampler(EqMEnergy.from_loss(eqm))                 # descend the energy
```

`EqMEnergy.from_loss` picks the adapter mode matching the trained `energy_type` (implicit vs dot/l2), so the sampled energy always matches what was trained. `InteractionModel` must wrap an explicit energy, never the implicit adapter.

**Energy matching** (arXiv:2504.10612) keeps a single time-independent scalar potential: an OT flow-matching warm-up shapes it as transport, then a contrastive phase with temperature-scheduled Langevin negatives sharpens its Boltzmann density near the data. It accepts a `coupling=` and consumes per-pair weights when the coupling provides them:

```python
from torchebm.couplings import SinkhornCoupling
from torchebm.losses import EnergyMatchingLoss
em = EnergyMatchingLoss(model=potential, coupling=SinkhornCoupling(reg=0.01),
                        epsilon_max=0.15, tau_star=0.8)
```

## Choosing an objective

| Objective            | Inner sampling     | Trains            | Generate with                                       | Reach for it when                                         |
| -------------------- | ------------------ | ----------------- | --------------------------------------------------- | --------------------------------------------------------- |
| CD / PCD / PT-CD     | yes (k MCMC steps) | energy            | MCMC                                                | you need a calibrated energy and can afford MCMC per step |
| Exact / sliced SM    | no (Hessian term)  | energy            | Langevin                                            | low dimension, no noise tolerance                         |
| Denoising SM         | no                 | energy (smoothed) | annealed Langevin                                   | fast sampling-free training, noise scale acceptable       |
| Equilibrium matching | no                 | field or energy   | `FlowSampler` ODE or `EqMEnergy` + gradient descent | generative quality with few integration steps             |
| Energy matching      | phase 2 only       | energy            | one Langevin sweep                                  | one potential for both transport and Boltzmann sampling   |

The rule of thumb embedded in the table: objectives with inner sampling buy energy fidelity at training cost; transport objectives buy training scalability and fast generation, and the hybrids exist to keep the energy while paying the transport price.

## Runnable counterparts

- [CD-k on Two Moons](https://soran-ghaderi.github.io/torchebm/latest/examples/20-training/01-mcmc-losses/01-cd-k/index.md)
- [Persistent CD](https://soran-ghaderi.github.io/torchebm/latest/examples/20-training/01-mcmc-losses/02-persistent-cd/index.md)
- [Denoising Score Matching](https://soran-ghaderi.github.io/torchebm/latest/examples/20-training/02-score-matching/01-denoising-score-matching/index.md)
- [Equilibrium Matching in 2D](https://soran-ghaderi.github.io/torchebm/latest/examples/20-training/03-equilibrium-matching/01-equilibrium-matching-2d/index.md)
- [Energy Matching in 2D](https://soran-ghaderi.github.io/torchebm/latest/examples/20-training/04-energy-matching/01-energy-matching-2d/index.md)

______________________________________________________________________

1. G. E. Hinton. Training products of experts by minimizing contrastive divergence. *Neural Computation*, 14(8), 2002. [↩](#fnref:1 "Jump back to footnote 1 in the text")
1. A. Hyvärinen. Estimation of non-normalized statistical models by score matching. *JMLR*, 6, 2005. [↩](#fnref:2 "Jump back to footnote 2 in the text")
1. P. Vincent. A connection between score matching and denoising autoencoders. *Neural Computation*, 23(7), 2011. [↩](#fnref:3 "Jump back to footnote 3 in the text")
1. Y. Song and S. Ermon. Generative modeling by estimating gradients of the data distribution. *NeurIPS*, 2019. [↩](#fnref:4 "Jump back to footnote 4 in the text")
