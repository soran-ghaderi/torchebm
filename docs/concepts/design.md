---
title: Design and Scope
description: The framing behind TorchEBM, learned dynamics on data space, the axes that place every method family, the training regimes, and which families are supported today.
icon: material/compass-outline
---

# Design and Scope

TorchEBM is a library for generative models that learn a dynamical system on data space, a process that carries samples to the data distribution. The name comes from energy-based models, which are one case of the framing below.

The consequential split among those compositions is the training regime.
Simulation-free objectives (flow matching, equilibrium matching, denoising
score matching) fit the model by regression, with no sampler inside the loss;
simulation-based objectives (contrastive divergence) run MCMC in the loop.
TorchEBM implements both and leans simulation-free, the regime whose training
cost stays close to supervised learning at scale.

## The unifying object

The unifying object is a Markov process on the state space[^gm]. A model in
TorchEBM describes it in one of two ways.

**A scalar potential.** An energy \(E_\theta\) defines the density

\[
p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z_\theta},
\qquad Z_\theta = \int e^{-E_\theta(x)}\,dx ,
\]

and the drift \(-\nabla_x E_\theta\). Langevin and Hamiltonian dynamics sample
\(p_\theta\) from the gradient alone, so the intractable \(Z_\theta\) never
appears, and score-based objectives fit \(s_\theta(x) = -\nabla_x E_\theta(x)\)
directly[^hyvarinen][^vincent]. A drift derived from a potential is
conservative (curl-free) by construction.

**A vector field.** The network outputs the drift itself: a velocity, a score,
or a noise prediction, optionally conditioned on time. Along a probability
path

\[
x_t = \alpha(t)\,x_1 + \sigma(t)\,x_0,
\qquad x_0 \sim \mathcal{N}(0, I),\; x_1 \sim p_{\text{data}},
\]

with \(x_0\) independent of \(x_1\), the marginal velocity, score, and noise
prediction determine one another through affine relations[^albergo][^song-sde],
which the interpolants implement (`velocity_to_score`, `score_to_velocity`, and
relatives). A field fitted by regression is not constrained to be a gradient.
Going from an energy to a field is differentiation; going back requires the
field to be conservative. For a Gaussian source paired independently with the
data, the exact marginal velocity is the gradient of a time-dependent
potential; for other sources and couplings it need not be. The time-invariant
field of implicit equilibrium matching is not a gradient in general even at
its optimum, because its target averages noise levels with weights that depend
on \(x\).

## The axes

Every method is placed by one choice on each axis. Options in *italics* are on
the roadmap; the rest are supported.

| Axis | Options |
| --- | --- |
| State space | Euclidean \(\mathbb{R}^d\); *manifolds*; *discrete states* |
| Representation | scalar potential; vector field (velocity, score, noise); *two-time flow map \(X_{s,t}\)*; *jump rates* |
| Time | autonomous \(f(x)\); time-conditioned \(f(x, t)\); *two times \((s, t)\)* |
| Noise | deterministic (ODE); diffusion (SDE); *jumps* |
| Endpoints | any source that can be sampled (Gaussian noise, another dataset) to the data, paired by a coupling (independent, greedy, OT, entropic OT, unbalanced, model-induced) |
| Training | regression along a path; MCMC in the loss; *iterative proportional fitting*; *distillation and self-consistency* |
| Generation | transport over a fixed time horizon (ODE or SDE integration); refinement toward attractors or a stationary distribution (MCMC, fixed-point iteration); *a few flow-map jumps* |

Refinement depends on the representation. On a potential, MCMC samples the
Boltzmann density \(p_\theta\), and the noise-free iteration
\(x \leftarrow x - \eta \nabla_x E_\theta(x)\) descends to its minima. On a
field \(f\) that is not a gradient, the iteration \(x \leftarrow x + \eta f(x)\)
moves toward the stable zeros (attractors) of \(f\). It is the explicit Euler
step of the autonomous ODE \(\dot x = f(x)\), so `FlowSampler` and
`GradientDescentSampler` apply the same update to such a field over different
horizons. The stationary distribution of Langevin dynamics with a
non-conservative drift, when it exists, is not of Boltzmann form in general,
so "energy", "minimum", and "Boltzmann density" apply to potentials only.

## The component algebra

Methods factor into orthogonal components, each owned by one subpackage:

| Component | Role | Package |
| --- | --- | --- |
| Model | the potential or field that defines the dynamics (`BaseModel`, `AnalyticModel`, backbones) | `core`, `models` |
| Interpolant | the path \(\alpha(t), \sigma(t)\) between a source and the data | `interpolants` |
| Coupling | which \(x_0\) is paired with which \(x_1\) | `couplings` |
| Objective | the loss fitting the model to data | `losses` |
| Sampler | the dynamics generating samples | `samplers` |
| Integrator | the numerical scheme inside the dynamics | `integrators` |

The list is open: new axes join as the field produces them, under the same
one-contract-per-axis rule. The current composition map, generated from the
installed package at build time:

<!-- torchebm:diagram components -->

The method families then read as rows of one table:

| Family | Representation | Time | Noise | Training | Generation | Status |
| --- | --- | --- | --- | --- | --- | --- |
| MCMC-trained EBMs[^hinton][^du] | potential | autonomous | SDE (Langevin, HMC) | MCMC in the loss (CD, persistent CD) | MCMC | supported |
| Score-based EBMs[^hyvarinen][^vincent][^song-ermon] | potential | autonomous | SDE | score regression (exact, denoising, sliced) | Langevin | supported |
| Denoising diffusion[^ho][^song-sde] | score or noise | \(t\) | SDE or its probability-flow ODE | denoising regression over \(t\) | transport | supported |
| Flow matching, rectified flow[^lipman][^tong][^liu] | velocity | \(t\) | ODE | path regression with OT or reflow couplings | transport | supported |
| Stochastic interpolants[^albergo] | velocity (+ score) | \(t\) | ODE or SDE | path regression | transport | partial: paths with a latent term \(\gamma(t) z\) are on the roadmap |
| Implicit equilibrium matching[^eqm] | vector field | autonomous | ODE | path regression | transport or fixed-point iteration | supported |
| Explicit equilibrium matching (EqM-E)[^eqm], energy matching[^em] | potential | autonomous | ODE or SDE | path regression, optional contrastive phase | refinement (descent, or one annealed Langevin sweep) | supported |
| Schrödinger bridges[^debortoli][^dsbm] | forward and backward drifts, or Schrödinger potentials | \(t\) | SDE with reference noise | iterative proportional fitting, iterative Markovian fitting, or bridge matching | stochastic transport between two distributions | roadmap |
| Flow maps[^cm][^fmm][^meanflow] | two-time map \(X_{s,t}\) or average velocity | \((s, t)\) | ODE | distillation or self-consistency | one to a few jumps | roadmap |
| Discrete diffusion and flow matching[^dfm] | jump rates | \(t\) | jumps | regression on a probability path over states | transport | roadmap |
| Riemannian flow matching[^rfm] | tangent vector field | \(t\) | ODE | path regression on a manifold | transport | roadmap |

For a Brownian reference, the optimal Schrödinger-bridge drift is
\(\sigma^2 \nabla_x \log \varphi_t\), the gradient of a time-dependent
potential; as the reference noise vanishes, the bridge reduces to dynamic
optimal transport, which flow matching with an OT coupling approximates. Flow
maps learn the finite-time map of an ODE; its instantaneous field is the limit
of the map over a vanishing interval, so the vector-field rows are the
infinitesimal case of the flow-map row.

The bottleneck of MCMC-trained EBMs is long-run sampling inside the loss; the
simulation-free families replace it with regression along a path, and EqM-E
and energy matching keep a scalar energy while training by regression. Moving
between rows swaps components in the training loop, and a new method adds an
option on one axis: discrete data adds jump rates, a manifold changes the
state space, a flow map adds a second time.

## Design principles

**Composition in user code.** A training loop is user code over plain
objects (a model, a loss, a sampler, an optimizer), the forward, loss,
backward loop of any PyTorch model: losses consume model outputs
(`loss_fn(model(x), model(x_neg))`), samplers take the model at call time
(`sampler(model, x, n_steps=...)`), and only numerical choices are
constructor arguments (`FlowSampler(integrator=...)`,
`EnergyMatchingLoss(lambda_cd=...)`). `BaseTrainer` is an optional wrapper
around that loop.

**One contract per axis.** A potential is a `BaseModel`, any differentiable
map `(N, d) -> (N,)`, called for its energy, its score, or both as its
`prediction_type` says; a field is any module whose output has the shape of
its input; `BaseInterpolant` is the pair \(\alpha(t), \sigma(t)\) and their
derivatives; `BaseCoupling` returns a `CouplingResult` that unpacks as the pair
`(x0, x1)` and carries optional per-pair weights; integrators expose one
`integrate` contract shared by MCMC and flow sampling. Registries
(`get_integrator`, `get_coupling`, `get_interpolant`) make every choice
addressable by string without hiding the class-based path.

**Vectorization first.** Chains, negatives, and path samples are batch
dimensions of one tensor program, so the number of chains is the batch size.

**Stateless mathematics, schedulable hyperparameters.** Interpolants and
couplings are pure functions of tensors. Anything that legitimately varies
during a run (step sizes, noise scales, temperatures) is a `Schedulable`
parameter accepting a float or a scheduler.

## Scope boundaries

TorchEBM implements primitives and reference recipes. It ships no pretrained
checkpoints, no dataset loaders beyond the 2D synthetic benchmarks, and no
training orchestration.

The roadmap rows above are the methods the current components cannot express
yet. Schrödinger bridges need a probability path with a latent term
\(\gamma(t) z\) and an SDE sampler that consumes a learned velocity and score
together; flow maps need a two-time model call, a sampler that composes jumps,
and losses that differentiate through the network with Jacobian-vector
products; discrete and manifold state spaces need their own paths, models,
and samplers.

[^hinton]: G. E. Hinton. Training products of experts by minimizing contrastive divergence. *Neural Computation*, 14(8), 2002.
[^hyvarinen]: A. Hyvärinen. Estimation of non-normalized statistical models by score matching. *JMLR*, 6, 2005.
[^vincent]: P. Vincent. A connection between score matching and denoising autoencoders. *Neural Computation*, 23(7), 2011.
[^song-ermon]: Y. Song and S. Ermon. Generative modeling by estimating gradients of the data distribution. *NeurIPS*, 2019.
[^du]: Y. Du and I. Mordatch. Implicit generation and modeling with energy-based models. *NeurIPS*, 2019.
[^ho]: J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. *NeurIPS*, 2020.
[^song-sde]: Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. *ICLR*, 2021.
[^lipman]: Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. *ICLR*, 2023.
[^liu]: X. Liu, C. Gong, and Q. Liu. Flow straight and fast: learning to generate and transfer data with rectified flow. *ICLR*, 2023.
[^albergo]: M. S. Albergo, N. M. Boffi, and E. Vanden-Eijnden. Stochastic interpolants: a unifying framework for flows and diffusions. arXiv:2303.08797, 2023.
[^tong]: A. Tong, K. Fatras, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, G. Wolf, and Y. Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. *TMLR*, 2024.
[^eqm]: R. Wang and Y. Du. Equilibrium matching: generative modeling with implicit energy-based models. arXiv:2510.02300, 2025.
[^em]: M. Balcerak, T. Amiranashvili, A. Terpin, S. Shit, L. Bogensperger, S. Kaltenbach, P. Koumoutsakos, and B. Menze. Energy matching: unifying flow matching and energy-based models for generative modeling. arXiv:2504.10612, 2025.
[^debortoli]: V. De Bortoli, J. Thornton, J. Heng, and A. Doucet. Diffusion Schrödinger bridge with applications to score-based generative modeling. *NeurIPS*, 2021.
[^dsbm]: Y. Shi, V. De Bortoli, A. Campbell, and A. Doucet. Diffusion Schrödinger bridge matching. *NeurIPS*, 2023.
[^gm]: P. Holderrieth, M. Havasi, J. Yim, N. Shaul, I. Gat, T. Jaakkola, B. Karrer, R. T. Q. Chen, and Y. Lipman. Generator matching: generative modeling with arbitrary Markov processes. *ICLR*, 2025.
[^cm]: Y. Song, P. Dhariwal, M. Chen, and I. Sutskever. Consistency models. *ICML*, 2023.
[^fmm]: N. M. Boffi, M. S. Albergo, and E. Vanden-Eijnden. Flow map matching. arXiv:2406.07507, 2024.
[^meanflow]: Z. Geng, M. Deng, X. Bai, J. Z. Kolter, and K. He. Mean flows for one-step generative modeling. arXiv:2505.13447, 2025.
[^dfm]: I. Gat, T. Remez, N. Shaul, F. Kreuk, R. T. Q. Chen, G. Synnaeve, Y. Adi, and Y. Lipman. Discrete flow matching. *NeurIPS*, 2024.
[^rfm]: R. T. Q. Chen and Y. Lipman. Flow matching on general geometries. *ICLR*, 2024.
