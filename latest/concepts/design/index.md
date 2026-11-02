# Design and Scope

TorchEBM is a library for generative models that learn a dynamical system on data space, a process that carries samples to the data distribution. The name comes from energy-based models, which are one case of the framing below.

The consequential split among those compositions is the training regime. Simulation-free objectives (flow matching, equilibrium matching, denoising score matching) fit the model by regression, with no sampler inside the loss; simulation-based objectives (contrastive divergence) run MCMC in the loop. TorchEBM implements both and leans simulation-free, the regime whose training cost stays close to supervised learning at scale.

## The unifying object

The unifying object is a Markov process on the state space[16](#fn:gm). A model in TorchEBM describes it in one of two ways.

**A scalar potential.** An energy (E\_\\theta) defines the density

[ p\_\\theta(x) = \\frac{e^{-E\_\\theta(x)}}{Z\_\\theta}, \\qquad Z\_\\theta = \\int e^{-E\_\\theta(x)},dx , ]

and the drift (-\\nabla_x E\_\\theta). Langevin and Hamiltonian dynamics sample (p\_\\theta) from the gradient alone, so the intractable (Z\_\\theta) never appears, and score-based objectives fit (s\_\\theta(x) = -\\nabla_x E\_\\theta(x)) directly[2](#fn:hyvarinen)[3](#fn:vincent). A drift derived from a potential is conservative (curl-free) by construction.

**A vector field.** The network outputs the drift itself: a velocity, a score, or a noise prediction, optionally conditioned on time. Along a probability path

[ x_t = \\alpha(t),x_1 + \\sigma(t),x_0, \\qquad x_0 \\sim \\mathcal{N}(0, I),; x_1 \\sim p\_{\\text{data}}, ]

with (x_0) independent of (x_1), the marginal velocity, score, and noise prediction determine one another through affine relations[10](#fn:albergo)[7](#fn:song-sde), which the interpolants implement (`velocity_to_score`, `score_to_velocity`, and relatives). A field fitted by regression is not constrained to be a gradient. Going from an energy to a field is differentiation; going back requires the field to be conservative. For a Gaussian source paired independently with the data, the exact marginal velocity is the gradient of a time-dependent potential; for other sources and couplings it need not be. The time-invariant field of implicit equilibrium matching is not a gradient in general even at its optimum, because its target averages noise levels with weights that depend on (x).

## The axes

Every method is placed by one choice on each axis. Options in *italics* are on the roadmap; the rest are supported.

| Axis           | Options                                                                                                                                                                       |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| State space    | Euclidean (\\mathbb{R}^d); *manifolds*; *discrete states*                                                                                                                     |
| Representation | scalar potential; vector field (velocity, score, noise); *two-time flow map (X\_{s,t})*; *jump rates*                                                                         |
| Time           | autonomous (f(x)); time-conditioned (f(x, t)); *two times ((s, t))*                                                                                                           |
| Noise          | deterministic (ODE); diffusion (SDE); *jumps*                                                                                                                                 |
| Endpoints      | any source that can be sampled (Gaussian noise, another dataset) to the data, paired by a coupling (independent, greedy, OT, entropic OT, unbalanced, model-induced)          |
| Training       | regression along a path; MCMC in the loss; *iterative proportional fitting*; *distillation and self-consistency*                                                              |
| Generation     | transport over a fixed time horizon (ODE or SDE integration); refinement toward attractors or a stationary distribution (MCMC, fixed-point iteration); *a few flow-map jumps* |

Refinement depends on the representation. On a potential, MCMC samples the Boltzmann density (p\_\\theta), and the noise-free iteration (x \\leftarrow x - \\eta \\nabla_x E\_\\theta(x)) descends to its minima. On a field (f) that is not a gradient, the iteration (x \\leftarrow x + \\eta f(x)) moves toward the stable zeros (attractors) of (f). It is the explicit Euler step of the autonomous ODE (\\dot x = f(x)), so `FlowSampler` and `GradientDescentSampler` apply the same update to such a field over different horizons. The stationary distribution of Langevin dynamics with a non-conservative drift, when it exists, is not of Boltzmann form in general, so "energy", "minimum", and "Boltzmann density" apply to potentials only.

## The component algebra

Methods factor into orthogonal components, each owned by one subpackage:

| Component   | Role                                                                                       | Package          |
| ----------- | ------------------------------------------------------------------------------------------ | ---------------- |
| Model       | the potential or field that defines the dynamics (`BaseModel`, `AnalyticModel`, backbones) | `core`, `models` |
| Interpolant | the path (\\alpha(t), \\sigma(t)) between a source and the data                            | `interpolants`   |
| Coupling    | which (x_0) is paired with which (x_1)                                                     | `couplings`      |
| Objective   | the loss fitting the model to data                                                         | `losses`         |
| Sampler     | the dynamics generating samples                                                            | `samplers`       |
| Integrator  | the numerical scheme inside the dynamics                                                   | `integrators`    |

The list is open: new axes join as the field produces them, under the same one-contract-per-axis rule. The current composition map, generated from the installed package at build time:

```
graph LR
    field["energy / field<br/>core: 10 analytic energies · models"]
    interp["interpolants (4)"]
    coup["couplings (7)"]
    integ["integrators (12)"]
    samp["samplers (6)"]
    loss["objectives (12)"]
    data[("datasets (8)")]
    out(("samples"))
    field --> samp
    field --> loss
    interp --> loss
    interp --> samp
    coup --> loss
    integ --> samp
    samp -- negatives --> loss
    data --> loss
    samp --> out
```

The method families then read as rows of one table:

| Family                                                                          | Representation                                         | Time       | Noise                           | Training                                                                        | Generation                                           | Status                                                              |
| ------------------------------------------------------------------------------- | ------------------------------------------------------ | ---------- | ------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------- |
| MCMC-trained EBMs[1](#fn:hinton)[5](#fn:du)                                     | potential                                              | autonomous | SDE (Langevin, HMC)             | MCMC in the loss (CD, persistent CD)                                            | MCMC                                                 | supported                                                           |
| Score-based EBMs[2](#fn:hyvarinen)[3](#fn:vincent)[4](#fn:song-ermon)           | potential                                              | autonomous | SDE                             | score regression (exact, denoising, sliced)                                     | Langevin                                             | supported                                                           |
| Denoising diffusion[6](#fn:ho)[7](#fn:song-sde)                                 | score or noise                                         | (t)        | SDE or its probability-flow ODE | denoising regression over (t)                                                   | transport                                            | supported                                                           |
| Flow matching, rectified flow[8](#fn:lipman)[11](#fn:tong)[9](#fn:liu)          | velocity                                               | (t)        | ODE                             | path regression with OT or reflow couplings                                     | transport                                            | supported                                                           |
| Stochastic interpolants[10](#fn:albergo)                                        | velocity (+ score)                                     | (t)        | ODE or SDE                      | path regression                                                                 | transport                                            | partial: paths with a latent term (\\gamma(t) z) are on the roadmap |
| Implicit equilibrium matching[12](#fn:eqm)                                      | vector field                                           | autonomous | ODE                             | path regression                                                                 | transport or fixed-point iteration                   | supported                                                           |
| Explicit equilibrium matching (EqM-E)[12](#fn:eqm), energy matching[13](#fn:em) | potential                                              | autonomous | ODE or SDE                      | path regression, optional contrastive phase                                     | refinement (descent, or one annealed Langevin sweep) | supported                                                           |
| Schrödinger bridges[14](#fn:debortoli)[15](#fn:dsbm)                            | forward and backward drifts, or Schrödinger potentials | (t)        | SDE with reference noise        | iterative proportional fitting, iterative Markovian fitting, or bridge matching | stochastic transport between two distributions       | roadmap                                                             |
| Flow maps[17](#fn:cm)[18](#fn:fmm)[19](#fn:meanflow)                            | two-time map (X\_{s,t}) or average velocity            | ((s, t))   | ODE                             | distillation or self-consistency                                                | one to a few jumps                                   | roadmap                                                             |
| Discrete diffusion and flow matching[20](#fn:dfm)                               | jump rates                                             | (t)        | jumps                           | regression on a probability path over states                                    | transport                                            | roadmap                                                             |
| Riemannian flow matching[21](#fn:rfm)                                           | tangent vector field                                   | (t)        | ODE                             | path regression on a manifold                                                   | transport                                            | roadmap                                                             |

For a Brownian reference, the optimal Schrödinger-bridge drift is (\\sigma^2 \\nabla_x \\log \\varphi_t), the gradient of a time-dependent potential; as the reference noise vanishes, the bridge reduces to dynamic optimal transport, which flow matching with an OT coupling approximates. Flow maps learn the finite-time map of an ODE; its instantaneous field is the limit of the map over a vanishing interval, so the vector-field rows are the infinitesimal case of the flow-map row.

The bottleneck of MCMC-trained EBMs is long-run sampling inside the loss; the simulation-free families replace it with regression along a path, and EqM-E and energy matching keep a scalar energy while training by regression. Moving between rows swaps components in the training loop, and a new method adds an option on one axis: discrete data adds jump rates, a manifold changes the state space, a flow map adds a second time.

## Design principles

**Composition in user code.** A training loop is user code over plain objects (a model, a loss, a sampler, an optimizer), the forward, loss, backward loop of any PyTorch model: losses consume model outputs (`loss_fn(model(x), model(x_neg))`), samplers take the model at call time (`sampler(model, x, n_steps=...)`), and only numerical choices are constructor arguments (`FlowSampler(integrator=...)`, `EnergyMatchingLoss(lambda_cd=...)`). `BaseTrainer` is an optional wrapper around that loop.

**One contract per axis.** A potential is a `BaseModel`, any differentiable map `(N, d) -> (N,)`, called for its energy, its score, or both as its `prediction_type` says; a field is any module whose output has the shape of its input; `BaseInterpolant` is the pair (\\alpha(t), \\sigma(t)) and their derivatives; `BaseCoupling` returns a `CouplingResult` that unpacks as the pair `(x0, x1)` and carries optional per-pair weights; integrators expose one `integrate` contract shared by MCMC and flow sampling. Registries (`get_integrator`, `get_coupling`, `get_interpolant`) make every choice addressable by string without hiding the class-based path.

**Vectorization first.** Chains, negatives, and path samples are batch dimensions of one tensor program, so the number of chains is the batch size.

**Stateless mathematics, schedulable hyperparameters.** Interpolants and couplings are pure functions of tensors. Anything that legitimately varies during a run (step sizes, noise scales, temperatures) is a `Schedulable` parameter accepting a float or a scheduler.

## Scope boundaries

TorchEBM implements primitives and reference recipes. It ships no pretrained checkpoints, no dataset loaders beyond the 2D synthetic benchmarks, and no training orchestration.

The roadmap rows above are the methods the current components cannot express yet. Schrödinger bridges need a probability path with a latent term (\\gamma(t) z) and an SDE sampler that consumes a learned velocity and score together; flow maps need a two-time model call, a sampler that composes jumps, and losses that differentiate through the network with Jacobian-vector products; discrete and manifold state spaces need their own paths, models, and samplers.

______________________________________________________________________

1. G. E. Hinton. Training products of experts by minimizing contrastive divergence. *Neural Computation*, 14(8), 2002. [↩](#fnref:hinton "Jump back to footnote 1 in the text")
1. A. Hyvärinen. Estimation of non-normalized statistical models by score matching. *JMLR*, 6, 2005. [↩](#fnref:hyvarinen "Jump back to footnote 2 in the text")[↩](#fnref2:hyvarinen "Jump back to footnote 2 in the text")
1. P. Vincent. A connection between score matching and denoising autoencoders. *Neural Computation*, 23(7), 2011. [↩](#fnref:vincent "Jump back to footnote 3 in the text")[↩](#fnref2:vincent "Jump back to footnote 3 in the text")
1. Y. Song and S. Ermon. Generative modeling by estimating gradients of the data distribution. *NeurIPS*, 2019. [↩](#fnref:song-ermon "Jump back to footnote 4 in the text")
1. Y. Du and I. Mordatch. Implicit generation and modeling with energy-based models. *NeurIPS*, 2019. [↩](#fnref:du "Jump back to footnote 5 in the text")
1. J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. *NeurIPS*, 2020. [↩](#fnref:ho "Jump back to footnote 6 in the text")
1. Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. *ICLR*, 2021. [↩](#fnref:song-sde "Jump back to footnote 7 in the text")[↩](#fnref2:song-sde "Jump back to footnote 7 in the text")
1. Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. *ICLR*, 2023. [↩](#fnref:lipman "Jump back to footnote 8 in the text")
1. X. Liu, C. Gong, and Q. Liu. Flow straight and fast: learning to generate and transfer data with rectified flow. *ICLR*, 2023. [↩](#fnref:liu "Jump back to footnote 9 in the text")
1. M. S. Albergo, N. M. Boffi, and E. Vanden-Eijnden. Stochastic interpolants: a unifying framework for flows and diffusions. arXiv:2303.08797, 2023. [↩](#fnref:albergo "Jump back to footnote 10 in the text")[↩](#fnref2:albergo "Jump back to footnote 10 in the text")
1. A. Tong, K. Fatras, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, G. Wolf, and Y. Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. *TMLR*, 2024. [↩](#fnref:tong "Jump back to footnote 11 in the text")
1. R. Wang and Y. Du. Equilibrium matching: generative modeling with implicit energy-based models. arXiv:2510.02300, 2025. [↩](#fnref:eqm "Jump back to footnote 12 in the text")[↩](#fnref2:eqm "Jump back to footnote 12 in the text")
1. M. Balcerak, T. Amiranashvili, A. Terpin, S. Shit, L. Bogensperger, S. Kaltenbach, P. Koumoutsakos, and B. Menze. Energy matching: unifying flow matching and energy-based models for generative modeling. arXiv:2504.10612, 2025. [↩](#fnref:em "Jump back to footnote 13 in the text")
1. V. De Bortoli, J. Thornton, J. Heng, and A. Doucet. Diffusion Schrödinger bridge with applications to score-based generative modeling. *NeurIPS*, 2021. [↩](#fnref:debortoli "Jump back to footnote 14 in the text")
1. Y. Shi, V. De Bortoli, A. Campbell, and A. Doucet. Diffusion Schrödinger bridge matching. *NeurIPS*, 2023. [↩](#fnref:dsbm "Jump back to footnote 15 in the text")
1. P. Holderrieth, M. Havasi, J. Yim, N. Shaul, I. Gat, T. Jaakkola, B. Karrer, R. T. Q. Chen, and Y. Lipman. Generator matching: generative modeling with arbitrary Markov processes. *ICLR*, 2025. [↩](#fnref:gm "Jump back to footnote 16 in the text")
1. Y. Song, P. Dhariwal, M. Chen, and I. Sutskever. Consistency models. *ICML*, 2023. [↩](#fnref:cm "Jump back to footnote 17 in the text")
1. N. M. Boffi, M. S. Albergo, and E. Vanden-Eijnden. Flow map matching. arXiv:2406.07507, 2024. [↩](#fnref:fmm "Jump back to footnote 18 in the text")
1. Z. Geng, M. Deng, X. Bai, J. Z. Kolter, and K. He. Mean flows for one-step generative modeling. arXiv:2505.13447, 2025. [↩](#fnref:meanflow "Jump back to footnote 19 in the text")
1. I. Gat, T. Remez, N. Shaul, F. Kreuk, R. T. Q. Chen, G. Synnaeve, Y. Adi, and Y. Lipman. Discrete flow matching. *NeurIPS*, 2024. [↩](#fnref:dfm "Jump back to footnote 20 in the text")
1. R. T. Q. Chen and Y. Lipman. Flow matching on general geometries. *ICLR*, 2024. [↩](#fnref:rfm "Jump back to footnote 21 in the text")
