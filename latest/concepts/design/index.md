# Design and Scope

TorchEBM is a library for energy-based generative modeling in the broad sense: the study of models that represent a distribution through a scalar potential, its score, or a velocity field transporting a reference distribution onto it. Its design premise is that the methods usually treated as separate families, EBMs, denoising diffusion, flow matching, stochastic interpolants, Schrödinger bridges, and their hybrids, are compositions of the same small set of components. The library implements the components once and lets the compositions be configuration rather than code.

The consequential split among those compositions is the training regime. Simulation-free objectives (flow matching, equilibrium matching, denoising score matching) fit the model by regression along a probability path, with no sampler inside the loss; simulation-based objectives (contrastive divergence) run MCMC in the loop. TorchEBM implements both and leans simulation-free: that is what lets modern energy-parameterized models train stably at scales where classical EBM training could not.

## The unifying object

An energy-based model defines a density

[ p\_\\theta(x) = \\frac{e^{-E\_\\theta(x)}}{Z\_\\theta}, \\qquad Z\_\\theta = \\int e^{-E\_\\theta(x)},dx , ]

where the normalizer (Z\_\\theta) is intractable but unnecessary for both sampling and most training objectives: Langevin and Hamiltonian dynamics need only (\\nabla_x E\_\\theta), and score-based objectives fit (s\_\\theta(x) = -\\nabla_x E\_\\theta(x)) directly[2](#fn:hyvarinen)[3](#fn:vincent).

Continuous-time generative models extend this static picture with a time axis. An interpolant

[ x_t = \\alpha(t),x_1 + \\sigma(t),x_0, \\qquad x_0 \\sim \\mathcal{N}(0, I),; x_1 \\sim p\_{\\text{data}}, ]

defines a probability path whose marginal velocity, score, and noise prediction are algebraically interchangeable[10](#fn:albergo)[7](#fn:song-sde). A model may therefore be parameterized as an energy, a score, or a velocity; TorchEBM converts between them (`velocity_to_score`, `score_to_velocity`, and relatives) and treats the choice as a modeling decision, not an architectural one.

## The component algebra

Methods factor into orthogonal components, each owned by one subpackage:

| Component      | Role                                                     | Package          |
| -------------- | -------------------------------------------------------- | ---------------- |
| Energy / field | the learned or analytic potential, score, or velocity    | `core`, `models` |
| Interpolant    | the path (\\alpha(t), \\sigma(t)) between noise and data | `interpolants`   |
| Coupling       | which (x_0) is paired with which (x_1)                   | `couplings`      |
| Objective      | the loss fitting the model to data                       | `losses`         |
| Sampler        | the dynamics generating samples                          | `samplers`       |
| Integrator     | the numerical scheme inside the dynamics                 | `integrators`    |

The list is open: new axes join as the field produces them, under the same one-contract-per-axis rule. The current composition map, generated from the installed package at build time:

```
graph LR
    field["energy / field<br/>core: 6 analytic energies · models"]
    interp["interpolants (3)"]
    coup["couplings (7)"]
    integ["integrators (12)"]
    samp["samplers (6)"]
    loss["objectives (8)"]
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

| Family                                                                                               | Field                            | Path                          | Coupling                                            | Objective                                               | Training                                    | Sampling                                         |
| ---------------------------------------------------------------------------------------------------- | -------------------------------- | ----------------------------- | --------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------ |
| MCMC-trained EBMs[1](#fn:hinton)[5](#fn:du)                                                          | energy                           | none                          | none                                                | contrastive family (e.g. CD, persistent CD)             | simulation-based (MCMC in the loss)         | MCMC                                             |
| Score-based EBMs[2](#fn:hyvarinen)[3](#fn:vincent)[4](#fn:song-ermon)                                | energy                           | fixed or annealed noise       | none                                                | score matching family                                   | simulation-free                             | annealed Langevin                                |
| Denoising diffusion[6](#fn:ho)[7](#fn:song-sde)                                                      | score or noise                   | variance-preserving schedules | independent                                         | denoising regression over (t)                           | simulation-free                             | reverse SDE or probability-flow ODE              |
| Flow matching[8](#fn:lipman)[11](#fn:tong)[9](#fn:liu)                                               | velocity                         | simple paths (e.g. linear)    | independent or OT-based (e.g. minibatch OT, reflow) | conditional path regression                             | simulation-free                             | ODE                                              |
| Stochastic interpolants[10](#fn:albergo)                                                             | velocity (+ score)               | any (\\alpha, \\sigma)        | any                                                 | interpolant regression                                  | simulation-free                             | ODE or SDE                                       |
| Energy-parameterized transport (e.g. equilibrium matching[12](#fn:eqm), energy matching[13](#fn:em)) | time-independent energy or field | any                           | any, including weighted OT                          | path regression, optionally with contrastive refinement | simulation-free, optional contrastive phase | few-step ODE or a single annealed Langevin sweep |
| Schrödinger bridges[14](#fn:debortoli)                                                               | forward and backward drifts      | diffusion bridge              | iterative proportional fitting                      | bridge matching                                         | simulation-free (bridge matching)           | SDE (roadmap)                                    |

The last row is deliberate: bridges are on the roadmap precisely because the components they need (couplings, interpolants, SDE integrators, dual parameterizations) are the ones the library already maintains.

The bottleneck of MCMC-trained EBMs is long-run sampling inside the loss; the simulation-free families replace it with regression along a path, and the hybrid families keep the energy parameterization while inheriting the simulation-free training cost. Moving between rows is a configuration change, not a rewrite, and the same holds for families the field has not named yet: anything expressible as a field, a path, a pairing, an objective, and a numerical scheme composes here without new abstractions.

## Design principles

**Composition over frameworks.** There is no `Trainer` orchestrating hidden state. A training loop is user code over plain objects (a model, a loss, a sampler, an optimizer). Components accept each other as constructor arguments (`ContrastiveDivergence(model=..., sampler=...)`, `EnergyMatchingLoss(coupling=...)`, `FlowSampler(integrator=...)`).

**One contract per axis.** `BaseModel` is any differentiable map `(N, d) -> (N,)`; `BaseInterpolant` is the pair (\\alpha(t), \\sigma(t)) and their derivatives; `BaseCoupling` returns a `CouplingResult` that unpacks as the pair `(x0, x1)` and carries optional per-pair weights; integrators expose one `integrate` contract shared by MCMC and flow sampling. Registries (`get_integrator`, `get_coupling`, `get_interpolant`) make every choice addressable by string without hiding the class-based path.

**Vectorization first.** Chains, negatives, and path samples are batch dimensions, never Python loops. The sampler that draws 100 chains draws 100,000 by changing one integer.

**Stateless mathematics, schedulable hyperparameters.** Interpolants and couplings are pure functions of tensors. Anything that legitimately varies during a run (step sizes, noise scales, temperatures) is a `Schedulable` parameter accepting a float or a scheduler.

## Scope boundaries

TorchEBM implements primitives and reference recipes, not model zoos: no pretrained checkpoints, no dataset loaders beyond 2D synthetic benchmarks, no training orchestration.

Where a method is not yet expressible (Schrödinger bridges, discrete-state models), the gap is stated in the roadmap.

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
1. R. Wang and Y. Du. Equilibrium matching: generative modeling with implicit energy-based models. arXiv:2510.02300, 2025. [↩](#fnref:eqm "Jump back to footnote 12 in the text")
1. M. Balcerak, T. Amiranashvili, A. Terpin, S. Shit, L. Bogensperger, S. Kaltenbach, P. Koumoutsakos, and B. Menze. Energy matching: unifying flow matching and energy-based models for generative modeling. arXiv:2504.10612, 2025. [↩](#fnref:em "Jump back to footnote 13 in the text")
1. V. De Bortoli, J. Thornton, J. Heng, and A. Doucet. Diffusion Schrödinger bridge with applications to score-based generative modeling. *NeurIPS*, 2021. [↩](#fnref:debortoli "Jump back to footnote 14 in the text")
