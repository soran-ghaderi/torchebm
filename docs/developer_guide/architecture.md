---
title: Architecture
description: Package layout and how the core abstractions compose
icon: material/folder-outline
---

# Architecture

TorchEBM is organised around a small set of base classes (`BaseModel`, `BaseSampler`, `BaseLoss`, `BaseIntegrator`, `BaseInterpolant`). Everything else is composition: a loss uses a sampler, a sampler uses an integrator, an integrator steps a field derived from a model.

---

## Package layout

```
torchebm/
├── core/          # Base classes + DeviceMixin
├── losses/        # ContrastiveDivergence, ScoreMatching, EquilibriumMatching, …
├── samplers/      # LangevinDynamics, HamiltonianMonteCarlo, FlowSampler, GradientDescent
├── integrators/   # EulerMaruyama, Heun, Leapfrog, RK4, DOPRI5, …
├── interpolants/  # Linear, Cosine, VariancePreserving (flow / diffusion paths)
├── datasets/      # Synthetic data generators (gaussian mixture, 2D shapes, etc.)
├── models/        # Neural architectures (MLPs, transformers) used as energies or fields
├── cuda/          # Custom CUDA kernels (placeholder, see cuRBLAS)
└── utils/         # Shared helpers
```

Mirror this layout under `tests/` when adding tests.

---

## Core abstractions

| Base class         | Contract                                          | Notable subclasses                    |
|--------------------|---------------------------------------------------|---------------------------------------|
| `BaseModel`        | `forward(x) -> energy` or score/velocity          | `GaussianEnergy`, `MLP2D`, transformers |
| `BaseSampler`      | `sample(x=None, dim, n_steps, n_samples, …)`      | `LangevinDynamics`, `HMC`, `FlowSampler` |
| `BaseIntegrator`   | One numerical step of an ODE/SDE                  | `Leapfrog`, `EulerMaruyama`, `DOPRI5` |
| `BaseLoss`         | `forward(x, *args, **kw) -> scalar`               | `ScoreMatching`, `ContrastiveDivergence`, `EquilibriumMatching` |
| `BaseInterpolant`  | `interpolate(x0, x1, t) -> (xt, ut)`              | `Linear`, `Cosine`, `VariancePreserving` |
| `DeviceMixin`      | `self.device`, `self.dtype`, `autocast_context()` | used by everything above              |

Every component exposed through `torchebm.*.__init__` is auto-discovered by the benchmark suite (see [Benchmarking](benchmarking.md)).

---

## How the pieces compose

Training wiring depends on the loss family. Two patterns cover everything in the library:

=== "Sampler-free (score / flow / EqM)"
    ```mermaid
    graph LR
        data[x1] --> loss
        noise["x0 ~ N(0,I)"] --> interp[interpolant]
        data --> interp
        interp -- xt, target --> loss
        model --> loss
        loss -- grad --> opt[optimizer]
        opt --> model
    ```
    Score matching, equilibrium matching, and flow matching compute their target from data plus a noise / interpolation step. **No sampler runs during training.** Samplers are only used at generation time.

=== "Sampler-based (CD family)"
    ```mermaid
    graph LR
        data[data x] --> loss
        model --> sampler
        sampler -- negatives --> loss
        loss -- grad --> opt[optimizer]
        opt --> model
    ```
    Contrastive divergence and its variants draw negatives from the current model via a sampler (e.g. `LangevinDynamics`, `HMC`) every step.

=== "Generation (all objectives)"
    ```mermaid
    graph LR
        model2[trained model] --> sampler2[sampler]
        sampler2 --> samples[x ~ p]
    ```
    A sampler drives a **field** derived from the trained model through an **integrator** to produce samples.

Swapping any one piece (e.g. replacing `EulerMaruyama` with `Heun` inside `LangevinDynamics`) does not require touching the others.

---

## Time conditioning

Not all objectives condition the model on \( t \). The distinction matters when wiring components:

- **EquilibriumMatching**: time-invariant. The loss passes \( x_t \) only; the model receives **no** time input.
- **FlowSampler / score-matching with diffusion**: time-conditional. The field is \( v(x, t) \); the sampler feeds \( t \) every step.

See `torchebm/losses/equilibrium_matching.py` and `torchebm/samplers/flow.py` for the reference patterns.

