# Sampling and Integration

Sampling answers one question: given a model, how are samples produced? The library covers the spectrum from Markov chain methods on a fixed energy to deterministic integration of a learned velocity field, and factors the numerics into a shared integrator layer. The current family, generated from the installed package at build time:

```
graph TD
    BaseSampler(["BaseSampler"])
    BaseSampler --> FlowSampler
    BaseSampler --> GradientDescentSampler
    BaseSampler --> HamiltonianMonteCarlo
    BaseSampler --> LangevinDynamics
    BaseSampler --> NesterovSampler
    BaseSampler --> RiemannianManifoldHMC
```

## MCMC on an energy

**Langevin dynamics** descends the energy gradient (equivalently, ascends the model score (-\\nabla_x E)) with injected noise,

[ x\_{k+1} = x_k - \\eta, \\nabla_x E(x_k) + \\sqrt{2\\eta}; \\varepsilon_k, \\qquad \\varepsilon_k \\sim \\mathcal{N}(0, I), ]

and is the workhorse behind contrastive-divergence negatives and annealed generation:

```python
from torchebm.samplers import LangevinDynamics
sampler = LangevinDynamics(model=model, step_size=0.02, noise_scale=1.0)
samples = sampler.sample(dim=2, n_samples=2000, n_steps=500)
```

**Hamiltonian Monte Carlo** augments the state with momentum and integrates Hamiltonian dynamics with a symplectic leapfrog before a Metropolis accept/reject, trading gradient evaluations for long, decorrelated proposals. **Riemannian manifold HMC** replaces the flat metric with a position-dependent one and integrates the resulting non-separable Hamiltonian with a generalised (implicit) leapfrog; it is the geometry-aware option for ill-conditioned targets.

```python
from torchebm.samplers import HamiltonianMonteCarlo
hmc = HamiltonianMonteCarlo(model=model, step_size=0.1, n_leapfrog_steps=10)
samples, diag = hmc.sample(dim=2, n_samples=1000, n_steps=300,
                           return_diagnostics=True)
diag["acceptance_rate"].mean()
```

Optimization-based samplers (plain gradient descent and momentum variants such as `NesterovSampler`) are the noise-free members of the family: mode seekers rather than samplers, useful for finding minima of a trained energy.

Three conventions hold across all of them:

- **Chains are a batch dimension.** `n_samples` chains advance as one tensor program; scaling chains costs one integer, not a loop.
- **Diagnostics are dictionaries.** `return_diagnostics=True` returns per-step tensors (means, variances, energies, acceptance rates) rather than opaque state.
- **Hyperparameters are schedulable.** Step sizes and noise scales accept schedulers, which is how annealed Langevin and temperature sweeps are expressed.

## Continuous-time generation

For trained flow, diffusion, and equilibrium models, `FlowSampler` integrates the learned field from noise to data:

```python
from torchebm.samplers import FlowSampler
flow = FlowSampler(model=field, interpolant="linear",
                   prediction="velocity", integrator="heun")
samples = flow.sample(x=torch.randn(4096, 2), n_steps=50)
```

`prediction` declares what the network outputs (`"velocity"`, `"score"`, or `"noise"`); the interpolant converts between them internally, so one sampler serves flow matching, score-based diffusion, and noise-prediction models. The process is chosen at construction: `mode="ode"` (default) integrates the probability-flow ODE, `mode="sde"` adds a diffusion term with selectable form (`diffusion_form=`, `last_step=`). `sample()` follows the same contract as every other sampler; with a fixed-step integrator it supports `thin`, `return_trajectory`, and `return_diagnostics`, while adaptive integrators (`dopri5`, ...) return the final state only. Equilibrium-matching models set `negate_velocity=True`, since they learn the data-to-noise direction.

## The integrator layer

Both worlds sit on `torchebm.integrators`, which covers the standard solver families: explicit Runge-Kutta schemes of increasing order (e.g. `heun`, `rk4`), adaptive step-size solvers (e.g. `dopri5`), implicit schemes for stiff dynamics, and symplectic integrators for Hamiltonian ones (e.g. `leapfrog`). The shipped set, generated from the installed package:

```
graph TD
    BaseIntegrator(["BaseIntegrator"])
    BaseRungeKuttaIntegrator(["BaseRungeKuttaIntegrator"])
    BaseSDERungeKuttaIntegrator(["BaseSDERungeKuttaIntegrator"])
    BaseSymplecticIntegrator(["BaseSymplecticIntegrator"])
    BaseRungeKuttaIntegrator --> AdaptiveHeunIntegrator
    BaseIntegrator --> BaseRungeKuttaIntegrator
    BaseSDERungeKuttaIntegrator --> BackwardEulerMaruyamaIntegrator
    BaseRungeKuttaIntegrator --> BaseSDERungeKuttaIntegrator
    BaseRungeKuttaIntegrator --> Bosh3Integrator
    BaseRungeKuttaIntegrator --> Dopri5Integrator
    BaseRungeKuttaIntegrator --> Dopri8Integrator
    BaseSDERungeKuttaIntegrator --> EulerMaruyamaIntegrator
    BaseSymplecticIntegrator --> GeneralisedLeapfrogIntegrator
    BaseIntegrator --> BaseSymplecticIntegrator
    BaseSDERungeKuttaIntegrator --> HeunIntegrator
    BaseSymplecticIntegrator --> LeapfrogIntegrator
    BaseRungeKuttaIntegrator --> RK438Integrator
    BaseRungeKuttaIntegrator --> RK4Integrator
```

Every sampler accepts `integrator=` as a registry name or an instance, and `get_integrator` constructs one directly:

```python
from torchebm.integrators import get_integrator
integ = get_integrator("rk4")
```

Higher-order solvers buy accuracy per step; adaptive solvers buy robustness on stiff fields; symplectic solvers preserve the phase-space volume HMC's correctness depends on. The [integrator comparison example](https://soran-ghaderi.github.io/torchebm/latest/examples/10-sampling/02-integrators/01-integrator-comparison/index.md) measures the orders directly against an exact solution.

## Runnable counterparts

- [Langevin Dynamics 101](https://soran-ghaderi.github.io/torchebm/latest/examples/10-sampling/01-mcmc/01-langevin-101/index.md)
- [HMC 101](https://soran-ghaderi.github.io/torchebm/latest/examples/10-sampling/01-mcmc/02-hmc-101/index.md)
- [Parallel Chains](https://soran-ghaderi.github.io/torchebm/latest/examples/10-sampling/01-mcmc/03-parallel-chains/index.md)
- [FlowSampler ODE 101](https://soran-ghaderi.github.io/torchebm/latest/examples/10-sampling/03-flow/01-flow-sampler-ode/index.md)
