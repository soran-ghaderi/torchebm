# Sampling

These examples draw samples from a *fixed*, known target. Nothing is learned;
the subject is the dynamics and the numerics.

- **MCMC**: Langevin dynamics and Hamiltonian Monte Carlo, with chains as a batch
  dimension, so one call runs thousands of them.
- **Integrators**: the numerical engines *inside* the samplers, with their order
  of accuracy measured against an exact solution.
- **Flow**: continuous-time generation, where `FlowSampler` integrates a velocity
  field as an ODE or an SDE using those same integrators.

Start with **Langevin Dynamics 101**; the integrator and flow examples show the
numerics that both MCMC and generative sampling stand on.

Theory: [Sampling and Integration](../../concepts/sampling.md).

Next: [Training](../20-training/index.md), where the target is learned from data.
