# Sampling

Given a *fixed* target, how do you draw samples from it? Nothing is learned here:
the model is known, and the question is one of dynamics and numerics.

- **MCMC**: Langevin dynamics and Hamiltonian Monte Carlo, plus the fact that
  chains are a batch dimension, so thousands of them cost one integer.
- **Integrators**: the numerical engines *inside* the samplers. Order of accuracy
  is measurable, so we measure it against an exact solution.
- **Flow**: continuous-time generation, where `FlowSampler` integrates a velocity
  field as an ODE or an SDE using those same integrators.

Start with **Langevin Dynamics 101**; the integrator and flow examples show the
numerics that both MCMC and generative sampling stand on.

Theory: [Sampling and Integration](../../concepts/sampling.md).

Next: [Training](../20-training/index.md), where the target is learned rather
than given.
