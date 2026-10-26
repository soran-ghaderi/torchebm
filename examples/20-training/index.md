# Training

These examples *learn* a target from data, ordered from the most classical
objective to the most recent.

- **MCMC-based losses**: contrastive divergence and its persistent variant, which
  fit a calibrated energy at the cost of sampling inside the training loop.
- **Score matching**: fit the score directly, with no sampler in the training loop.
- **Equilibrium matching**: learn a time-invariant field, then generate by
  integrating it as an ODE or by iterating it toward its attractors; the
  explicit variant learns a scalar energy.
- **Energy matching**: a single time-independent potential serving both transport
  and Boltzmann sampling.
- **Couplings**: which noise sample is paired with which datum, which decides how
  straight the transport paths are, and so how few steps generation needs.

Start with **CD-k on Two Moons**, then **Equilibrium Matching in 2D** to sample
one time-invariant field two ways and an explicit energy with repulsion.

Theory: [Learning Objectives](../../concepts/objectives.md) for the losses, and
[Interpolants and Couplings](../../concepts/transport.md) for the paths and
pairings they consume.

Next: [Showcase](../90-showcase/index.md).
