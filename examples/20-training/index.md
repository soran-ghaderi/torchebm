# Training

The payoff: *learn* a target from data instead of sampling one you already have.
Ordered from the most classical objective to the most recent.

- **MCMC-based losses**: contrastive divergence and its persistent variant, which
  buy a calibrated energy at the cost of sampling inside the training loop.
- **Score matching**: fit the score directly and drop the inner sampler entirely.
- **Equilibrium matching**: learn a time-invariant field, then generate by
  integrating it *or* by descending it as an energy.
- **Energy matching**: a single time-independent potential serving both transport
  and Boltzmann sampling.
- **Couplings**: which noise sample is paired with which datum, which decides how
  straight the transport paths are, and so how few steps generation needs.

Start with **CD-k on Two Moons**, then **Equilibrium Matching in 2D** to watch one
model act as both a flow and an energy.

Theory: [Learning Objectives](../../concepts/objectives.md) for the losses, and
[Interpolants and Couplings](../../concepts/transport.md) for the paths and
pairings they consume.

Next: [Showcase](../90-showcase/index.md).
