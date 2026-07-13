# Foundations

The static objects of energy-based modeling, before anything moves. These
examples are fast, CPU-friendly, and involve no training: they build the
intuition the rest of the curriculum rests on.

- **Energies**: an energy is an unnormalised negative log-density; everything
  follows from its gradient \(-\nabla_x E\), the score of the model.
- **Datasets**: the synthetic 2D targets you sample and train against.
- **Schedulers**: the parameter schedules samplers and losses consume.
- **Interpolants**: the probability paths behind flow and diffusion sampling.

Theory: [The Energy-Based View](../../concepts/energy_view.md), and for the
paths, [Interpolants and Couplings](../../concepts/transport.md).

Next: [Sampling](../10-sampling/index.md), where these objects start to move.
