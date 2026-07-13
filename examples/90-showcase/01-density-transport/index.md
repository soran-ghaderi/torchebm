# Density Transport

Treat an ordinary image as an unnormalised probability density and watch a cloud
of noise particles assemble into it under **annealed Langevin dynamics**, sampled
with TorchEBM's `LangevinDynamics` through a custom `BaseModel` energy.

## What it teaches

- An image's pixel intensity defines a Boltzmann density \(p(x) \propto \rho(x)\),
  so the energy is \(E(x) = -\log \rho(x)\). A differentiable bilinear lookup
  (`grid_sample`) lets autograd supply the score \(\nabla_x \log \rho(x)\) the
  sampler rides.
- Coarse-to-fine **annealing**, blur the density then sharpen it in stages, gives
  broad basins of attraction first and crisp detail last. This is the same
  intuition as annealed Langevin / score-based generative sampling, with a real
  image as the target instead of a toy mixture.

## Requires

In addition to TorchEBM, this showcase needs **Pillow**: `pip install pillow`.

## Expected output

One GIF per `--target`, written to the `assets/` folder next to the script
(`density_transport_<target>.gif`), showing a cloud of particles transported
from noise into the target image (`logo`, `galaxy`, `neuron`, `starry_night`).
