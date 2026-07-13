"""Parallel chains: samplers are vectorized over the batch dimension, chains are free.

Running 10,000 Langevin chains is one tensor program, not 10,000 loops; on a GPU
the same call scales to millions of chains.
"""

import os

import torch

from torchebm.core import DoubleWellModel
from torchebm.samplers import LangevinDynamics

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 500
N_CHAINS = 10_000

# A symmetric double well: two basins around x = -a and x = +a in the first coord.
model = DoubleWellModel(barrier_height=2.0)
sampler = LangevinDynamics(model=model, step_size=0.01, noise_scale=1.0)

# One call = N_CHAINS independent chains advanced N_STEPS times, fully batched.
samples = sampler.sample(dim=2, n_samples=N_CHAINS, n_steps=N_STEPS)

left = (samples[:, 0] < 0).float().mean()
print("samples:", tuple(samples.shape))
print(f"basin occupancy: left {left:.1%} / right {1 - left:.1%}  (symmetric target: 50/50)")
print(f"mean energy: {model(samples).mean():.2f}")

# Population statistics from one shot of parallel chains replace long single-chain
# runs; this is the pattern every training loss in TorchEBM builds on.
