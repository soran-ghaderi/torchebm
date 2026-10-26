"""Parallel chains: samplers are vectorized over the batch dimension.

The 10,000 Langevin chains below advance as one tensor program, and on a GPU
the same call runs millions of chains. Population statistics come from many
parallel chains at once; contrastive divergence draws its negatives the same
way. The target is a double well whose first coordinate has two symmetric
modes, at -b and +b.
"""

import os

import torch

from torchebm.core import DoubleWellModel
from torchebm.samplers import LangevinDynamics

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 500
N_CHAINS = 10_000

model = DoubleWellModel(barrier_height=2.0)
sampler = LangevinDynamics(step_size=0.01, noise_scale=1.0)
samples = sampler(model, torch.randn(N_CHAINS, 2), n_steps=N_STEPS)

left = (samples[:, 0] < 0).float().mean()
print("samples:", tuple(samples.shape))
print(f"basin occupancy: left {left:.1%} / right {1 - left:.1%}  (symmetric target: 50/50)")
print(f"mean energy: {model(samples).mean():.2f}")
