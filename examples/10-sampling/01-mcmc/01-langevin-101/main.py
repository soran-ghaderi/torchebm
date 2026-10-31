"""Langevin Dynamics: sample a 2D energy by descending its gradient with noise.

    x <- x - step_size * grad E(x) + sqrt(2 * step_size) * noise_scale * eps

The target is a correlated Gaussian, E(x) = 1/2 x^T Sigma^-1 x. The call runs
2000 independent chains as one batch and returns their final points.
"""

import os

import torch

from torchebm.core import GaussianModel
from torchebm.samplers import LangevinDynamics

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 500

model = GaussianModel(mean=torch.zeros(2), cov=torch.tensor([[1.0, 0.8], [0.8, 1.0]]))
sampler = LangevinDynamics(step_size=0.02, noise_scale=1.0)
samples = sampler(model, torch.randn(2000, 2), n_steps=N_STEPS)

print("samples:", tuple(samples.shape))
print("mean:", samples.mean(0).round(decimals=2).tolist())
print("recovered covariance:\n", torch.cov(samples.T).round(decimals=2))
