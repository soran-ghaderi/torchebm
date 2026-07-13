"""Schedulers: time-varying hyperparameters that samplers advance automatically.

Any parameter registered as Schedulable (step_size, noise_scale, ...) accepts a
float or a BaseScheduler; the sampler calls step() once per iteration.
"""

import os

import torch

from torchebm.core import (
    CosineScheduler,
    ExponentialDecayScheduler,
    GaussianModel,
    LinearScheduler,
)
from torchebm.samplers import LangevinDynamics

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 500

# A scheduler is a stateful value: step() advances it, get_value() reads it.
schedules = {
    "linear 1.0 -> 0.1": LinearScheduler(start_value=1.0, end_value=0.1, n_steps=5),
    "cosine 1.0 -> 0.1": CosineScheduler(start_value=1.0, end_value=0.1, n_steps=5),
    "exp decay x0.5": ExponentialDecayScheduler(start_value=1.0, decay_rate=0.5),
}
for name, sched in schedules.items():
    values = [round(sched.step(), 3) for _ in range(5)]
    print(f"{name:20s} {values}")

# Annealed Langevin: a cosine-decaying step size, driven by the sampler itself.
model = GaussianModel(mean=torch.zeros(2), cov=torch.tensor([[1.0, 0.8], [0.8, 1.0]]))
sampler = LangevinDynamics(
    model=model,
    step_size=CosineScheduler(start_value=0.05, end_value=0.005, n_steps=N_STEPS),
    noise_scale=1.0,
)
samples = sampler.sample(dim=2, n_samples=2000, n_steps=N_STEPS)
print("annealed samples:", tuple(samples.shape))
print("recovered covariance:\n", torch.cov(samples.T).round(decimals=2))
