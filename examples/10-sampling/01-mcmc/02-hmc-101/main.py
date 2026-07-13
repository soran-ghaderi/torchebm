"""Hamiltonian Monte Carlo: long, informed proposals with Metropolis acceptance.

HMC simulates Hamiltonian dynamics with a leapfrog integrator for a few steps,
then accepts or rejects; longer trajectories decorrelate more per proposal but
cost more gradient evaluations.
"""

import os

import torch

from torchebm.core import GaussianModel
from torchebm.samplers import HamiltonianMonteCarlo

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 300

# A strongly correlated Gaussian: hard for plain Langevin, easy for HMC.
model = GaussianModel(mean=torch.zeros(2), cov=torch.tensor([[1.0, 0.95], [0.95, 1.0]]))

for n_leapfrog in (2, 10, 50):
    sampler = HamiltonianMonteCarlo(
        model=model, step_size=0.1, n_leapfrog_steps=n_leapfrog
    )
    samples, diag = sampler.sample(
        dim=2, n_samples=1000, n_steps=N_STEPS, return_diagnostics=True
    )
    print(f"n_leapfrog={n_leapfrog:3d}  acceptance {diag['acceptance_rate'].mean():.2f}"
          f"   recovered cov(0,1) = {torch.cov(samples.T)[0, 1]:.2f}  (target 0.95)")

# The trade: 2 leapfrog steps behaves like a timid random walk; 50 steps buys
# nearly independent proposals at 25x the gradient cost per MH step.
