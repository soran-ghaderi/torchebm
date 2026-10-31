"""Hamiltonian Monte Carlo: long, informed proposals with Metropolis acceptance.

HMC simulates Hamiltonian dynamics with a leapfrog integrator for a few steps,
then accepts or rejects; longer trajectories decorrelate more per proposal but
cost more gradient evaluations. On a strongly correlated Gaussian (hard for
plain Langevin), 2 leapfrog steps behave like a timid random walk while 50 buy
nearly independent proposals at 25x the gradient cost per transition; the
lag-1 autocorrelation of the chains measures the trade.
"""

import os

import torch

from torchebm.core import GaussianModel
from torchebm.samplers import HamiltonianMonteCarlo

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 300

model = GaussianModel(mean=torch.zeros(2), cov=torch.tensor([[1.0, 0.95], [0.95, 1.0]]))
x0 = torch.randn(1000, 2)

for n_leapfrog in (2, 10, 50):
    sampler = HamiltonianMonteCarlo(step_size=0.1, n_leapfrog_steps=n_leapfrog)
    trajectory = sampler(model, x0, n_steps=N_STEPS, return_trajectory=True)
    samples = trajectory[:, -1]
    x = trajectory[:, N_STEPS // 2 :, 0]
    x = x - x.mean()
    autocorrelation = (x[:, 1:] * x[:, :-1]).mean() / x.pow(2).mean()
    print(f"n_leapfrog={n_leapfrog:3d}  lag-1 autocorrelation {autocorrelation:.2f}"
          f"   recovered cov(0,1) = {torch.cov(samples.T)[0, 1]:.2f}  (target 0.95)")
