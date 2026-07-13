"""Contrastive Divergence: train an energy net so data sits in low-energy regions.

CD pushes E down on real data and up on short-run MCMC "negatives" from the model.
"""

import os

import torch
from torch import nn

from torchebm.core import BaseModel
from torchebm.datasets import TwoMoonsDataset
from torchebm.losses import ContrastiveDivergence
from torchebm.samplers import LangevinDynamics

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 1000


class MLPEnergy(BaseModel):           # E(x): R^2 -> R, one scalar energy per point
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


data = TwoMoonsDataset(n_samples=3000, noise=0.05, seed=0).get_data()

energy = MLPEnergy()
sampler = LangevinDynamics(model=energy, step_size=0.1, noise_scale=1.0)  # draws negatives
cd = ContrastiveDivergence(model=energy, sampler=sampler, k_steps=10)     # CD-10
opt = torch.optim.Adam(energy.parameters(), lr=1e-3)

for step in range(N_STEPS):
    batch = data[torch.randint(len(data), (256,))]
    loss, _negatives = cd(batch)      # gradient of loss shapes the energy landscape
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 200 == 0:
        print(f"step {step:4d}  loss {loss.item():+.3f}")
