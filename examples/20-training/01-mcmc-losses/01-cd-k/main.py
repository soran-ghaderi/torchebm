"""Contrastive Divergence: train an energy net so data sits in low-energy regions.

CD pushes E down on real data and up on short-run MCMC "negatives" from the model:
the sampler draws them from the current model with k Langevin steps started at
the data batch (CD-10 here), and the loss is mean E(data) - mean E(negatives).
"""

import os

import torch
from torch import nn

from torchebm.core import BaseModel
from torchebm.datasets import TwoMoonsDataset
from torchebm.losses import ContrastiveDivergenceLoss
from torchebm.samplers import LangevinDynamics

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 1000


class MLP(BaseModel):
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

model = MLP()
sampler = LangevinDynamics(step_size=0.1, noise_scale=1.0)
loss_fn = ContrastiveDivergenceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(N_STEPS):
    batch = data[torch.randint(len(data), (256,))]
    negatives = sampler(model, batch, n_steps=10)
    loss = loss_fn(model(batch), model(negatives))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 200 == 0:
        print(f"step {step:4d}  loss {loss.item():+.3f}")
