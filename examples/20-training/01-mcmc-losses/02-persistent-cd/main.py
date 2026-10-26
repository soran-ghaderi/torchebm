"""Persistent CD: negative chains resume from a replay buffer across updates.

Chains that persist across updates explore the model distribution far beyond
k Langevin steps, at identical per-step cost to CD-k. The buffer is the only
difference from cd-k: chains start from it and their end points go back in,
with 5% of each draw restarting from noise (Du and Mordatch, 2019) so the pool
keeps exploring. A healthy run drives E(neg) - E(data) toward zero: the chains
reach the model's own typical set, so the negative phase of the gradient is
less biased than with chains restarted at the data. BaseTrainer(model,
optimizer, loss_fn, sampler=sampler, buffer=buffer) runs the same loop.
"""

import os

import torch
from torch import nn

from torchebm.core import BaseModel
from torchebm.datasets import TwoMoonsDataset
from torchebm.losses import ContrastiveDivergenceLoss
from torchebm.samplers import LangevinDynamics
from torchebm.utils import ReplayBuffer

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
buffer = ReplayBuffer(capacity=8192, sample_shape=(2,))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(N_STEPS):
    batch = data[torch.randint(len(data), (256,))]
    negatives = sampler(model, buffer.sample(256), n_steps=10)
    buffer.add(negatives)
    loss = loss_fn(model(batch), model(negatives))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 200 == 0:
        gap = model(negatives).mean() - model(batch).mean()
        print(f"step {step:4d}  loss {loss.item():+.3f}  E(neg) - E(data) = {gap.item():+.3f}")
