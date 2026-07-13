"""Persistent CD: negatives resume from a replay buffer instead of the data batch.

Chains that persist across updates explore the model distribution far beyond
k Langevin steps, at identical per-step cost to CD-k.
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


class MLPEnergy(BaseModel):
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
sampler = LangevinDynamics(model=energy, step_size=0.1, noise_scale=1.0)
# persistent=True switches on the replay buffer; negatives continue where the
# previous update's chains stopped instead of restarting from data.
pcd = ContrastiveDivergence(
    model=energy, sampler=sampler, k_steps=10,
    persistent=True, buffer_size=8192,
)
opt = torch.optim.Adam(energy.parameters(), lr=1e-3)

for step in range(N_STEPS):
    batch = data[torch.randint(len(data), (256,))]
    loss, negatives = pcd(batch)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 200 == 0:
        gap = energy(negatives).mean() - energy(batch).mean()
        print(f"step {step:4d}  loss {loss.item():+.3f}  E(neg) - E(data) = {gap.item():+.3f}")

# A healthy run drives the gap toward zero: buffer chains reach the model's
# own typical set, giving lower-variance gradients than restarting from data.
