"""Equilibrium Matching: learn a vector field that generates data, sample via an ODE.

EqM is TorchEBM's generative path. A time-invariant field is trained toward data
with EquilibriumMatchingLoss, then FlowSampler integrates it from noise. EqM models
sample with negate_velocity=True (the learned field points data -> noise).
"""

import os

import torch
from torch import nn

from torchebm.datasets import TwoMoonsDataset
from torchebm.losses import EquilibriumMatchingLoss
from torchebm.samplers import FlowSampler

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 3000


class Field(nn.Module):               # f(x, t) -> R^2; EqM is time-invariant, so t is unused
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x, t, **kwargs):
        return self.net(x)


data = TwoMoonsDataset(n_samples=4000, noise=0.05, seed=0).get_data()

model = Field()
loss_fn = EquilibriumMatchingLoss(model=model, interpolant="linear", energy_type="dot")
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

for step in range(N_STEPS):
    batch = data[torch.randint(len(data), (256,))]
    loss = loss_fn(batch)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")

# Generate: integrate the learned field from Gaussian noise to data.
flow = FlowSampler(
    model=model, interpolant="linear", negate_velocity=True, integrator="euler"
)
samples = flow.sample(x=torch.randn(4000, 2), n_steps=100)
print("generated:", tuple(samples.shape))
