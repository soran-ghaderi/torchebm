"""Denoising Score Matching: train an energy without any MCMC sampling.

DSM (Vincent, 2011) perturbs data with Gaussian noise and matches the model
score to the denoising direction; no negatives, no Hessian trace.
"""

import os

import torch
from torch import nn

from torchebm.core import BaseModel
from torchebm.datasets import TwoMoonsDataset
from torchebm.losses import DenoisingScoreMatching
from torchebm.samplers import LangevinDynamics

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 2000
SAMPLE_STEPS = 20 if SMOKE else 500


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
# noise_scale sets the smoothing bandwidth: the model learns the score of the
# noise-convolved data density, so too large blurs the moons, too small is noisy.
dsm = DenoisingScoreMatching(model=energy, noise_scale=0.1)
opt = torch.optim.Adam(energy.parameters(), lr=1e-3)

for step in range(N_STEPS):
    batch = data[torch.randint(len(data), (256,))]
    loss = dsm(batch)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 400 == 0:
        print(f"step {step:4d}  loss {loss.item():.3f}")

# The trained energy separates data from off-manifold points...
box = torch.rand(2000, 2) * 4 - torch.tensor([1.5, 1.0])
print(f"E(data) = {energy(data).mean():.2f}   E(random box) = {energy(box).mean():.2f}")

# ...and, unlike the pure regression it was trained by, it is still an EBM:
# the same Langevin sampler used for CD models draws samples from it.
sampler = LangevinDynamics(model=energy, step_size=0.01, noise_scale=0.3)
samples = sampler.sample(x=torch.randn(2000, 2), n_steps=SAMPLE_STEPS)
print("sampled from the DSM energy:", tuple(samples.shape))
