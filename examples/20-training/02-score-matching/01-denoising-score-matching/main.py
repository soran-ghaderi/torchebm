"""Denoising Score Matching: train an energy without any MCMC sampling.

DSM (Vincent, 2011) perturbs data with Gaussian noise and matches the model
score to the denoising direction, the score of the perturbation kernel
-noise / sigma; no negatives, no Hessian trace. sigma sets the smoothing
bandwidth: the model learns the score of the noise-convolved data density, so
too large blurs the moons and too small is noisy.

The model returns its energy and its score from one call: the loss consumes
the score, the diagnostics read the energy, and the Langevin sampler picks the
score again. The trained energy separates data from off-manifold points and,
unlike the pure regression it was trained by, is still an EBM the same sampler
used for CD models draws from.
"""

import os

import torch
from torch import nn

from torchebm.core import BaseModel
from torchebm.datasets import TwoMoonsDataset
from torchebm.losses import DenoisingScoreMatchingLoss
from torchebm.samplers import LangevinDynamics

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 2000
SAMPLE_STEPS = 20 if SMOKE else 500


class MLP(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


data = TwoMoonsDataset(n_samples=3000, noise=0.05, seed=0).get_data()

model = MLP(prediction_type=("energy", "score"))
loss_fn = DenoisingScoreMatchingLoss()
sigma = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(N_STEPS):
    batch = data[torch.randint(len(data), (256,))]
    noise = torch.randn_like(batch)
    _, score = model(batch + sigma * noise)
    loss = loss_fn(score, -noise / sigma)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 400 == 0:
        print(f"step {step:4d}  loss {loss.item():.3f}")

x_random = torch.rand(2000, 2) * 4 - torch.tensor([1.5, 1.0])
with torch.no_grad():
    energy_data, _ = model(data)
    energy_random, _ = model(x_random)
print(f"E(data) = {energy_data.mean():.2f}   E(random box) = {energy_random.mean():.2f}")

sampler = LangevinDynamics(step_size=0.01, noise_scale=0.3)
samples = sampler(model, torch.randn(2000, 2), n_steps=SAMPLE_STEPS)
print("sampled from the DSM energy:", tuple(samples.shape))
