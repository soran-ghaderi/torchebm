"""MeanFlow in 2D: one-step generation from the MeanFlow identity.

A MeanFlow model (Geng et al., 2025) outputs the average velocity
u(x_s, s, t) of the flow from time s to time t, so one call jumps from noise
to data. Training regresses the model on the MeanFlow identity

    u(x_s, s, t) = sg(v + (t - s) du/ds),

where du/ds is the JVP of the model along (v, 1) in (x_s, s), computed by
average_velocity_jvp on the pairs with s < t only. As in the reference
recipe, 25% of the time pairs have s < t, drawn from a logit-normal, the loss
uses the adaptive weight with p = 1, and Adam uses betas (0.9, 0.95).
FlowMapSampler then samples in 1, 2 or 4 jumps.
"""

import os

import torch

from torchebm.datasets import TwoMoonsDataset
from torchebm.interpolants import LinearInterpolant
from torchebm.losses import MeanFlowLoss
from torchebm.models import TimeConditionedMLP
from torchebm.samplers import FlowMapSampler
from torchebm.utils import average_velocity_jvp, sample_time_pairs

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 6000
N_GEN = 128 if SMOKE else 2000
BATCH_SIZE = 256


def median_dist(samples, data):
    """Median distance from each sample to the nearest data point."""
    return torch.cdist(samples, data).min(dim=1).values.median().item()


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
data = TwoMoonsDataset(n_samples=4000, noise=0.05, seed=0, device=device).get_data()
model = TimeConditionedMLP(in_dim=2, prediction_type="average_velocity").to(device)
path = LinearInterpolant()
loss_fn = MeanFlowLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.95))

for _ in range(N_STEPS):
    x1 = data[torch.randint(len(data), (BATCH_SIZE,), device=device)]
    s, t = sample_time_pairs(BATCH_SIZE, device=device)
    x_s, v = path(torch.randn_like(x1), x1, s)
    u, du_ds = average_velocity_jvp(model, x_s, s, t, v)
    loss = loss_fn(u, du_ds, v, s, t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.manual_seed(1)
x0 = torch.randn(N_GEN, 2, device=device)
sampler = FlowMapSampler(prediction_type="average_velocity")
for n_steps in (1, 2, 4):
    samples = sampler(model, x0, n_steps=n_steps)
    print(f"{n_steps:3d} steps: median dist to data {median_dist(samples, data):.3f}")
