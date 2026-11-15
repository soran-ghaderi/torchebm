"""Shortcut Models in 2D: one-step generation by self-consistency.

A shortcut model (Frans et al., 2024) outputs the average velocity
u(x_t, t, t + d) of a jump of size d, so one call jumps from noise to data.
Training mixes two terms, as in the reference code:

    flow matching:     || u(x_t, t, t + 1/M) - (x1 - x0) ||^2
    self-consistency:  || u(x_t, t, t + 2d) - sg((u1 + u2) / 2) ||^2

where u1 and u2 are two jumps of size d from the EMA model. One eighth of each
batch goes to self-consistency, with big jumps 2d in {1, 1/2, ..., 1/64} and t
on the grid of 2d. FlowMapSampler then samples in 1, 2, 4 or 128 jumps.
"""

import os

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from torchebm.datasets import TwoMoonsDataset
from torchebm.interpolants import LinearInterpolant
from torchebm.losses import FlowMatchingLoss, ShortcutLoss
from torchebm.models import TimeConditionedMLP
from torchebm.samplers import FlowMapSampler

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 6000
N_GEN = 128 if SMOKE else 2000
BATCH_SIZE = 256
M = 128
LOG2_M = 7


def median_dist(samples, data):
    """Median distance from each sample to the nearest data point."""
    return torch.cdist(samples, data).min(dim=1).values.median().item()


torch.manual_seed(0)
data = TwoMoonsDataset(n_samples=4000, noise=0.05, seed=0).get_data()
model = TimeConditionedMLP(in_dim=2, prediction_type="average_velocity")
ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
path = LinearInterpolant()
loss_fn = ShortcutLoss(target_clip=4.0, reduction="none")
fm_loss_fn = FlowMatchingLoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

n = BATCH_SIZE // 8
k = torch.arange(LOG2_M - 1, -1, -1).repeat_interleave(n // LOG2_M)
k = torch.cat([k, k.new_zeros(n - len(k))])
d = 0.5 ** (k + 1)

for _ in range(N_STEPS):
    x1 = data[torch.randint(len(data), (BATCH_SIZE,))]
    x0 = torch.randn_like(x1)
    t = torch.floor(torch.rand(n) * 2.0**k) / 2.0**k
    x_t, _ = path(x0[:n], x1[:n], t)
    with torch.no_grad():
        u1 = ema(x_t, t + d, s=t)
        x_mid = (x_t + d[:, None] * u1).clamp(-4, 4)
        u2 = ema(x_mid, t + 2 * d, s=t + d)
    t_fm = torch.randint(M, (BATCH_SIZE - n,)) / M
    x_fm, v = path(x0[n:], x1[n:], t_fm)
    loss = torch.cat(
        [
            loss_fn(model(x_t, t + 2 * d, s=t), u1, u2),
            fm_loss_fn(model(x_fm, t_fm + 1 / M, s=t_fm), v),
        ]
    ).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ema.update_parameters(model)

torch.manual_seed(1)
x0 = torch.randn(N_GEN, 2)
sampler = FlowMapSampler(prediction_type="average_velocity")
for n_steps in (1, 2, 4, 128):
    samples = sampler(ema.module, x0, n_steps=n_steps)
    print(f"{n_steps:3d} steps: median dist to data {median_dist(samples, data):.3f}")
