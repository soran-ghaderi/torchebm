"""Improved MeanFlow in 2D: one-step class-conditional generation with guidance as a model input.

Improved MeanFlow (Geng et al., 2025b) trains a MeanFlow model on the guided
velocity

    v_g = v + (1 - 1/w) (v_c - v_u),

where v_c and v_u are the model's own conditional and unconditional
velocities u(x_s, s, s), and gives the model the guidance scale w and the
guidance interval as inputs (Algorithm 2). One model then samples at any
scale in one step, with one forward pass per step. The JVP runs along the
model's conditional velocity. As in the reference recipe (Lyy-iiis/imeanflow):
50% of the time pairs with s < t, w drawn from p(w) ~ 1/w on [1, 8], the
interval start uniform on [0, 1/2] and its end on [1/2, 1], no guidance
outside the interval, 10% of the labels dropped, the adaptive weight with
p = 1, and Adam with betas (0.9, 0.95). The ImageNet models also train an
auxiliary velocity head (imf_dit_b_2 and the larger presets); this example
uses u(x_s, s, s), the variant of Algorithm 1.

The data are eight Gaussians on the unit circle, one class each. Samples in
one step at w = 1, 2 and 4 are scored by the share that lands nearest to the
center of its own class and by the median distance to that center, next to
the same scores of the data.
"""

import math
import os

import torch

from torchebm.interpolants import LinearInterpolant
from torchebm.losses import MeanFlowLoss
from torchebm.models import TimeConditionedMLP
from torchebm.samplers import FlowMapSampler
from torchebm.utils import average_velocity_jvp, sample_guidance, sample_time_pairs

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 6000
N_GEN = 128 if SMOKE else 2000
BATCH_SIZE = 256
N_CLASSES = 8
STD = 0.2

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
angles = 2 * math.pi * torch.arange(N_CLASSES, device=device) / N_CLASSES
centers = torch.stack([angles.cos(), angles.sin()], dim=1)
model = TimeConditionedMLP(
    in_dim=2,
    num_classes=N_CLASSES,
    guidance_conditioning="scale_and_interval",
    time_invariant=True,
    prediction_type="average_velocity",
).to(device)
path = LinearInterpolant()
loss_fn = MeanFlowLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
null = torch.full((BATCH_SIZE,), N_CLASSES, device=device)
full_interval = torch.tensor([0.0, 1.0], device=device).expand(2 * BATCH_SIZE, 2)

for _ in range(N_STEPS):
    y = torch.randint(N_CLASSES, (BATCH_SIZE,), device=device)
    x1 = centers[y] + STD * torch.randn(BATCH_SIZE, 2, device=device)
    s, t = sample_time_pairs(BATCH_SIZE, flow_matching_ratio=0.5, device=device)
    scale, interval = sample_guidance(BATCH_SIZE, flow_matching_ratio=0.5, device=device)
    x_s, v = path(torch.randn_like(x1), x1, s)
    inside = (s >= interval[:, 0]) & (s <= interval[:, 1])
    w = torch.where(inside, scale, 1.0)
    with torch.no_grad():
        v_c, v_u = model(
            x_s.repeat(2, 1),
            s.repeat(2),
            torch.cat([y, null]),
            s=s.repeat(2),
            guidance_scale=torch.cat([w, torch.ones_like(w)]),
            guidance_interval=full_interval,
        ).chunk(2)
    v_guided = v + (1 - 1 / w).unsqueeze(1) * (v_c - v_u)
    drop = torch.arange(BATCH_SIZE, device=device) < (torch.rand(BATCH_SIZE, device=device) < 0.1).sum()
    y = torch.where(drop, null, y)
    v_guided = torch.where(drop.unsqueeze(1), v, v_guided)
    u, du_ds = average_velocity_jvp(
        model,
        x_s,
        s,
        t,
        v_c,
        y,
        flow_matching_ratio=0.5,
        model_kwargs={"guidance_scale": scale, "guidance_interval": interval},
    )
    loss = loss_fn(u, du_ds, v_guided, s, t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def scores(samples, y):
    """Share of samples nearest to their class center, and median distance to it."""
    accuracy = (torch.cdist(samples, centers).argmin(dim=1) == y).float().mean().item()
    return accuracy, (samples - centers[y]).norm(dim=1).median().item()


torch.manual_seed(1)
x0 = torch.randn(N_GEN, 2, device=device)
y = torch.arange(N_GEN, device=device) % N_CLASSES
accuracy, distance = scores(centers[y] + STD * torch.randn(N_GEN, 2, device=device), y)
print(f" data: {accuracy:.1%} nearest to their class center, median distance {distance:.3f}")
sampler = FlowMapSampler(prediction_type="average_velocity")
for omega in (1.0, 2.0, 4.0):
    model_kwargs = {
        "y": y,
        "guidance_scale": torch.full((N_GEN,), omega, device=device),
        "guidance_interval": torch.tensor([0.0, 1.0], device=device).expand(N_GEN, 2),
    }
    accuracy, distance = scores(sampler(model, x0, n_steps=1, model_kwargs=model_kwargs), y)
    print(f"w = {omega:.0f}: {accuracy:.1%} nearest to their class center, median distance {distance:.3f}")
