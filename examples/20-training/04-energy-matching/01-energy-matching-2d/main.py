"""Energy Matching: one scalar potential for OT transport and Boltzmann sampling.

EM (arXiv:2504.10612) trains a time-independent potential V(x) in two phases:
a warm-up where -grad V regresses onto minibatch-OT displacements, then a joint
phase that adds contrastive divergence with temperature-scheduled Langevin
negatives, sharpening exp(-V/eps_max) near the data. Generation is one SDE
sweep: deterministic transport below tau*, a Langevin ramp up to eps_max.

The loop is the standard one: couple, interpolate, call the model, loss. The
model returns (V(x), -grad V(x)) from one call, so the flow term, the energies
of the contrastive term and the samplers all read the same network. Sinkhorn
OT pairs each batch in milliseconds at near-exact quality (reg=0.01;
ExactOTCoupling is exact but ~0.5 s per step). The negatives come half from
noise, chains sweeping the temperature profile eps(t), and half from data,
chains held at eps_max; the flow term is gated at tau* (flow_weight_cutoff).

This reproduces the paper's 2D toy experiment (two moons; tau*=0.8,
eps_max=0.15, 200-step Langevin negatives, dt=0.01, lambda_cd=2.0, lr=1e-4,
20000 flow + 2000 joint steps, 0.1 smoothing noise on x_t, plain CD without
trim/clamp). A healthy run populates both moons ~50/50 with samples hugging
the manifold (median distance ~0.1; the data itself measures ~0.03). A GPU is
recommended (~7 minutes end to end; CPU is much slower).
"""

import math
import os

import torch
from torch import nn

from torchebm.core import BaseModel, TemperatureScheduler
from torchebm.couplings import SinkhornCoupling
from torchebm.datasets import TwoMoonsDataset
from torchebm.interpolants import LinearInterpolant
from torchebm.losses import EnergyMatchingLoss
from torchebm.samplers import LangevinDynamics


class MLP(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def progress(label, step, total, loss):
    """Inline terminal progress bar (stdlib only)."""
    if step % 100 and step != total - 1:
        return
    bar = "=" * int(30 * (step + 1) / total)
    end = "\n" if step == total - 1 else ""
    print(
        f"\r{label} [{bar:<30}] {step + 1:5d}/{total}  loss {loss.item():.4f}",
        end=end, flush=True,
    )


SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
FLOW_STEPS, JOINT_STEPS = (20, 5) if SMOKE else (20000, 2000)
LANGEVIN_STEPS = 10 if SMOKE else 200
SWEEP_STEPS = 20 if SMOKE else 200
EPS_MAX, TAU_STAR, DT = 0.15, 0.8, 0.01

device = "cuda" if torch.cuda.is_available() else "cpu"
data = TwoMoonsDataset(n_samples=8000, noise=0.05, seed=0, device=device).get_data()

model = MLP(prediction_type=("energy", "score")).to(device)
coupling = SinkhornCoupling(reg=0.01)
path = LinearInterpolant()
loss_fn = EnergyMatchingLoss(lambda_cd=2.0, cd_trim_fraction=0.0, cd_clamp=None)
langevin_sweep = LangevinDynamics(
    step_size=DT,
    noise_scale=TemperatureScheduler(
        epsilon_max=EPS_MAX, tau_star=TAU_STAR, n_steps=LANGEVIN_STEPS
    ),
)
langevin_hold = LangevinDynamics(step_size=DT, noise_scale=math.sqrt(EPS_MAX))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def compute_loss(x1, contrastive):
    x0, x1 = coupling(torch.randn_like(x1), x1)
    t = torch.rand(len(x1), device=device)
    x_t, u_t = path(x0, x1, t)
    x_t = x_t + 0.1 * torch.randn_like(x_t)
    _, score = model(x_t)
    if not contrastive:
        return loss_fn(score, u_t, t)
    half = len(x1) // 2
    negatives = torch.cat([
        langevin_sweep(model, x0[:half], n_steps=LANGEVIN_STEPS),
        langevin_hold(model, x1[half:], n_steps=LANGEVIN_STEPS),
    ])
    energy_pos, _ = model(x1)
    energy_neg, _ = model(negatives)
    return loss_fn(score, u_t, t, energy_pos, energy_neg)


for phase, n_steps in ((1, FLOW_STEPS), (2, JOINT_STEPS)):
    for step in range(n_steps):
        batch = data[torch.randint(len(data), (128,), device=device)]
        loss = compute_loss(batch, contrastive=phase == 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress(f"phase {phase}", step, n_steps, loss)

temperature = TemperatureScheduler(
    epsilon_max=EPS_MAX, tau_star=TAU_STAR, n_steps=SWEEP_STEPS, t_end=1.0
)
sampler = LangevinDynamics(step_size=DT, noise_scale=temperature)
samples = sampler(model, torch.randn(4000, 2, device=device), n_steps=SWEEP_STEPS)


def moon_distance(p):
    """Distance to the nearest point of the two analytic moon arcs."""
    dists = []
    for cx, cy, upper in ((0.0, 0.0, True), (1.0, 0.5, False)):
        v = p - torch.tensor([cx, cy], device=p.device)
        ang = torch.atan2(v[:, 1], v[:, 0])
        ang = ang.clamp(0.0, torch.pi) if upper else ang.clamp(-torch.pi, 0.0)
        arc = torch.stack([ang.cos() + cx, ang.sin() + cy], dim=-1)
        dists.append((p - arc).norm(dim=1))
    return torch.stack(dists, dim=1).min(dim=1)


near = moon_distance(samples)
outer = (near.indices == 0).float().mean()
print(
    f"median distance to moons: {near.values.median():.3f}"
    f"  within 0.15: {(near.values < 0.15).float().mean():.1%}"
    f"  outer/inner split: {outer:.1%}/{1 - outer:.1%}"
)
