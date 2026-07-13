"""Energy Matching: one scalar potential for OT transport and Boltzmann sampling.

EM (arXiv:2504.10612) trains a time-independent potential V(x) in two phases:
a warm-up where -grad V regresses onto minibatch-OT displacements (lambda_cd=0),
then a joint phase where contrastive divergence with temperature-scheduled
Langevin negatives sharpens exp(-V/eps_max) near the data. Generation is one
SDE sweep: deterministic transport below tau*, a Langevin ramp up to eps_max.

This reproduces the paper's 2D toy experiment (two moons; tau*=0.8,
eps_max=0.15, 200-step Langevin negatives, dt=0.01, lambda_cd=2.0, lr=1e-4,
20000 flow + 2000 joint steps, plain CD without trim/clamp). A GPU is
recommended (~7 minutes end to end; CPU is much slower).
"""

import os

import torch
from torch import nn

from torchebm.core import BaseModel, TemperatureScheduler
from torchebm.datasets import TwoMoonsDataset
from torchebm.couplings import SinkhornCoupling
from torchebm.losses import EnergyMatchingLoss
from torchebm.samplers import LangevinDynamics


class Potential(BaseModel):  # V(x) -> (B,); time-independent scalar energy
    def __init__(self):
        super().__init__()
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
LANGEVIN_STEPS = 10 if SMOKE else 200  # negatives inside the CD term
SWEEP_STEPS = 20 if SMOKE else 200     # the generation SDE sweep

device = "cuda" if torch.cuda.is_available() else "cpu"
data = TwoMoonsDataset(n_samples=8000, noise=0.05, seed=0, device=device).get_data()

model = Potential().to(device)
# Sinkhorn OT coupling: milliseconds per batch at near-exact quality
# (reg=0.01). method="exact" (auction EMD) is exact but ~0.5 s per step.
loss_fn = EnergyMatchingLoss(
    model=model,
    coupling=SinkhornCoupling(reg=0.01),
    lambda_cd=0.0,           # phase 1: pure OT flow matching, no Langevin
    epsilon_max=0.15,
    tau_star=0.8,
    n_langevin_steps=LANGEVIN_STEPS,
    langevin_dt=0.01,
    cd_trim_fraction=0.0,    # 2D config: plain CD (trim/clamp are image-scale knobs)
    cd_clamp=None,
    device=device,
)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

for phase, n_steps in ((1, FLOW_STEPS), (2, JOINT_STEPS)):
    if phase == 2:
        loss_fn.lambda_cd = 2.0  # phase 2: contrastive sharpening
    for step in range(n_steps):
        batch = data[torch.randint(len(data), (128,), device=device)]
        loss = loss_fn(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        progress(f"phase {phase}", step, n_steps, loss)

# Generation, paper toy recipe: one sweep t: 0 -> 1 over 200 steps (dt = 0.01).
# Noise-free gradient flow below tau*, then a Langevin ramp up to eps_max.
temp = TemperatureScheduler(epsilon_max=0.15, tau_star=0.8, n_steps=SWEEP_STEPS, t_end=1.0)
sampler = LangevinDynamics(model=model, step_size=0.01, noise_scale=temp, device=device)
samples = sampler.sample(x=torch.randn(4000, 2, device=device), n_steps=SWEEP_STEPS)


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


# Reproduction check: both moons populated ~50/50 and samples hugging the
# manifold (median distance ~0.1; the data itself measures ~0.03).
near = moon_distance(samples)
outer = (near.indices == 0).float().mean()
print(
    f"median distance to moons: {near.values.median():.3f}"
    f"  within 0.15: {(near.values < 0.15).float().mean():.1%}"
    f"  outer/inner split: {outer:.1%}/{1 - outer:.1%}"
)
