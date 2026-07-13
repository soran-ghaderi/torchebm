"""Energy Matching paper suite: 8 Gaussians -> Two Moons with diagnostics.

Reproduces the 2D experiments of arXiv:2504.10612 on the paper's own toy
datasets: an 8-Gaussian ring is transported onto two moons by one
time-independent potential V(x), trained with the two-phase EM recipe
(OT flow warm-up, then contrastive sharpening). Beyond generation quality,
this covers the paper's toy diagnostics: local intrinsic dimension from the
Hessian of V (their Sec. on LID) and diverse sampling with the repulsive
interaction energy W. Run plot.py afterwards for the full figure suite.

A GPU is recommended (~8 minutes end to end; CPU is much slower).
"""

import math
import os

import torch
from torch import nn
from torch.func import hessian, vmap

from torchebm.core import BaseModel, TemperatureScheduler
from torchebm.datasets import GaussianMixtureDataset, TwoMoonsDataset
from torchebm.couplings import SinkhornCoupling
from torchebm.losses import EnergyMatchingLoss
from torchebm.models import InteractionModel
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


SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
FLOW_STEPS, JOINT_STEPS = (20, 5) if SMOKE else (20000, 2000)
LANGEVIN_STEPS = 10 if SMOKE else 200  # negatives inside the CD term
SWEEP_STEPS = 20 if SMOKE else 200     # the generation SDE sweep

device = "cuda" if torch.cuda.is_available() else "cpu"

# The paper's toy datasets: source = 8-Gaussian ring, target = two moons.
source = GaussianMixtureDataset(
    n_samples=8000, n_components=8, std=0.1, radius=2.5, seed=0, device=device
).get_data()
target = TwoMoonsDataset(n_samples=8000, noise=0.05, seed=1, device=device).get_data()

model = Potential().to(device)
loss_fn = EnergyMatchingLoss(
    model=model,
    coupling=SinkhornCoupling(reg=0.01),
    lambda_cd=0.0,
    epsilon_max=0.15,
    tau_star=0.8,
    n_langevin_steps=LANGEVIN_STEPS,
    langevin_dt=0.01,
    cd_trim_fraction=0.0,
    cd_clamp=None,
    device=device,
)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)


def batches():
    i1 = torch.randint(len(target), (128,), device=device)
    i0 = torch.randint(len(source), (128,), device=device)
    return target[i1], source[i0]


for step in range(FLOW_STEPS):  # phase 1: OT flow matching (lambda_cd = 0)
    x1, x0 = batches()
    loss = loss_fn(x1, x0=x0)
    opt.zero_grad()
    loss.backward()
    opt.step()
    progress("phase 1 (OT flow)", step, FLOW_STEPS, loss)

loss_fn.lambda_cd = 2.0
for step in range(JOINT_STEPS):  # phase 2: joint flow + contrastive sharpening
    x1, x0 = batches()
    loss = loss_fn(x1, x0=x0)
    opt.zero_grad()
    loss.backward()
    opt.step()
    progress("phase 2 (EM)     ", step, JOINT_STEPS, loss)

# --- Generation: one SDE sweep t: 0 -> 1 starting FROM THE SOURCE. ---------
temp = TemperatureScheduler(epsilon_max=0.15, tau_star=0.8, n_steps=SWEEP_STEPS, t_end=1.0)
sampler = LangevinDynamics(model=model, step_size=0.01, noise_scale=temp, device=device)
start = source[torch.randperm(len(source), device=device)[:4000]]
samples = sampler.sample(x=start, n_steps=SWEEP_STEPS)

near = moon_distance(samples)
outer = (near.indices == 0).float().mean()
print(
    f"transport 8 Gaussians -> moons: median distance {near.values.median():.3f}"
    f"  within 0.15: {(near.values < 0.15).float().mean():.1%}"
    f"  outer/inner split: {outer:.1%}/{1 - outer:.1%}"
)

# --- Local intrinsic dimension from the Hessian of V (paper Sec. 3.3). -----
# LID = d - rank(grad^2 V) at data points: flat (small-eigenvalue) directions
# are tangent to the manifold. Rank uses a per-point relative threshold; it is
# meaningful on the density ridge, where V is convex. The moons are curves,
# so the true LID is 1.
def hessian_eigs(points):
    v_single = lambda p: model(p.unsqueeze(0)).squeeze(0)
    return torch.linalg.eigvalsh(vmap(hessian(v_single))(points)).detach()


eig_data = hessian_eigs(target[:512])
rank = (eig_data > 0.3 * eig_data[:, 1:].clamp(min=1e-6)).sum(dim=1)
lid_data = 2.0 - rank.float()
box = torch.rand(512, 2, device=device) * 4.0 - torch.tensor([1.5, 1.5], device=device)
eig_box = hessian_eigs(box)
print(
    f"LID at data points: mean {lid_data.mean():.2f} (true manifold dim: 1)"
    f"   stiff-eigenvalue medians: data {eig_data[:, 1].median():.1f}"
    f" vs off-manifold {eig_box[:, 1].median():.1f}"
)

# --- Diverse generation: repulsive interaction energy W. -------------------
# 64 chains from ONE point at eps_max; the repulsion spreads them along the
# manifold (the paper's inverse-design mechanism).
one_point = target[:1].expand(64, -1).contiguous()
plain = LangevinDynamics(
    model=model, step_size=0.01, noise_scale=math.sqrt(0.15), device=device
).sample(x=one_point, n_steps=SWEEP_STEPS)
repulsive = InteractionModel(model, sigma_w=4.0, strength=0.15)
diverse = LangevinDynamics(
    model=repulsive, step_size=0.01, noise_scale=math.sqrt(0.15), device=device
).sample(x=one_point, n_steps=SWEEP_STEPS)


def mean_pairwise(x):
    return torch.pdist(x).mean()


print(
    f"diversity (mean pairwise distance) from one seed point:"
    f"  plain {mean_pairwise(plain):.2f}  vs  repulsive {mean_pairwise(diverse):.2f}"
)
