"""Energy Matching paper suite: 8 Gaussians -> Two Moons with diagnostics.

Reproduces the 2D experiments of arXiv:2504.10612 on the paper's own toy
datasets: an 8-Gaussian ring is transported onto two moons by one
time-independent potential V(x), trained with the two-phase EM recipe
(OT flow warm-up, then contrastive sharpening). Beyond generation quality,
this covers the paper's toy diagnostics: local intrinsic dimension from the
Hessian of V (their Sec. 3.3) and diverse sampling with the repulsive
interaction energy W, which fans 64 chains started from one data point out
along the manifold (the paper's inverse-design mechanism). Run plot.py
afterwards for the full figure suite.

Training calls the model for (V(x), -grad V(x)) at once; afterwards it
switches to its energy view, which the samplers differentiate, the Hessian
diagnostics read directly and InteractionModel wraps. Generation is one SDE
sweep t: 0 -> 1 started from the source distribution.

A GPU is recommended (~8 minutes end to end; CPU is much slower).
"""

import math
import os

import torch
from torch import nn
from torch.func import hessian, vmap

from torchebm.core import BaseModel, TemperatureScheduler
from torchebm.couplings import SinkhornCoupling
from torchebm.datasets import GaussianMixtureDataset, TwoMoonsDataset
from torchebm.interpolants import LinearInterpolant
from torchebm.losses import EnergyMatchingLoss
from torchebm.models import InteractionModel
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
LANGEVIN_STEPS = 10 if SMOKE else 200
SWEEP_STEPS = 20 if SMOKE else 200
EPS_MAX, TAU_STAR, DT = 0.15, 0.8, 0.01

device = "cuda" if torch.cuda.is_available() else "cpu"

source = GaussianMixtureDataset(
    n_samples=8000, n_components=8, std=0.1, radius=2.5, seed=0, device=device
).get_data()
target = TwoMoonsDataset(n_samples=8000, noise=0.05, seed=1, device=device).get_data()

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


def next_batch():
    i1 = torch.randint(len(target), (128,), device=device)
    i0 = torch.randint(len(source), (128,), device=device)
    return target[i1], source[i0]


def compute_loss(x1, x0, contrastive):
    x0, x1 = coupling(x0, x1)
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


for step in range(FLOW_STEPS):
    x1, x0 = next_batch()
    loss = compute_loss(x1, x0, contrastive=False)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    progress("phase 1 (OT flow)", step, FLOW_STEPS, loss)

for step in range(JOINT_STEPS):
    x1, x0 = next_batch()
    loss = compute_loss(x1, x0, contrastive=True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    progress("phase 2 (EM)     ", step, JOINT_STEPS, loss)

model.prediction_type = "energy"

temperature = TemperatureScheduler(
    epsilon_max=EPS_MAX, tau_star=TAU_STAR, n_steps=SWEEP_STEPS, t_end=1.0
)
sampler = LangevinDynamics(step_size=DT, noise_scale=temperature)
x_init = source[torch.randperm(len(source), device=device)[:4000]]
samples = sampler(model, x_init, n_steps=SWEEP_STEPS)

near = moon_distance(samples)
outer = (near.indices == 0).float().mean()
print(
    f"transport 8 Gaussians -> moons: median distance {near.values.median():.3f}"
    f"  within 0.15: {(near.values < 0.15).float().mean():.1%}"
    f"  outer/inner split: {outer:.1%}/{1 - outer:.1%}"
)


def hessian_eigs(points):
    """Eigenvalues of the Hessian of V at `points`.

    LID = d - rank(grad^2 V): flat (small-eigenvalue) directions are tangent to
    the manifold. The rank uses a per-point relative threshold and is
    meaningful on the density ridge, where V is convex. The moons are curves,
    so the true LID is 1.
    """
    v_single = lambda p: model(p.unsqueeze(0)).squeeze(0)
    return torch.linalg.eigvalsh(vmap(hessian(v_single))(points)).detach()


eig_data = hessian_eigs(target[:512])
rank = (eig_data > 0.3 * eig_data[:, 1:].clamp(min=1e-6)).sum(dim=1)
lid_data = 2.0 - rank.float()
x_random = torch.rand(512, 2, device=device) * 4.0 - torch.tensor([1.5, 1.5], device=device)
eig_random = hessian_eigs(x_random)
print(
    f"LID at data points: mean {lid_data.mean():.2f} (true manifold dim: 1)"
    f"   stiff-eigenvalue medians: data {eig_data[:, 1].median():.1f}"
    f" vs off-manifold {eig_random[:, 1].median():.1f}"
)

x_seed = target[:1].expand(64, -1).contiguous()
samples_plain = langevin_hold(model, x_seed, n_steps=SWEEP_STEPS)
interaction_model = InteractionModel(model, sigma_w=4.0, strength=0.15)
samples_repulsive = langevin_hold(interaction_model, x_seed, n_steps=SWEEP_STEPS)


def mean_pairwise(x):
    return torch.pdist(x).mean()


print(
    f"diversity (mean pairwise distance) from one seed point:"
    f"  plain {mean_pairwise(samples_plain):.2f}"
    f"  vs  repulsive {mean_pairwise(samples_repulsive):.2f}"
)
