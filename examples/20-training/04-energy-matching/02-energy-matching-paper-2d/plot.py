"""Figure suite for energy-matching-paper-2d (needs: pip install matplotlib). Retrains, then plots.

Shared visual system: energy is a light-to-dark blue field (low energy recedes
into the surface so samples read instantly), data is blue, everything the model
produces is red, references (source, off-manifold) are gray.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from torch import nn
from torch.func import hessian, vmap

from torchebm.core import BaseModel, TemperatureScheduler
from torchebm.datasets import GaussianMixtureDataset, TwoMoonsDataset
from torchebm.couplings import SinkhornCoupling
from torchebm.losses import EnergyMatchingLoss
from torchebm.models import InteractionModel
from torchebm.samplers import LangevinDynamics

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, RED, GRAY = "#2a78d6", "#e34948", "#898781"  # data / model output / reference
FIELD = LinearSegmentedColormap.from_list(
    "energy",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "legend.frameon": False,
})


class Potential(BaseModel):
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
    if step % 100 and step != total - 1:
        return
    bar = "=" * int(30 * (step + 1) / total)
    end = "\n" if step == total - 1 else ""
    print(
        f"\r{label} [{bar:<30}] {step + 1:5d}/{total}  loss {loss.item():.4f}",
        end=end, flush=True,
    )


device = "cuda" if torch.cuda.is_available() else "cpu"
assets = Path(__file__).parent / "assets"
assets.mkdir(parents=True, exist_ok=True)

source = GaussianMixtureDataset(
    n_samples=8000, n_components=8, std=0.1, radius=2.5, seed=0, device=device
).get_data()
target = TwoMoonsDataset(n_samples=8000, noise=0.05, seed=1, device=device).get_data()

model = Potential().to(device)
loss_fn = EnergyMatchingLoss(
    model=model, coupling=SinkhornCoupling(reg=0.01), lambda_cd=0.0,
    epsilon_max=0.15, tau_star=0.8, n_langevin_steps=200, langevin_dt=0.01,
    cd_trim_fraction=0.0, cd_clamp=None, device=device,
)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
for phase, n_steps in ((1, 20000), (2, 2000)):
    if phase == 2:
        loss_fn.lambda_cd = 2.0
    for step in range(n_steps):
        i1 = torch.randint(len(target), (128,), device=device)
        i0 = torch.randint(len(source), (128,), device=device)
        loss = loss_fn(target[i1], x0=source[i0])
        opt.zero_grad()
        loss.backward()
        opt.step()
        progress(f"phase {phase}", step, n_steps, loss)

# Generation sweep with full trajectories.
temp = TemperatureScheduler(epsilon_max=0.15, tau_star=0.8, n_steps=200, t_end=1.0)
sampler = LangevinDynamics(model=model, step_size=0.01, noise_scale=temp, device=device)
start = source[torch.randperm(len(source), device=device)[:3000]]
traj = sampler.sample(x=start, n_steps=200, return_trajectory=True)  # [N, 200, 2]
samples = traj[:, -1]

# Shared energy field.
xs = torch.linspace(-3.2, 3.6, 220)
ys = torch.linspace(-3.0, 3.2, 220)
gx, gy = torch.meshgrid(xs, ys, indexing="xy")
grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).to(device)
with torch.no_grad():
    energy = model(grid).reshape(220, 220).cpu()


def field(ax):
    cs = ax.contourf(gx, gy, energy, levels=40, cmap=FIELD)
    ax.set_aspect("equal")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_xticks([])
    ax.set_yticks([])
    return cs


def flat(ax):
    ax.set_aspect("equal")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_xticks([])
    ax.set_yticks([])


# Figure 1 (thumbnail): datasets | potential + samples | trajectories.
fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=130)
flat(axes[0])
axes[0].scatter(*source.cpu().T, s=2, c=GRAY, label="source: 8 Gaussians")
axes[0].scatter(*target.cpu().T, s=2, c=BLUE, label="target: two moons")
axes[0].legend(loc="upper right", fontsize=7, markerscale=3)
axes[0].set_title("The paper's toy datasets")
cs = field(axes[1])
axes[1].scatter(*samples.cpu().T, s=2.5, c=RED, alpha=0.4, lw=0)
axes[1].set_title("Learned potential V(x) and samples")
cbar = fig.colorbar(cs, ax=axes[1], shrink=0.85, pad=0.02)
cbar.set_label("V(x)", color=INK2, fontsize=8)
cbar.ax.tick_params(labelsize=7, colors=INK2)
cbar.outline.set_edgecolor(GRID)
field(axes[2])
paths = traj[:40].cpu()
axes[2].scatter(*paths[:, 0].T, s=10, c=GRAY, zorder=3, label="start (source)")
for path in paths:
    axes[2].plot(path[:, 0], path[:, 1], lw=0.7, c=RED, alpha=0.35)
axes[2].scatter(*paths[:, -1].T, s=10, c=RED, zorder=3, label="end (sample)")
axes[2].legend(loc="upper right", fontsize=7)
axes[2].set_title("Transport trajectories")
fig.tight_layout()
fig.savefig(assets / "output.png", bbox_inches="tight")

# Figure 2: sample evolution across the sweep.
fig, axes = plt.subplots(1, 6, figsize=(16, 3), dpi=130)
for ax, t in zip(axes, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    idx = min(int(t * 200), 199)
    field(ax)
    ax.scatter(*traj[:, idx].cpu().T, s=1.5, c=RED, alpha=0.35, lw=0)
    ax.set_title(f"t = {t:.1f}", fontsize=9)
fig.tight_layout()
fig.savefig(assets / "evolution.png", bbox_inches="tight")

# Figure 3: temperature schedule and flow gate.
t_axis = torch.linspace(0, 1.3, 400)
eps = torch.where(
    t_axis < 0.8,
    torch.zeros_like(t_axis),
    (0.15 * (t_axis - 0.8) / 0.2).clamp(max=0.15),
)
gate = ((1 - t_axis) / 0.2).clamp(0.0, 1.0)
fig, ax = plt.subplots(figsize=(5, 3), dpi=130)
ax.plot(t_axis, eps, color=RED, label=r"temperature $\epsilon(t)$", lw=2)
ax.plot(t_axis, gate * 0.15, color=BLUE, label=r"flow gate $w(t)$ (scaled)", lw=2, ls="--")
ax.axvline(0.8, c=GRAY, lw=0.8, ls=":")
ax.text(0.8, 0.157, r"$\tau^*$", ha="center", color=INK2)
ax.set_xlabel("t")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.legend(fontsize=8, loc="center left")
fig.tight_layout()
fig.savefig(assets / "schedule.png", bbox_inches="tight")

# Figure 4: local intrinsic dimension from the Hessian of V.
v_single = lambda p: model(p.unsqueeze(0)).squeeze(0)
eig_data = torch.linalg.eigvalsh(vmap(hessian(v_single))(target[:512])).detach().cpu()
box = torch.rand(512, 2, device=device) * 4.0 - torch.tensor([1.5, 1.5], device=device)
eig_box = torch.linalg.eigvalsh(vmap(hessian(v_single))(box)).detach().cpu()
rank = (eig_data > 0.3 * eig_data[:, 1:].clamp(min=1e-6)).sum(dim=1)
lid_mean = (2.0 - rank.float()).mean()
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=130)
axes[0].scatter(eig_box[:, 0], eig_box[:, 1], s=5, c=GRAY, label="off-manifold", lw=0)
axes[0].scatter(eig_data[:, 0], eig_data[:, 1], s=5, c=BLUE, label="data points", lw=0)
axes[0].set_xlabel(r"$\lambda_1(\nabla^2 V)$")
axes[0].set_ylabel(r"$\lambda_2(\nabla^2 V)$")
axes[0].spines[["top", "right"]].set_visible(False)
axes[0].legend(fontsize=8)
axes[0].set_title(f"Hessian spectrum (data LID $\\approx$ {lid_mean:.2f}, dim 1)")
bins = torch.linspace(-5, 25, 60).tolist()
axes[1].hist(eig_box[:, 1], bins=bins, alpha=0.85, label="off-manifold", color=GRAY)
axes[1].hist(eig_data[:, 1], bins=bins, alpha=0.85, label="data", color=BLUE)
axes[1].set_xlabel(r"stiff eigenvalue $\lambda_2(\nabla^2 V)$")
axes[1].spines[["top", "right"]].set_visible(False)
axes[1].legend(fontsize=8)
axes[1].set_title("Normal-direction stiffness on vs off manifold")
fig.tight_layout()
fig.savefig(assets / "lid.png", bbox_inches="tight")

# Figure 5: diverse generation via repulsive interaction energy.
one_point = target[:1].expand(64, -1).contiguous()
plain = LangevinDynamics(
    model=model, step_size=0.01, noise_scale=math.sqrt(0.15), device=device
).sample(x=one_point, n_steps=200)
diverse = LangevinDynamics(
    model=InteractionModel(model, sigma_w=4.0, strength=0.15),
    step_size=0.01, noise_scale=math.sqrt(0.15), device=device,
).sample(x=one_point, n_steps=200)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=130)
for ax, pts, title in (
    (axes[0], plain, "64 chains from one seed: plain V"),
    (axes[1], diverse, "with repulsive interaction W"),
):
    field(ax)
    ax.scatter(*pts.cpu().T, s=12, c=RED, alpha=0.85, lw=0)
    ax.scatter(
        *one_point[:1].cpu().T, marker="*", s=170, c=INK,
        edgecolors=SURFACE, linewidths=0.8, zorder=3,
    )
    ax.set_title(title, fontsize=9)
fig.tight_layout()
fig.savefig(assets / "repulsion.png", bbox_inches="tight")

print(f"wrote 5 figures to {assets}")
