"""Optional visualization for energy-matching-2d (needs: pip install matplotlib). Retrains, then plots."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from torch import nn

from torchebm.core import BaseModel, TemperatureScheduler
from torchebm.datasets import TwoMoonsDataset
from torchebm.couplings import SinkhornCoupling
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


device = "cuda" if torch.cuda.is_available() else "cpu"
data = TwoMoonsDataset(n_samples=8000, noise=0.05, seed=0, device=device).get_data()
model = MLP(prediction_type=("energy", "score")).to(device)
coupling = SinkhornCoupling(reg=0.01)
path = LinearInterpolant()
loss_fn = EnergyMatchingLoss(lambda_cd=2.0, cd_trim_fraction=0.0, cd_clamp=None)
langevin_sweep = LangevinDynamics(
    step_size=0.01,
    noise_scale=TemperatureScheduler(epsilon_max=0.15, tau_star=0.8, n_steps=200),
)
langevin_hold = LangevinDynamics(step_size=0.01, noise_scale=math.sqrt(0.15))
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
        langevin_sweep(model, x0[:half], n_steps=200),
        langevin_hold(model, x1[half:], n_steps=200),
    ])
    energy_pos, _ = model(x1)
    energy_neg, _ = model(negatives)
    return loss_fn(score, u_t, t, energy_pos, energy_neg)


for step in range(22000):
    batch = data[torch.randint(len(data), (128,), device=device)]
    loss = compute_loss(batch, contrastive=step >= 20000)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 4000 == 0:
        print(f"step {step:5d}  loss {loss.item():.4f}")

model.prediction_type = "energy"
temperature = TemperatureScheduler(epsilon_max=0.15, tau_star=0.8, n_steps=200, t_end=1.0)
sampler = LangevinDynamics(step_size=0.01, noise_scale=temperature)
samples = sampler(model, torch.randn(3000, 2, device=device), n_steps=200)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
RED = "#e34948"
FIELD = LinearSegmentedColormap.from_list(
    "energy",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": GRID, "axes.titlesize": 10,
    "axes.titleweight": "semibold", "axes.titlecolor": INK,
})

xs = torch.linspace(-1.6, 2.6, 200)
ys = torch.linspace(-1.3, 1.8, 200)
gx, gy = torch.meshgrid(xs, ys, indexing="xy")
grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).to(device)
with torch.no_grad():
    e = model(grid).reshape(200, 200).cpu()

fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=130)
cs = ax.contourf(gx, gy, e, levels=40, cmap=FIELD)
s = samples.cpu().numpy()
ax.scatter(s[:, 0], s[:, 1], s=3, c=RED, alpha=0.4, lw=0)
ax.set_aspect("equal")
ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(ys[0], ys[-1])
ax.set_title("Energy Matching: V(x) and samples")
ax.set_xticks([])
ax.set_yticks([])
cbar = fig.colorbar(cs, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("V(x)", color=INK2, fontsize=8)
cbar.ax.tick_params(labelsize=7, colors=INK2)
cbar.outline.set_edgecolor(GRID)

out = Path(__file__).parent / "assets" / "output.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
