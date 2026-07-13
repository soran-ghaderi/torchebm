"""Optional visualization for energy-matching-2d (needs: pip install matplotlib). Retrains, then plots."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from torch import nn

from torchebm.core import BaseModel, TemperatureScheduler
from torchebm.datasets import TwoMoonsDataset
from torchebm.couplings import SinkhornCoupling
from torchebm.losses import EnergyMatchingLoss
from torchebm.samplers import LangevinDynamics


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


device = "cuda" if torch.cuda.is_available() else "cpu"
data = TwoMoonsDataset(n_samples=8000, noise=0.05, seed=0, device=device).get_data()
model = Potential().to(device)
loss_fn = EnergyMatchingLoss(
    model=model, coupling=SinkhornCoupling(reg=0.01), lambda_cd=0.0,
    epsilon_max=0.15, tau_star=0.8, n_langevin_steps=200, langevin_dt=0.01,
    cd_trim_fraction=0.0, cd_clamp=None, device=device,
)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
for step in range(22000):
    if step == 20000:
        loss_fn.lambda_cd = 2.0
    batch = data[torch.randint(len(data), (128,), device=device)]
    loss = loss_fn(batch)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 4000 == 0:
        print(f"step {step:5d}  loss {loss.item():.4f}")

temp = TemperatureScheduler(epsilon_max=0.15, tau_star=0.8, n_steps=200, t_end=1.0)
sampler = LangevinDynamics(model=model, step_size=0.01, noise_scale=temp, device=device)
samples = sampler.sample(x=torch.randn(3000, 2, device=device), n_steps=200)

# Visual system shared with the paper-suite example: light-to-dark blue energy
# field (low energy recedes into the surface), model samples in red.
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
