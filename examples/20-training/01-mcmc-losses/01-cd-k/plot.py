"""Optional visualization for cd-k (needs: pip install matplotlib). Retrains, then plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from torchebm.core import BaseModel
from torchebm.datasets import TwoMoonsDataset
from torchebm.losses import ContrastiveDivergenceLoss
from torchebm.samplers import LangevinDynamics


class MLP(BaseModel):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


data = TwoMoonsDataset(n_samples=3000, noise=0.05, seed=0).get_data()
model = MLP()
sampler = LangevinDynamics(step_size=0.1, noise_scale=1.0)
loss_fn = ContrastiveDivergenceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for _ in range(1000):
    batch = data[torch.randint(len(data), (256,))]
    negatives = sampler(model, batch, n_steps=10)
    loss = loss_fn(model(batch), model(negatives))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

xs = torch.linspace(-2.5, 2.5, 200)
gx, gy = torch.meshgrid(xs, xs, indexing="xy")
grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
with torch.no_grad():
    energy = model(grid).reshape(200, 200)

plt.figure(figsize=(4, 4), dpi=130)
plt.contourf(gx, gy, energy, levels=40, cmap="magma")
d = data.numpy()
plt.scatter(d[:, 0], d[:, 1], s=3, c="white", alpha=0.3)
plt.gca().set_aspect("equal")
plt.title("Learned energy (CD-10)")
plt.xticks([])
plt.yticks([])

out = Path(__file__).parent / "assets" / "output.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
