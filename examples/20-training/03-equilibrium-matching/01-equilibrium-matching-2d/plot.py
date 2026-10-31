"""Optional visualization for equilibrium-matching (needs: pip install matplotlib).

Trains the implicit field and the explicit energy, then draws the coherent
sampling routes side by side: the probability-flow ODE and gradient ascent on
the field both generate the data, and a repulsive InteractionModel around the
energy fans a single seed out along the manifold.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from torchebm.core import BaseModel
from torchebm.datasets import TwoMoonsDataset
from torchebm.interpolants import LinearInterpolant
from torchebm.losses import EquilibriumMatchingLoss
from torchebm.models import InteractionModel
from torchebm.samplers import FlowSampler, GradientDescentSampler


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x, t=None, **kwargs):
        return self.net(x)


class EnergyMLP(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(model, steps=3000):
    torch.manual_seed(0)
    data = TwoMoonsDataset(n_samples=4000, noise=0.05, seed=0).get_data()
    path = LinearInterpolant()
    loss_fn = EquilibriumMatchingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        x1 = data[torch.randint(len(data), (256,))]
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1))
        x_t, u_t = path(x0, x1, t)
        loss = loss_fn(model(x_t), u_t, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return data


model = MLP()
data = train(model)
energy_model = EnergyMLP(prediction_type="score")
train(energy_model)
x0 = torch.randn(4000, 2)

samples_ode = FlowSampler(integrator="euler")(model, x0, n_steps=100)
samples_gd = GradientDescentSampler(step_size=0.05)(model, x0, n_steps=200)

x_seed = torch.tensor([0.5, 1.6]) + 1e-3 * torch.randn(64, 2)
energy_model.prediction_type = "energy"
interaction_model = InteractionModel(energy_model, sigma_w=4.5, strength=0.15)
samples_repulsive = GradientDescentSampler(step_size=0.01)(
    interaction_model, x_seed, n_steps=200
)

panels = [
    ("data", data.numpy(), "#9aa0aa"),
    ("implicit + ODE", samples_ode.numpy(), "#C7FF00"),
    ("implicit + gradient ascent", samples_gd.numpy(), "#00C7A6"),
    ("diverse (repulsion W)", samples_repulsive.numpy(), "#e11212"),
]
fig, axes = plt.subplots(1, 4, figsize=(15, 4), dpi=130)
for ax, (title, pts, color) in zip(axes, panels):
    ax.scatter(data.numpy()[:, 0], data.numpy()[:, 1], s=4, c="#dfe3e8", alpha=0.35)
    ax.scatter(pts[:, 0], pts[:, 1], s=6, c=color, alpha=0.6)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
out = Path(__file__).parent / "assets" / "output.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
