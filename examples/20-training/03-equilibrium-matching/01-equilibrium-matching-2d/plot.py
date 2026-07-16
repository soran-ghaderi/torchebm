"""Optional visualization for equilibrium-matching (needs: pip install matplotlib).

Trains the implicit and explicit fields, then draws the coherent sampling
routes side by side: the probability-flow ODE and gradient descent on the
EqMEnergy landscape both generate the data, and a repulsive InteractionModel
fans a single seed out along the manifold.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from torchebm.datasets import TwoMoonsDataset
from torchebm.losses import EquilibriumMatchingLoss
from torchebm.models import EqMEnergy, InteractionModel
from torchebm.samplers import FlowSampler, GradientDescentSampler


class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x, t, **kwargs):
        return self.net(x)


def train(energy_type, steps=3000):
    torch.manual_seed(0)
    data = TwoMoonsDataset(n_samples=4000, noise=0.05, seed=0).get_data()
    field = Field()
    loss_fn = EquilibriumMatchingLoss(model=field, energy_type=energy_type)
    opt = torch.optim.Adam(field.parameters(), lr=1e-3)
    for _ in range(steps):
        batch = data[torch.randint(len(data), (256,))]
        loss = loss_fn(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return field, data


field_impl, data = train("none")
field_dot, _ = train("dot")
x0 = torch.randn(4000, 2)

ode = FlowSampler(
    field_impl, interpolant="linear", negate_velocity=True, integrator="euler"
).sample(x=x0.clone(), n_steps=100)
gd = GradientDescentSampler(
    EqMEnergy(field_impl, "implicit"), step_size=0.05
).sample(x=x0.clone(), n_steps=200)

seed = torch.tensor([0.5, 1.6]) + 1e-3 * torch.randn(64, 2)
coda = GradientDescentSampler(
    InteractionModel(EqMEnergy(field_dot, "dot"), sigma_w=4.5, strength=0.15),
    step_size=0.01,
).sample(x=seed.clone(), n_steps=200)

panels = [
    ("data", data.numpy(), "#9aa0aa"),
    ("implicit + ODE", ode.numpy(), "#C7FF00"),
    ("implicit + gradient descent", gd.numpy(), "#00C7A6"),
    ("diverse (repulsion W)", coda.numpy(), "#e11212"),
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
