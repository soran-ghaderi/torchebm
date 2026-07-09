"""Optional visualization for equilibrium-matching (needs: pip install matplotlib)."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from torchebm.datasets import TwoMoonsDataset
from torchebm.losses import EquilibriumMatchingLoss
from torchebm.samplers import FlowSampler


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


data = TwoMoonsDataset(n_samples=4000, noise=0.05, seed=0).get_data()
model = Field()
loss_fn = EquilibriumMatchingLoss(model=model, interpolant="linear", energy_type="dot")
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
for _ in range(3000):
    batch = data[torch.randint(len(data), (256,))]
    loss = loss_fn(batch)
    opt.zero_grad()
    loss.backward()
    opt.step()

flow = FlowSampler(
    model=model, interpolant="linear", negate_velocity=True, integrator="euler"
)
samples = flow.sample_ode(torch.randn(4000, 2), num_steps=100).numpy()
d = data.numpy()

fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4), dpi=130)
a1.scatter(d[:, 0], d[:, 1], s=4, c="#9aa0aa", alpha=0.4)
a1.set_title("data")
a2.scatter(samples[:, 0], samples[:, 1], s=4, c="#C7FF00", alpha=0.4)
a2.set_title("generated (EqM + ODE)")
for a in (a1, a2):
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])

fig.tight_layout()
out = Path(__file__).parent / "assets" / "output.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
