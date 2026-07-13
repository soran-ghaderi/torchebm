"""Custom energies: subclass BaseModel and get gradients, sampling, and losses for free.

    E(x) defines p(x) proportional to exp(-E(x)); anything that maps (N, d) -> (N,)
    and is differentiable is a valid energy, analytic or neural.
"""

import torch
from torch import nn

from torchebm.core import BaseModel, GaussianModel


class RingEnergy(BaseModel):  # analytic: a circular valley of radius r
    def __init__(self, radius=2.0, width=0.25):
        super().__init__()
        self.radius, self.width = radius, width

    def forward(self, x):
        return (x.norm(dim=-1) - self.radius) ** 2 / (2 * self.width**2)


class MLPEnergy(BaseModel):  # neural: the same contract, learned instead of closed-form
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


points = torch.tensor([[2.0, 0.0], [0.0, 0.0], [3.0, 0.0]])  # on-ring, center, outside

for name, model in {
    "ring": RingEnergy(),
    "gaussian": GaussianModel(mean=torch.zeros(2), cov=torch.eye(2)),
    "mlp (untrained)": MLPEnergy(),
}.items():
    energy = model(points)             # (3,)  scalar energy per point
    force = -model.gradient(points)    # (3, 2)  autograd supplies -grad E (the score)
    print(f"{name:16s} E = {[round(e, 2) for e in energy.tolist()]}"
          f"   |force| = {[round(f, 2) for f in force.norm(dim=1).tolist()]}")

# The ring's minimum is the circle |x| = 2: zero force on the ring, strong pull outside.
