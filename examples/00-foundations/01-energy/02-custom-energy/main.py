"""Custom energies: a differentiable BaseModel subclass has an autograd score and works with every sampler.

    E(x) defines p(x) proportional to exp(-E(x)); anything that maps (N, d) -> (N,)
    and is differentiable is a valid energy, analytic or neural.

Built with prediction_type=("energy", "score"), the call returns the energy and
the autograd score -grad E together. The ring's minimum is the circle |x| = 2:
zero score on the ring, a strong pull outside.
"""

import torch
from torch import nn

from torchebm.core import BaseModel, GaussianModel


class RingEnergy(BaseModel):
    def __init__(self, radius=2.0, width=0.25, **kwargs):
        super().__init__(**kwargs)
        self.radius, self.width = radius, width

    def forward(self, x):
        return (x.norm(dim=-1) - self.radius) ** 2 / (2 * self.width**2)


class MLP(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


points = torch.tensor([[2.0, 0.0], [0.0, 0.0], [3.0, 0.0]])

for name, model in {
    "ring": RingEnergy(prediction_type=("energy", "score")),
    "gaussian": GaussianModel(
        mean=torch.zeros(2), cov=torch.eye(2), prediction_type=("energy", "score")
    ),
    "mlp (untrained)": MLP(prediction_type=("energy", "score")),
}.items():
    energy, score = model(points)
    print(f"{name:16s} E = {[round(e, 2) for e in energy.tolist()]}"
          f"   |score| = {[round(s, 2) for s in score.norm(dim=1).tolist()]}")
