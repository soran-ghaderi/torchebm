"""Equilibrium Matching in 2D: one time-invariant field, two ways to sample it.

EqM trains a time-invariant model toward c(t) u_t, the path velocity from noise
to data scaled by a decay that vanishes at the data. This example trains two
models on two moons:

  - implicit: an MLP outputting the field f(x). It declares no
    prediction_type, so FlowSampler integrates it as the velocity, and
    GradientDescentSampler applies the same update x <- x + eta f(x) for more
    steps, iterating toward the field's attractors. The field is not
    constrained to be the gradient of an energy.
  - explicit (EqM-E): an MLP outputting a scalar energy E(x), trained through
    its score -grad E, so its field is a gradient by construction. Sample by
    descending E; wrap it in InteractionModel for diverse (repulsive)
    sampling.

The diverse coda starts 64 chains from one seed: plain gradient descent
collapses them to one point, the repulsive W spreads them along the low-energy
manifold. InteractionModel wraps an energy, so the explicit model switches to
its energy view and the sampler differentiates E(x) - W(x) itself; strength
keeps the per-step repulsion 2*s*B*dt/sigma_w^2 well below 1 (stable) yet
strong enough to overcome the pull to a single point.
"""

import os

import torch
from torch import nn

from torchebm.core import BaseModel
from torchebm.datasets import TwoMoonsDataset
from torchebm.interpolants import LinearInterpolant
from torchebm.losses import EquilibriumMatchingLoss
from torchebm.models import InteractionModel
from torchebm.samplers import FlowSampler, GradientDescentSampler

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 2000
N_GEN = 128 if SMOKE else 2000


class MLP(nn.Module):
    """f(x) -> R^2; EqM is time-invariant, so the sampler's t is unused."""

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
    """E(x) -> R; built with prediction_type="score" the call returns -grad E."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(model):
    """Regress the model's output onto c(t) u_t along the linear path."""
    torch.manual_seed(0)
    data = TwoMoonsDataset(n_samples=4000, noise=0.05, seed=0).get_data()
    path = LinearInterpolant()
    loss_fn = EquilibriumMatchingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(N_STEPS):
        x1 = data[torch.randint(len(data), (256,))]
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1))
        x_t, u_t = path(x0, x1, t)
        loss = loss_fn(model(x_t), u_t, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return data


def median_dist(samples, data):
    """Median distance from each sample to the nearest data point."""
    return torch.cdist(samples, data).min(dim=1).values.median().item()


model = MLP()
data = train(model)
x0 = torch.randn(N_GEN, 2)

samples_ode = FlowSampler(integrator="euler")(model, x0, n_steps=100)
samples_gd = GradientDescentSampler(step_size=0.05)(model, x0, n_steps=200)

print(f"implicit + ODE   median dist to data: {median_dist(samples_ode, data):.3f}")
print(f"implicit + GD    median dist to data: {median_dist(samples_gd, data):.3f}")

energy_model = EnergyMLP(prediction_type="score")
train(energy_model)
samples_energy = GradientDescentSampler(step_size=0.02)(energy_model, x0, n_steps=300)
print(f"explicit + GD    median dist to data: {median_dist(samples_energy, data):.3f}")

x_seed = torch.tensor([0.5, 1.6]) + 1e-3 * torch.randn(64, 2)
energy_model.prediction_type = "energy"
interaction_model = InteractionModel(energy_model, sigma_w=4.5, strength=0.5)
samples_repulsive = GradientDescentSampler(step_size=0.01)(interaction_model, x_seed, n_steps=200)
samples_plain = GradientDescentSampler(step_size=0.01)(energy_model, x_seed, n_steps=200)
spread = lambda s: s.std(dim=0).mean().item()
print(f"diverse coda spread   with W: {spread(samples_repulsive):.3f}   without W: {spread(samples_plain):.3f}")
