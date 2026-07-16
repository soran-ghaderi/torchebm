"""Equilibrium Matching in 2D: one field, several coherent sampling routes.

EqM trains a time-invariant field f(x) whose direction points data -> noise, so
sampling transports noise -> data by moving along -f. This example trains two
small fields on two-moons and shows every coherent way to sample them:

  - implicit (energy_type="none"): f IS the gradient field. Sample by
    integrating -f with FlowSampler(negate_velocity=True), OR by descending it
    with EqMEnergy(field, "implicit") + a gradient sampler. Both agree.
  - explicit (energy_type="dot"): the scalar energy g(x) = x . f(x) is trained.
    Sample by descending g with EqMEnergy(field, "dot"); wrap it in
    InteractionModel for diverse (repulsive) sampling.

EqMEnergy is the adapter that turns the field into the scalar BaseModel the
gradient-based samplers and InteractionModel consume - no hand-written wrapper.
"""

import os

import torch
from torch import nn

from torchebm.datasets import TwoMoonsDataset
from torchebm.losses import EquilibriumMatchingLoss
from torchebm.models import EqMEnergy, InteractionModel
from torchebm.samplers import FlowSampler, GradientDescentSampler

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 2000
N_GEN = 128 if SMOKE else 2000


class Field(nn.Module):
    """f(x, t) -> R^2; EqM is time-invariant, so t is unused."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x, t, **kwargs):
        return self.net(x)


def train(energy_type):
    """Train a time-invariant field toward the two-moons distribution."""
    torch.manual_seed(0)
    data = TwoMoonsDataset(n_samples=4000, noise=0.05, seed=0).get_data()
    field = Field()
    loss_fn = EquilibriumMatchingLoss(model=field, energy_type=energy_type)
    opt = torch.optim.Adam(field.parameters(), lr=1e-3)
    for _ in range(N_STEPS):
        batch = data[torch.randint(len(data), (256,))]
        loss = loss_fn(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return field, data


def median_dist(samples, data):
    """Median distance from each sample to the nearest data point."""
    return torch.cdist(samples, data).min(dim=1).values.median().item()


# ---- implicit field: f is the gradient field; two equivalent routes ----------
field_impl, data = train("none")
x0 = torch.randn(N_GEN, 2)

ode = FlowSampler(
    field_impl, interpolant="linear", negate_velocity=True, integrator="euler"
).sample(x=x0.clone(), n_steps=100)

gd_impl = GradientDescentSampler(
    EqMEnergy(field_impl, "implicit"), step_size=0.05
).sample(x=x0.clone(), n_steps=200)

print(f"implicit + ODE   median dist to data: {median_dist(ode, data):.3f}")
print(f"implicit + GD    median dist to data: {median_dist(gd_impl, data):.3f}")

# ---- explicit field: descend the scalar energy g(x) = x . f(x) ---------------
field_dot, _ = train("dot")
energy = EqMEnergy(field_dot, "dot")
gd_dot = GradientDescentSampler(energy, step_size=0.02).sample(
    x=x0.clone(), n_steps=300
)
print(f"explicit + GD    median dist to data: {median_dist(gd_dot, data):.3f}")

# ---- diverse sampling: pairwise repulsion fans chains out --------------------
# 64 chains from one seed; the repulsive W spreads them along the low-energy
# manifold, while plain gradient descent collapses them to one point.
seed = torch.tensor([0.5, 1.6]) + 1e-3 * torch.randn(64, 2)
# strength keeps the per-step repulsion 2*s*B*dt/sigma_w^2 well below 1 (stable)
# yet strong enough to visibly overcome the energy's pull to a single point.
repulsive = InteractionModel(energy, sigma_w=4.5, strength=0.5)
with_w = GradientDescentSampler(repulsive, step_size=0.01).sample(
    x=seed.clone(), n_steps=200
)
without_w = GradientDescentSampler(energy, step_size=0.01).sample(
    x=seed.clone(), n_steps=200
)
spread = lambda s: s.std(dim=0).mean().item()
print(f"diverse coda spread   with W: {spread(with_w):.3f}   without W: {spread(without_w):.3f}")
