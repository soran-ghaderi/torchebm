r"""EqMEnergy adapter: scalar-energy sampling of Equilibrium Matching fields (#238).

Pins the adapter's forward/gradient formulas to the loss, the `from_loss`
mapping, conditioning passthrough, and that every gradient-based sampler drives
an EqM field with no hand-written wrapper. A slower end-to-end test checks the
sign invariant: the implicit ODE and implicit gradient-descent routes both land
on the data manifold (a wrong sign would send chains away from the data).
"""

import math

import pytest
import torch
import torch.nn as nn

from torchebm.datasets import TwoMoonsDataset
from torchebm.losses import EquilibriumMatchingLoss
from torchebm.models import EqMEnergy, InteractionModel
from torchebm.samplers import (
    FlowSampler,
    GradientDescentSampler,
    HamiltonianMonteCarlo,
    LangevinDynamics,
    NesterovSampler,
)


class LinearField(nn.Module):
    r"""Deterministic field f(x, t) = W x + b, time-invariant."""

    def __init__(self, dim=2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(dim, dim, generator=g))
        self.b = nn.Parameter(torch.randn(dim, generator=g))

    def forward(self, x, t=None, **kwargs):
        return x @ self.W.T + self.b


class RecordingField(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.seen = []

    def forward(self, x, t=None, y=None):
        self.seen.append(y)
        return self.lin(x)


def _f(field, x):
    return field(x, torch.zeros(x.shape[0]))


# --------------------------------------------------------------------------- #
# Forward / gradient formulas pinned to the loss
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["dot", "mean", "implicit"])
def test_forward_dot_family(mode):
    field = LinearField()
    x = torch.randn(6, 2)
    e = EqMEnergy(field, mode)
    assert torch.allclose(e(x), (x * _f(field, x)).sum(dim=-1))


def test_forward_l2():
    field = LinearField()
    x = torch.randn(6, 2)
    e = EqMEnergy(field, "l2")
    assert torch.allclose(e(x), -0.5 * _f(field, x).square().sum(dim=-1))


def test_dot_gradient_is_autograd_of_energy():
    field = LinearField()
    x = torch.randn(6, 2)
    e = EqMEnergy(field, "dot")
    xg = x.clone().requires_grad_(True)
    g_auto = torch.autograd.grad((xg * field(xg, torch.zeros(6))).sum(), xg)[0]
    assert torch.allclose(e.gradient(x), g_auto, atol=1e-5)


def test_implicit_gradient_is_field():
    field = LinearField()
    x = torch.randn(6, 2)
    e = EqMEnergy(field, "implicit")
    assert torch.allclose(e.gradient(x), _f(field, x), atol=1e-6)


def test_invalid_energy_type_raises():
    with pytest.raises(ValueError):
        EqMEnergy(LinearField(), "bogus")


# --------------------------------------------------------------------------- #
# from_loss mapping and conditioning passthrough
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "loss_type,expected", [("none", "implicit"), ("dot", "dot"), ("l2", "l2")]
)
def test_from_loss_maps_energy_type(loss_type, expected):
    field = LinearField()
    loss = EquilibriumMatchingLoss(model=field, energy_type=loss_type)
    assert EqMEnergy.from_loss(loss).energy_type == expected


def test_model_kwargs_reach_field():
    field = RecordingField()
    e = EqMEnergy(field, "dot")
    y = torch.zeros(4, dtype=torch.long)
    e.gradient(torch.randn(4, 2), model_kwargs={"y": y})
    assert field.seen and field.seen[-1] is not None


# --------------------------------------------------------------------------- #
# Every gradient-based sampler drives an EqM field
# --------------------------------------------------------------------------- #
GRAD_SAMPLERS = [
    lambda m: LangevinDynamics(m, step_size=0.01),
    lambda m: GradientDescentSampler(m, step_size=0.01),
    lambda m: NesterovSampler(m, step_size=0.01),
    lambda m: HamiltonianMonteCarlo(m, step_size=0.02, n_leapfrog_steps=2),
]


@pytest.mark.parametrize("factory", GRAD_SAMPLERS, ids=["langevin", "gd", "nesterov", "hmc"])
@pytest.mark.parametrize("mode", ["dot", "implicit"])
def test_samplers_drive_eqm_energy(factory, mode):
    energy = EqMEnergy(LinearField(), mode)
    out = factory(energy).sample(x=torch.randn(8, 2), n_steps=4)
    assert out.shape == (8, 2) and torch.isfinite(out).all()


def test_interaction_model_wraps_explicit_energy():
    energy = EqMEnergy(LinearField(), "dot")
    repulsive = InteractionModel(energy, sigma_w=4.0, strength=0.1)
    out = GradientDescentSampler(repulsive, step_size=0.01).sample(
        x=torch.randn(16, 2), n_steps=4
    )
    assert out.shape == (16, 2) and torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Sign invariant: both routes transport noise -> data
# --------------------------------------------------------------------------- #
def _median_dist_to_data(samples, data):
    return torch.cdist(samples, data).min(dim=1).values.median().item()


def test_implicit_routes_agree_on_manifold():
    torch.manual_seed(0)
    data = TwoMoonsDataset(n_samples=2000, noise=0.05, seed=0).get_data()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
                nn.Linear(128, 2),
            )

        def forward(self, x, t=None, **kw):
            return self.net(x)

    field = Net()
    loss_fn = EquilibriumMatchingLoss(model=field, energy_type="none")
    opt = torch.optim.Adam(field.parameters(), lr=2e-3)
    for _ in range(1500):
        batch = data[torch.randint(len(data), (256,))]
        loss = loss_fn(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()

    x0 = torch.randn(1000, 2)
    ode = FlowSampler(
        field, interpolant="linear", negate_velocity=True, integrator="euler"
    ).sample(x=x0.clone(), n_steps=100)
    gd = GradientDescentSampler(
        EqMEnergy(field, "implicit"), step_size=0.05
    ).sample(x=x0.clone(), n_steps=200)

    # Both routes must land near the data manifold (a flipped sign would blow up).
    assert _median_dist_to_data(ode, data) < 0.25
    assert _median_dist_to_data(gd, data) < 0.25
