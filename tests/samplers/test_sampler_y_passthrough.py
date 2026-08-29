"""Explicit y= conditioning on sampler sample(), mirroring the loss surface."""

import warnings

import pytest
import torch
import torch.nn as nn

from torchebm.core import BaseModel
from torchebm.samplers import (
    FlowSampler,
    GradientDescentSampler,
    HamiltonianMonteCarlo,
    LangevinDynamics,
    NesterovSampler,
)


class CondField(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.received_y = None

    def forward(self, x, t=None, y=None, **kwargs):
        self.received_y = y
        out = self.linear(x)
        if y is not None:
            out = out + y.view(-1, 1).float()
        return out


class CondEnergy(BaseModel):
    def __init__(self, dim=2):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        self.received_y = None

    def forward(self, x, y=None, **kwargs):
        self.received_y = y
        out = 0.5 * x.square().sum(dim=1) + self.linear(x).squeeze(-1)
        if y is not None:
            out = out + y.float()
        return out


SAMPLERS = [
    pytest.param(
        CondField,
        lambda m: FlowSampler(m, integrator="euler"),
        id="FlowSampler",
    ),
    pytest.param(
        CondEnergy,
        lambda m: LangevinDynamics(m, step_size=0.01),
        id="LangevinDynamics",
    ),
    pytest.param(
        CondEnergy,
        lambda m: HamiltonianMonteCarlo(m, step_size=0.01, n_leapfrog_steps=3),
        id="HamiltonianMonteCarlo",
    ),
    pytest.param(
        CondEnergy,
        lambda m: GradientDescentSampler(m, step_size=0.01),
        id="GradientDescentSampler",
    ),
    pytest.param(
        CondEnergy,
        lambda m: NesterovSampler(m, step_size=0.01),
        id="NesterovSampler",
    ),
]


@pytest.mark.parametrize("model_cls,factory", SAMPLERS)
def test_sample_y_equivalent_to_model_kwargs(model_cls, factory):
    y = torch.randint(0, 3, (4,))
    x0 = torch.randn(4, 2)

    model_a = model_cls()
    out_sugar = factory(model_a).sample(
        x=x0.clone(), n_steps=3, y=y, generator=torch.Generator().manual_seed(9)
    )
    state = model_a.state_dict()

    model_b = model_cls()
    model_b.load_state_dict(state)
    out_explicit = factory(model_b).sample(
        x=x0.clone(),
        n_steps=3,
        model_kwargs={"y": y},
        generator=torch.Generator().manual_seed(9),
    )
    assert torch.equal(out_sugar, out_explicit)
    assert model_a.received_y is not None


@pytest.mark.parametrize("model_cls,factory", SAMPLERS)
def test_sample_y_emits_no_deprecation_warning(model_cls, factory):
    sampler = factory(model_cls())
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        sampler.sample(x=torch.randn(4, 2), n_steps=2, y=torch.randint(0, 3, (4,)))


def test_sample_y_conflict_raises():
    sampler = LangevinDynamics(CondEnergy(), step_size=0.01)
    y = torch.randint(0, 3, (4,))
    with pytest.raises(ValueError, match="model_kwargs"):
        sampler.sample(x=torch.randn(4, 2), n_steps=2, y=y, model_kwargs={"y": y})
