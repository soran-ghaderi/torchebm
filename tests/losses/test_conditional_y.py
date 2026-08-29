"""Explicit y= conditioning passthrough on loss forwards.

y is forwarded to the model as model(x, t, y=y) (an alias into the
model_kwargs channel), y=None keeps the unconditional path identical, and
passing y both ways at once fails loudly.
"""

import unittest.mock

import pytest
import torch
import torch.nn as nn

from torchebm.core import BaseModel
from torchebm.losses import (
    ContrastiveDivergence,
    DenoisingScoreMatching,
    EnergyMatchingLoss,
    EquilibriumMatchingLoss,
    ScoreMatching,
    SlicedScoreMatching,
)
from torchebm.samplers import LangevinDynamics


class RecordingPotential(BaseModel):
    """Scalar energy that records the y it receives."""

    def __init__(self, dim=4):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        self.received_y = None

    def forward(self, x, t=None, y=None, **kwargs):
        self.received_y = y
        out = self.linear(x).squeeze(-1)
        if y is not None:
            out = out + y.view(y.shape[0], -1).float().sum(dim=1)
        return out


class RecordingField(nn.Module):
    """Vector field that records the y it receives."""

    def __init__(self, dim=4):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.received_y = None

    def forward(self, x, t=None, y=None, **kwargs):
        self.received_y = y
        out = self.linear(x)
        if y is not None:
            out = out + y.view(y.shape[0], *([1] * (x.ndim - 1))).float()
        return out


def _cd(model):
    return ContrastiveDivergence(
        model=model, sampler=LangevinDynamics(model=model), k_steps=2
    )


LOSSES = [
    pytest.param(RecordingField, EquilibriumMatchingLoss, id="EquilibriumMatchingLoss"),
    pytest.param(
        RecordingPotential,
        lambda model: EnergyMatchingLoss(model=model, lambda_cd=0.0),
        id="EnergyMatchingLoss",
    ),
    pytest.param(
        RecordingPotential,
        lambda model: ScoreMatching(model=model, hessian_method="approx"),
        id="ScoreMatching-approx",
    ),
    pytest.param(RecordingPotential, DenoisingScoreMatching, id="DenoisingScoreMatching"),
    pytest.param(RecordingPotential, _cd, id="ContrastiveDivergence"),
]

UNSUPPORTED = [
    pytest.param(ScoreMatching, id="ScoreMatching-exact"),
    pytest.param(SlicedScoreMatching, id="SlicedScoreMatching"),
]


@pytest.mark.parametrize("loss_cls", UNSUPPORTED)
def test_y_surfaces_unsupported_conditioning_error(loss_cls):
    """Losses that reject conditioning reject y= the same clear way."""
    loss_fn = loss_cls(model=RecordingPotential())
    with pytest.raises(NotImplementedError, match="[Cc]onditional"):
        loss_fn(torch.randn(8, 4), y=torch.randint(0, 3, (8,)))


def _loss_of(result):
    return result[0] if isinstance(result, tuple) else result


@pytest.mark.parametrize("model_cls,factory", LOSSES)
def test_y_reaches_the_model(model_cls, factory):
    model = model_cls()
    loss_fn = factory(model)
    y = torch.randint(0, 3, (8,))
    loss = _loss_of(loss_fn(torch.randn(8, 4), y=y))
    assert torch.isfinite(loss)
    assert model.received_y is not None
    assert torch.equal(model.received_y, y.to(model.received_y.device))


@pytest.mark.parametrize("model_cls,factory", LOSSES)
def test_y_equivalent_to_model_kwargs(model_cls, factory):
    model = model_cls()
    loss_fn = factory(model)
    x = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    loss_sugar = _loss_of(
        loss_fn(x, y=y, generator=torch.Generator().manual_seed(3))
    )
    loss_explicit = _loss_of(
        loss_fn(
            x, model_kwargs={"y": y}, generator=torch.Generator().manual_seed(3)
        )
    )
    assert torch.equal(loss_sugar, loss_explicit)


@pytest.mark.parametrize("model_cls,factory", LOSSES)
def test_y_none_is_identical_to_omitting(model_cls, factory):
    model = model_cls()
    loss_fn = factory(model)
    x = torch.randn(8, 4)
    loss_none = _loss_of(loss_fn(x, y=None, generator=torch.Generator().manual_seed(5)))
    loss_omitted = _loss_of(loss_fn(x, generator=torch.Generator().manual_seed(5)))
    assert torch.equal(loss_none, loss_omitted)
    assert model.received_y is None


def test_y_does_not_trigger_deprecated_kwargs_path():
    loss_fn = EquilibriumMatchingLoss(model=RecordingField())
    with unittest.mock.patch("torchebm.core.base_loss.warn_once") as warn:
        loss_fn(torch.randn(4, 4), y=torch.randint(0, 3, (4,)))
    warn.assert_not_called()


def test_y_conflict_with_model_kwargs_raises():
    loss_fn = EquilibriumMatchingLoss(model=RecordingField())
    y = torch.randint(0, 3, (4,))
    with pytest.raises(ValueError, match="model_kwargs"):
        loss_fn(torch.randn(4, 4), y=y, model_kwargs={"y": y})


def test_conditional_model_trains():
    model = RecordingField()
    loss_fn = EquilibriumMatchingLoss(model=model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    y = torch.randint(0, 3, (16,))
    for _ in range(3):
        optimizer.zero_grad()
        loss = loss_fn(torch.randn(16, 4), y=y)
        loss.backward()
        optimizer.step()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )
