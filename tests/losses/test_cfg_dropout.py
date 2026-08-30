"""Classifier-free-guidance label dropout on loss forwards.

cfg_dropout replaces y with the null condition per sample during training;
eval mode and the unconditional path are untouched.
"""

import pytest
import torch
import torch.nn as nn

from torchebm.core import BaseModel
from torchebm.losses import (
    ContrastiveDivergence,
    DenoisingScoreMatching,
    EnergyMatchingLoss,
    EquilibriumMatchingLoss,
    FlowMatchingLoss,
    ScoreMatching,
)
from torchebm.samplers import LangevinDynamics

NUM_CLASSES = 5


class RecordingPotential(BaseModel):
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
    def __init__(self, dim=4):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.received_y = None

    def forward(self, x, t=None, y=None, **kwargs):
        self.received_y = y
        out = self.linear(x)
        if y is not None:
            out = out + y.view(y.shape[0], -1).float().sum(dim=1, keepdim=True)
        return out


def _cd(model, **kw):
    return ContrastiveDivergence(
        model=model, sampler=LangevinDynamics(model=model), k_steps=2, **kw
    )


LOSSES = [
    pytest.param(RecordingField, EquilibriumMatchingLoss, id="EquilibriumMatchingLoss"),
    pytest.param(
        RecordingPotential,
        lambda model, **kw: EnergyMatchingLoss(model=model, lambda_cd=0.0, **kw),
        id="EnergyMatchingLoss",
    ),
    pytest.param(
        RecordingPotential,
        lambda model, **kw: ScoreMatching(model=model, hessian_method="approx", **kw),
        id="ScoreMatching-approx",
    ),
    pytest.param(RecordingPotential, DenoisingScoreMatching, id="DenoisingScoreMatching"),
    pytest.param(RecordingPotential, _cd, id="ContrastiveDivergence"),
    pytest.param(
        RecordingField,
        lambda model, **kw: FlowMatchingLoss(model=model, **kw),
        id="FlowMatchingLoss",
    ),
]


@pytest.mark.parametrize("model_cls,factory", LOSSES)
def test_cfg_dropout_one_replaces_every_label(model_cls, factory):
    model = model_cls()
    loss_fn = factory(model, cfg_dropout=1.0, null_condition=NUM_CLASSES)
    loss_fn.train()
    y = torch.randint(0, NUM_CLASSES, (8,))
    loss_fn(torch.randn(8, 4), y=y)
    assert torch.all(model.received_y == NUM_CLASSES)


@pytest.mark.parametrize("model_cls,factory", LOSSES)
def test_cfg_dropout_zero_keeps_labels(model_cls, factory):
    model = model_cls()
    loss_fn = factory(model)
    loss_fn.train()
    y = torch.randint(0, NUM_CLASSES, (8,))
    loss_fn(torch.randn(8, 4), y=y)
    assert torch.equal(model.received_y, y.to(model.received_y.device))


def test_cfg_dropout_inactive_in_eval_mode():
    model = RecordingField()
    loss_fn = EquilibriumMatchingLoss(
        model=model, cfg_dropout=1.0, null_condition=NUM_CLASSES
    )
    loss_fn.eval()
    y = torch.randint(0, NUM_CLASSES, (8,))
    loss_fn(torch.randn(8, 4), y=y)
    assert torch.equal(model.received_y, y)


def test_cfg_dropout_mask_is_generator_deterministic():
    model = RecordingField()
    loss_fn = EquilibriumMatchingLoss(
        model=model, cfg_dropout=0.5, null_condition=NUM_CLASSES
    )
    loss_fn.train()
    x = torch.randn(64, 4)
    y = torch.randint(0, NUM_CLASSES, (64,))
    loss_fn(x, y=y, generator=torch.Generator().manual_seed(7))
    first = model.received_y.clone()
    loss_fn(x, y=y, generator=torch.Generator().manual_seed(7))
    assert torch.equal(first, model.received_y)
    dropped = (first == NUM_CLASSES) & (y != NUM_CLASSES)
    assert dropped.any() and not dropped.all()
    assert torch.equal(first[~dropped], y[~dropped])


def test_cfg_dropout_tensor_null_for_embeddings():
    model = RecordingField()
    loss_fn = EquilibriumMatchingLoss(
        model=model, cfg_dropout=1.0, null_condition=torch.zeros(3)
    )
    loss_fn.train()
    y = torch.randn(8, 3)
    loss_fn(torch.randn(8, 4), y=y)
    assert torch.all(model.received_y == 0)
    assert model.received_y.shape == y.shape


def test_cfg_dropout_callable_null():
    def null_fn(y, mask):
        return torch.where(mask, torch.full_like(y, -1), y)

    model = RecordingField()
    loss_fn = EquilibriumMatchingLoss(
        model=model, cfg_dropout=1.0, null_condition=null_fn
    )
    loss_fn.train()
    y = torch.randint(0, NUM_CLASSES, (8,))
    loss_fn(torch.randn(8, 4), y=y)
    assert torch.all(model.received_y == -1)


def test_cfg_dropout_noop_without_y():
    loss_fn = EquilibriumMatchingLoss(
        model=RecordingField(), cfg_dropout=0.5, null_condition=NUM_CLASSES
    )
    loss_fn.train()
    assert torch.isfinite(loss_fn(torch.randn(8, 4)))


def test_cfg_dropout_out_of_range_raises():
    with pytest.raises(ValueError, match="cfg_dropout"):
        EquilibriumMatchingLoss(
            model=RecordingField(), cfg_dropout=1.5, null_condition=NUM_CLASSES
        )


def test_cfg_dropout_requires_null_condition():
    with pytest.raises(ValueError, match="null_condition"):
        EquilibriumMatchingLoss(model=RecordingField(), cfg_dropout=0.1)
