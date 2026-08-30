"""Contract probe: a model handed y must actually consume it.

On the first conditional loss call, the same input is evaluated under two
distinct in-batch y values; identical outputs mean the backbone silently
drops its conditioning and training would produce an unconditional model.
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


class IgnoringPotential(BaseModel):
    def __init__(self, dim=4):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x, t=None, y=None, **kwargs):
        return self.linear(x).squeeze(-1)


class IgnoringField(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x, t=None, y=None, **kwargs):
        return self.linear(x)


class ConsumingField(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.calls = 0

    def forward(self, x, t=None, y=None, **kwargs):
        self.calls += 1
        out = self.linear(x)
        if y is not None:
            out = out + y.view(y.shape[0], -1).float().sum(dim=1, keepdim=True)
        return out


def _cd(model, **kw):
    return ContrastiveDivergence(
        model=model, sampler=LangevinDynamics(model=model), k_steps=2, **kw
    )


LOSSES = [
    pytest.param(IgnoringField, EquilibriumMatchingLoss, id="EquilibriumMatchingLoss"),
    pytest.param(
        IgnoringPotential,
        lambda model, **kw: EnergyMatchingLoss(model=model, lambda_cd=0.0, **kw),
        id="EnergyMatchingLoss",
    ),
    pytest.param(
        IgnoringPotential,
        lambda model, **kw: ScoreMatching(model=model, hessian_method="approx", **kw),
        id="ScoreMatching-approx",
    ),
    pytest.param(IgnoringPotential, DenoisingScoreMatching, id="DenoisingScoreMatching"),
    pytest.param(IgnoringPotential, _cd, id="ContrastiveDivergence"),
    pytest.param(
        IgnoringField,
        lambda model, **kw: FlowMatchingLoss(model=model, **kw),
        id="FlowMatchingLoss",
    ),
]


def _distinct_y(batch=8):
    y = torch.zeros(batch, dtype=torch.long)
    y[batch // 2 :] = 1
    return y


@pytest.mark.parametrize("model_cls,factory", LOSSES)
def test_ignoring_model_raises(model_cls, factory):
    model = model_cls()
    loss_fn = factory(model)
    with pytest.raises(ValueError, match=type(model).__name__):
        loss_fn(torch.randn(8, 4), y=_distinct_y())


def test_consuming_model_passes():
    loss_fn = EquilibriumMatchingLoss(model=ConsumingField())
    assert torch.isfinite(loss_fn(torch.randn(8, 4), y=_distinct_y()))


def test_opt_out_flag_skips_probe():
    loss_fn = EquilibriumMatchingLoss(
        model=IgnoringField(), check_conditioning=False
    )
    assert torch.isfinite(loss_fn(torch.randn(8, 4), y=_distinct_y()))


def test_probe_runs_once():
    model = ConsumingField()
    loss_fn = EquilibriumMatchingLoss(model=model)
    loss_fn(torch.randn(8, 4), y=_distinct_y())
    assert model.calls == 3  # probe pair + training forward
    loss_fn(torch.randn(8, 4), y=_distinct_y())
    assert model.calls == 4


def test_unconditional_path_never_probes():
    model = ConsumingField()
    loss_fn = EquilibriumMatchingLoss(model=model)
    loss_fn(torch.randn(8, 4))
    assert model.calls == 1


def test_uniform_labels_warn_and_disarm():
    model = IgnoringField()
    loss_fn = EquilibriumMatchingLoss(model=model)
    y_uniform = torch.zeros(8, dtype=torch.long)
    with pytest.warns(UserWarning, match="[Cc]ould not"):
        loss_fn(torch.randn(8, 4), y=y_uniform)
    # Disarmed: a later distinct batch must not raise.
    assert torch.isfinite(loss_fn(torch.randn(8, 4), y=_distinct_y()))


def test_probe_restores_training_mode():
    model = ConsumingField()
    model.train()
    loss_fn = EquilibriumMatchingLoss(model=model)
    loss_fn(torch.randn(8, 4), y=_distinct_y())
    assert model.training
