"""Unknown constructor kwargs must fail with an actionable TypeError.

Without validation, a bogus keyword falls through BaseLoss -> Schedulable ->
TorchEBMModule into ``nn.Module.__init__`` and dies with a bare "unexpected
keyword argument" that names neither the loss's supported parameters nor the
installed version.
"""

import pytest
import torch
import torch.nn as nn

from torchebm import __version__
from torchebm.core import BaseModel
from torchebm.losses import (
    ContrastiveDivergence,
    DenoisingScoreMatching,
    EnergyMatchingLoss,
    EquilibriumMatchingLoss,
    FlowMatchingLoss,
    ScoreMatching,
    SlicedScoreMatching,
)
from torchebm.samplers import LangevinDynamics


class QuadraticModel(BaseModel):
    def forward(self, x):
        return 0.5 * x.flatten(1).square().sum(dim=1)


class VelocityModel(nn.Module):
    def forward(self, x, t=None, **kwargs):
        return x


def _make_cd(**kwargs):
    model = QuadraticModel()
    return ContrastiveDivergence(
        model=model, sampler=LangevinDynamics(model=model), **kwargs
    )


LOSS_FACTORIES = [
    pytest.param(ContrastiveDivergence, _make_cd, id="ContrastiveDivergence"),
    pytest.param(
        ScoreMatching,
        lambda **kw: ScoreMatching(model=QuadraticModel(), **kw),
        id="ScoreMatching",
    ),
    pytest.param(
        DenoisingScoreMatching,
        lambda **kw: DenoisingScoreMatching(model=QuadraticModel(), **kw),
        id="DenoisingScoreMatching",
    ),
    pytest.param(
        SlicedScoreMatching,
        lambda **kw: SlicedScoreMatching(model=QuadraticModel(), **kw),
        id="SlicedScoreMatching",
    ),
    pytest.param(
        EquilibriumMatchingLoss,
        lambda **kw: EquilibriumMatchingLoss(model=VelocityModel(), **kw),
        id="EquilibriumMatchingLoss",
    ),
    pytest.param(
        EnergyMatchingLoss,
        lambda **kw: EnergyMatchingLoss(model=QuadraticModel(), **kw),
        id="EnergyMatchingLoss",
    ),
    pytest.param(
        FlowMatchingLoss,
        lambda **kw: FlowMatchingLoss(model=VelocityModel(), **kw),
        id="FlowMatchingLoss",
    ),
]


@pytest.mark.parametrize("cls,factory", LOSS_FACTORIES)
def test_bogus_kwarg_raises_actionable_typeerror(cls, factory):
    with pytest.raises(TypeError) as excinfo:
        factory(bogus_kwarg=123)
    msg = str(excinfo.value)
    assert cls.__name__ in msg
    assert "bogus_kwarg" in msg
    assert "Supported parameters" in msg
    assert "model" in msg
    assert "device" in msg
    assert __version__ in msg


def test_intermediate_base_params_still_cascade():
    loss = ScoreMatching(model=QuadraticModel(), hutchinson_samples=4)
    assert loss.hutchinson_samples == 4


def test_supported_parameters_cover_full_chain():
    with pytest.raises(TypeError) as excinfo:
        ScoreMatching(model=QuadraticModel(), bogus_kwarg=123)
    assert "hutchinson_samples" in str(excinfo.value)
