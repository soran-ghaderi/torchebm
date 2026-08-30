"""Unknown model-constructor kwargs must fail with an actionable TypeError.

Mirrors the loss-side guard: without validation, a bogus keyword cascades
through BaseModel -> TorchEBMModule into nn.Module and dies with a bare
error naming neither the supported parameters nor the installed version.
"""

import pytest
import torch
import torch.nn as nn

from torchebm import __version__
from torchebm.core import (
    AckleyModel,
    DoubleWellModel,
    GaussianModel,
    HarmonicModel,
    RastriginModel,
    RosenbrockModel,
)
from torchebm.models import (
    ClassifierFreeGuidance,
    ConditionalTransformer2D,
    EqMEnergy,
    InteractionModel,
)


class _Potential(nn.Module):
    def forward(self, x, **kwargs):
        return 0.5 * x.square().sum(dim=-1)


CASCADING = [
    pytest.param(DoubleWellModel, lambda **kw: DoubleWellModel(**kw), id="DoubleWellModel"),
    pytest.param(
        GaussianModel,
        lambda **kw: GaussianModel(mean=torch.zeros(2), cov=torch.eye(2), **kw),
        id="GaussianModel",
    ),
    pytest.param(HarmonicModel, lambda **kw: HarmonicModel(**kw), id="HarmonicModel"),
    pytest.param(RosenbrockModel, lambda **kw: RosenbrockModel(**kw), id="RosenbrockModel"),
    pytest.param(AckleyModel, lambda **kw: AckleyModel(**kw), id="AckleyModel"),
    pytest.param(RastriginModel, lambda **kw: RastriginModel(**kw), id="RastriginModel"),
    pytest.param(
        InteractionModel,
        lambda **kw: InteractionModel(_Potential(), sigma_w=4.0, strength=0.1, **kw),
        id="InteractionModel",
    ),
]

FIXED_SIGNATURE = [
    pytest.param(lambda **kw: EqMEnergy(_Potential(), **kw), id="EqMEnergy"),
    pytest.param(
        lambda **kw: ClassifierFreeGuidance(
            _Potential(), guidance_scale=1.0, null_condition=3, **kw
        ),
        id="ClassifierFreeGuidance",
    ),
    pytest.param(
        lambda **kw: ConditionalTransformer2D(
            in_channels=1,
            out_channels=1,
            input_size=4,
            patch_size=2,
            embed_dim=8,
            depth=1,
            **kw,
        ),
        id="ConditionalTransformer2D",
    ),
]


@pytest.mark.parametrize("cls,factory", CASCADING)
def test_bogus_kwarg_raises_actionable_typeerror(cls, factory):
    with pytest.raises(TypeError) as excinfo:
        factory(bogus_kwarg=123)
    msg = str(excinfo.value)
    assert cls.__name__ in msg
    assert "bogus_kwarg" in msg
    assert "Supported parameters" in msg
    assert "device" in msg
    assert __version__ in msg


@pytest.mark.parametrize("factory", FIXED_SIGNATURE)
def test_fixed_signature_models_reject_bogus_kwarg(factory):
    with pytest.raises(TypeError):
        factory(bogus_kwarg=123)


def test_device_still_flows_through_cascade():
    model = DoubleWellModel(barrier_height=1.0, device="cpu", dtype=torch.float32)
    assert model.device.type == "cpu"
