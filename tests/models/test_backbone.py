import pytest
import torch

from torchebm.models import DiT, TimeConditionedMLP


def mlp(**kwargs):
    return TimeConditionedMLP(in_dim=2, hidden_dim=8, depth=1, **kwargs)


def dit(**kwargs):
    return DiT(
        input_size=4,
        patch_size=2,
        in_channels=1,
        embed_dim=8,
        depth=1,
        num_heads=2,
        **kwargs,
    )


@pytest.mark.parametrize("backbone", [mlp, dit])
def test_backbone_prediction_type_defaults_to_velocity(backbone):
    assert backbone().prediction_type == "velocity"


@pytest.mark.parametrize("backbone", [mlp, dit])
@pytest.mark.parametrize("prediction_type", ["velocity", "score", "epsilon"])
def test_backbone_accepts_field_prediction_types(backbone, prediction_type):
    assert backbone(prediction_type=prediction_type).prediction_type == prediction_type


@pytest.mark.parametrize("backbone", [mlp, dit])
@pytest.mark.parametrize("prediction_type", ["energy", ("energy", "score"), "noise"])
def test_backbone_rejects_non_field_prediction_types(backbone, prediction_type):
    with pytest.raises(ValueError, match="prediction_type"):
        backbone(prediction_type=prediction_type)


def test_prediction_type_does_not_change_backbone_output():
    torch.manual_seed(0)
    velocity = mlp(conditioning="concat")
    torch.manual_seed(0)
    score = mlp(conditioning="concat", prediction_type="score")
    x, t = torch.randn(4, 2), torch.rand(4)
    assert torch.equal(velocity(x, t), score(x, t))
