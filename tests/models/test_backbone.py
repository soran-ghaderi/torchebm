import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

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


TWO_TIME = ["average_velocity", "flow_map"]


def _input(model, n=4):
    if isinstance(model, TimeConditionedMLP):
        return torch.randn(n, 2, dtype=torch.float64)
    return torch.randn(n, 1, 4, 4, dtype=torch.float64)


def _times(n=4):
    s = torch.rand(n, dtype=torch.float64)
    return s, s + (1 - s) * torch.rand(n, dtype=torch.float64)


def _perturbed(model):
    model = model.double()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return model


@pytest.mark.parametrize("backbone", [mlp, dit])
def test_one_time_backbone_has_no_dt_embedder(backbone):
    model = backbone()
    assert model.dt_embedder is None
    assert not any(k.startswith("dt_embedder") for k in model.state_dict())


@pytest.mark.parametrize("backbone", [mlp, dit])
def test_one_time_backbone_rejects_s(backbone):
    model = backbone()
    x = _input(model).float()
    with pytest.raises(ValueError, match="one-time model"):
        model(x, torch.rand(4), s=torch.rand(4))


@pytest.mark.parametrize("backbone", [mlp, dit])
@pytest.mark.parametrize("prediction_type", TWO_TIME)
def test_two_time_backbone_requires_s_and_t(backbone, prediction_type):
    model = backbone(prediction_type=prediction_type)
    x = _input(model).float()
    with pytest.raises(ValueError, match="both s and t"):
        model(x, torch.rand(4))
    with pytest.raises(ValueError, match="both s and t"):
        model(x, s=torch.rand(4))


@pytest.mark.parametrize("backbone", [mlp, dit])
@pytest.mark.parametrize("prediction_type", TWO_TIME)
def test_two_time_backbone_rejects_time_invariant(backbone, prediction_type):
    with pytest.raises(ValueError, match="time_invariant"):
        backbone(prediction_type=prediction_type, time_invariant=True)


def test_two_time_mlp_rejects_unconditional():
    with pytest.raises(ValueError, match="cond_dim"):
        mlp(conditioning="concat", cond_dim=0, prediction_type="flow_map")


def test_dt_embedder_requires_two_time_prediction_type():
    with pytest.raises(ValueError, match="dt_embedder"):
        mlp(dt_embedder=torch.nn.Linear(1, 8))


def test_flow_map_requires_matching_output_width():
    with pytest.raises(ValueError, match="out_dim"):
        mlp(out_dim=3, prediction_type="flow_map")
    with pytest.raises(ValueError, match="out_channels"):
        dit(out_channels=2, prediction_type="flow_map")


@pytest.mark.parametrize("backbone", [mlp, dit])
def test_two_time_backbone_initializes_dt_embedder_like_t_embedder(backbone):
    model = backbone(prediction_type="flow_map")
    for embedder in (model.t_embedder, model.dt_embedder):
        assert embedder.mlp[0].weight.std().item() < 0.03
        assert torch.equal(embedder.mlp[0].bias, torch.zeros_like(embedder.mlp[0].bias))
        assert torch.equal(embedder.mlp[2].bias, torch.zeros_like(embedder.mlp[2].bias))


@pytest.mark.parametrize("backbone", [mlp, dit])
def test_average_velocity_sums_the_time_and_interval_embeddings(backbone):
    torch.manual_seed(0)
    model = _perturbed(backbone(prediction_type="average_velocity", num_classes=3))
    velocity = backbone(num_classes=3).double()
    velocity.load_state_dict(
        {k: v for k, v in model.state_dict().items() if not k.startswith("dt_embedder")}
    )
    x, (s, t), y = _input(model), _times(), torch.tensor([0, 1, 2, 3])
    cond = model.dt_embedder(t - s) + model.y_embedder(y, training=False)
    assert torch.allclose(model(x, t, y, s=s), velocity(x, s, cond=cond), atol=1e-12)


@pytest.mark.parametrize("backbone", [mlp, dit])
def test_flow_map_is_x_plus_time_step_times_average_velocity(backbone):
    torch.manual_seed(0)
    flow_map = _perturbed(backbone(prediction_type="flow_map"))
    average_velocity = backbone(prediction_type="average_velocity").double()
    average_velocity.load_state_dict(flow_map.state_dict())
    x, (s, t) = _input(flow_map), _times()
    u = average_velocity(x, t, s=s)
    step = (t - s).reshape(-1, *(1,) * (x.ndim - 1))
    assert u.abs().min() > 0
    assert torch.equal(flow_map(x, t, s=s), x + step * u)
    assert torch.equal(flow_map(x, s, s=s), x)


@pytest.mark.parametrize("backbone", [mlp, dit])
def test_flow_map_jvp_matches_finite_differences(backbone):
    torch.manual_seed(0)
    flow_map = _perturbed(backbone(prediction_type="flow_map"))
    average_velocity = backbone(prediction_type="average_velocity").double()
    average_velocity.load_state_dict(flow_map.state_dict())
    x, (s, t) = _input(flow_map), _times()
    v, ds = torch.randn_like(x), torch.ones_like(s)

    def along(model, eps):
        return model(x + eps * v, t, s=s + eps * ds)

    with sdpa_kernel(SDPBackend.MATH):
        _, dF = torch.func.jvp(lambda x, s: flow_map(x, t, s=s), (x, s), (v, ds))
        u, du = torch.func.jvp(
            lambda x, s: average_velocity(x, t, s=s), (x, s), (v, ds)
        )
    eps = 1e-6
    finite_difference = (along(flow_map, eps) - along(flow_map, -eps)) / (2 * eps)
    step = (t - s).reshape(-1, *(1,) * (x.ndim - 1))
    assert torch.allclose(dF, finite_difference, atol=1e-8)
    assert torch.allclose(dF, v - u + step * du, atol=1e-12)


@pytest.mark.parametrize("backbone", [mlp, dit])
def test_two_time_backbone_compiles_without_graph_breaks(backbone):
    torch.manual_seed(0)
    model = _perturbed(backbone(prediction_type="flow_map"))
    x, (s, t) = _input(model), _times()
    compiled = torch.compile(model, fullgraph=True)
    assert torch.allclose(compiled(x, t, s=s), model(x, t, s=s), atol=1e-12)
