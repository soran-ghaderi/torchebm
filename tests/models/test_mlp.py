import pytest
import torch
import torch.nn as nn

from torchebm.models import LabelEmbedder, TimeConditionedMLP


def _tiny(**overrides):
    cfg = dict(in_dim=2, hidden_dim=16, depth=2)
    cfg.update(overrides)
    return TimeConditionedMLP(**cfg)


def test_adaln_shape_with_timestep():
    model = _tiny()
    y = model(torch.randn(4, 2), torch.rand(4))
    assert y.shape == (4, 2)


def test_out_dim_override():
    model = _tiny(out_dim=5)
    y = model(torch.randn(4, 2), torch.rand(4))
    assert y.shape == (4, 5)


def test_concat_shape_with_timestep():
    model = _tiny(conditioning="concat")
    y = model(torch.randn(4, 2), torch.rand(4))
    assert y.shape == (4, 2)
    assert model.net[0].in_features == 2 + model.cond_dim


def test_adaln_zero_init_output():
    model = _tiny()
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(4, 2), torch.rand(4))
    assert torch.allclose(y, torch.zeros_like(y))


def test_adaln_reference_init_scheme():
    model = _tiny(num_classes=10)
    assert torch.all(model.stem.bias == 0)
    assert (model.stem.weight != 0).any()
    for block in model.blocks:
        assert torch.all(block.modulation[-1].weight == 0)
        assert torch.all(block.modulation[-1].bias == 0)
        for layer in block.mlp.net:
            if isinstance(layer, nn.Linear):
                assert torch.all(layer.bias == 0)
                assert (layer.weight != 0).any()
    assert torch.all(model.head.proj.weight == 0)
    assert torch.all(model.t_embedder.mlp[0].bias == 0)
    assert 0.01 < model.t_embedder.mlp[0].weight.std().item() < 0.03
    assert 0.01 < model.y_embedder.embedding.weight.std().item() < 0.03


def test_label_table_always_has_null_row():
    model = _tiny(num_classes=10)
    assert isinstance(model.y_embedder, LabelEmbedder)
    assert model.y_embedder.embedding.num_embeddings == 11
    null_y = torch.full((4,), 10, dtype=torch.long)
    y = model(torch.randn(4, 2), torch.rand(4), null_y)
    assert y.shape == (4, 2)


def test_summed_conditioning():
    model = _tiny(num_classes=4)
    y = model(
        torch.randn(4, 2),
        torch.rand(4),
        torch.zeros(4, dtype=torch.long),
        cond=torch.randn(4, 16),
    )
    assert y.shape == (4, 2)


def test_no_conditioning_raises():
    model = _tiny()
    with pytest.raises(ValueError, match="at least one"):
        model(torch.randn(4, 2))


def test_y_without_label_embedder_raises():
    model = _tiny()
    with pytest.raises(ValueError, match="label embedder"):
        model(torch.randn(4, 2), torch.rand(4), torch.zeros(4, dtype=torch.long))


def test_cond_width_mismatch_raises():
    model = _tiny()
    with pytest.raises(ValueError, match="cond_dim"):
        model(torch.randn(4, 2), cond=torch.randn(4, 8))


def test_unconditional_ignores_t():
    model = _tiny(conditioning="concat", cond_dim=0)
    assert model.t_embedder is None
    x = torch.randn(4, 2)
    with torch.no_grad():
        a = model(x)
        b = model(x, torch.rand(4))
    assert torch.equal(a, b)


def test_unconditional_rejects_y_and_cond():
    model = _tiny(conditioning="concat", cond_dim=0)
    with pytest.raises(ValueError, match="unconditional"):
        model(torch.randn(4, 2), y=torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError, match="unconditional"):
        model(torch.randn(4, 2), cond=torch.randn(4, 16))


def test_unconditional_rejects_embedder_config():
    with pytest.raises(ValueError, match="unconditional"):
        _tiny(conditioning="concat", cond_dim=0, num_classes=4)


def test_adaln_requires_cond_dim():
    with pytest.raises(ValueError, match="cond_dim > 0"):
        _tiny(cond_dim=0)


def test_mlp_ratio_rejected_in_concat():
    with pytest.raises(ValueError, match="mlp_ratio"):
        _tiny(conditioning="concat", mlp_ratio=2.0)


def test_invalid_conditioning_raises():
    with pytest.raises(ValueError, match="conditioning"):
        _tiny(conditioning="film")


def test_conflicting_y_embedder_and_num_classes():
    with pytest.raises(ValueError, match="not both"):
        _tiny(y_embedder=nn.Embedding(4, 16), num_classes=4)


def test_custom_activation_used():
    model = _tiny(conditioning="concat", activation=nn.Tanh)
    assert isinstance(model.net[1], nn.Tanh)
    adaln = _tiny(activation=nn.Tanh)
    assert isinstance(adaln.blocks[0].mlp.net[1], nn.Tanh)


def test_custom_embedder_init_untouched():
    t_emb = nn.Linear(1, 16)
    with torch.no_grad():
        t_emb.weight.fill_(7.0)
    _tiny(t_embedder=t_emb)
    assert torch.all(t_emb.weight == 7.0)


@pytest.mark.parametrize("conditioning", ["adaln_zero", "concat"])
def test_gradient_flows(conditioning):
    model = _tiny(conditioning=conditioning)
    target = torch.randn(4, 2)
    loss = (model(torch.randn(4, 2), torch.rand(4)) - target).pow(2).sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and (g.abs() > 0).any() for g in grads)


def test_flow_matching_smoke():
    from torchebm.losses import FlowMatchingLoss

    torch.manual_seed(0)
    model = _tiny()
    loss_fn = FlowMatchingLoss(model=model)
    loss = loss_fn(torch.randn(16, 2))
    assert torch.isfinite(loss)
    loss.backward()
