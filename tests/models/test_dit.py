import pytest
import torch
import torch.nn as nn

import torchebm.models as models
from torchebm.models import DiT, LabelEmbedder, dit_s_4
from torchebm.models.dit import _DIT_CONFIGS


def _tiny(**overrides):
    cfg = dict(
        input_size=8,
        patch_size=4,
        in_channels=3,
        embed_dim=16,
        depth=2,
        num_heads=2,
    )
    cfg.update(overrides)
    return DiT(**cfg)


def test_shape_with_timestep():
    model = _tiny()
    y = model(torch.randn(2, 3, 8, 8), torch.rand(2))
    assert y.shape == (2, 3, 8, 8)


def test_out_channels_learned_sigma():
    model = _tiny(out_channels=6)
    y = model(torch.randn(2, 3, 8, 8), torch.rand(2))
    assert y.shape == (2, 6, 8, 8)


def test_rectangular_input():
    model = _tiny(input_size=(8, 12))
    y = model(torch.randn(2, 3, 8, 12), torch.rand(2))
    assert y.shape == (2, 3, 8, 12)


def test_spatial_size_mismatch_raises():
    model = _tiny()
    with pytest.raises(ValueError, match="spatial size"):
        model(torch.randn(2, 3, 4, 4), torch.rand(2))


def test_unaligned_input_size_raises():
    with pytest.raises(ValueError, match="divisible"):
        _tiny(input_size=9)


def test_cond_only():
    model = _tiny()
    y = model(torch.randn(2, 3, 8, 8), cond=torch.randn(2, 16))
    assert y.shape == (2, 3, 8, 8)


def test_cond_width_mismatch_raises():
    model = _tiny()
    with pytest.raises(ValueError, match="cond_dim"):
        model(torch.randn(2, 3, 8, 8), cond=torch.randn(2, 8))


def test_no_conditioning_raises():
    model = _tiny()
    with pytest.raises(ValueError, match="at least one"):
        model(torch.randn(2, 3, 8, 8))


def test_y_without_label_embedder_raises():
    model = _tiny()
    with pytest.raises(ValueError, match="label embedder"):
        model(torch.randn(2, 3, 8, 8), torch.rand(2), torch.zeros(2, dtype=torch.long))


def test_unconditional_has_no_label_table():
    assert _tiny().y_embedder is None


def test_label_table_always_has_null_row():
    model = _tiny(num_classes=10)
    assert isinstance(model.y_embedder, LabelEmbedder)
    assert model.y_embedder.embedding.num_embeddings == 11
    assert model.y_embedder.null_label_id == 10
    null_y = torch.full((2,), 10, dtype=torch.long)
    y = model(torch.randn(2, 3, 8, 8), torch.rand(2), null_y)
    assert y.shape == (2, 3, 8, 8)


def test_class_dropout_wiring():
    model = _tiny(num_classes=10, class_dropout_prob=0.5)
    assert model.y_embedder.dropout_prob == 0.5


def test_class_dropout_without_num_classes_raises():
    with pytest.raises(ValueError, match="num_classes"):
        _tiny(class_dropout_prob=0.5)


def test_custom_y_embedder_conflicts_with_num_classes():
    with pytest.raises(ValueError, match="not both"):
        _tiny(y_embedder=nn.Embedding(4, 16), num_classes=4)


def test_custom_embedders_used():
    model = _tiny(y_embedder=nn.Embedding(4, 16))
    labels = torch.zeros(2, dtype=torch.long)
    y = model(torch.randn(2, 3, 8, 8), torch.rand(2), labels)
    assert y.shape == (2, 3, 8, 8)


def test_summed_conditioning():
    model = _tiny(num_classes=4)
    y = model(
        torch.randn(2, 3, 8, 8),
        torch.rand(2),
        torch.zeros(2, dtype=torch.long),
        cond=torch.randn(2, 16),
    )
    assert y.shape == (2, 3, 8, 8)


def test_zero_init_output():
    model = _tiny()
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(2, 3, 8, 8), torch.rand(2))
    assert torch.allclose(y, torch.zeros_like(y))


def test_wide_head():
    model = _tiny(head_dim=32, head_depth=1, head_num_heads=4)
    assert model.head_proj is not None
    assert len(model.head_blocks) == 1
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(2, 3, 8, 8), torch.rand(2))
    assert y.shape == (2, 3, 8, 8)
    assert torch.allclose(y, torch.zeros_like(y))


def test_default_head_adds_nothing():
    model = _tiny()
    assert model.head_proj is None
    assert len(model.head_blocks) == 0


def test_pos_embed_learnable():
    model = _tiny(pos_embed="learnable")
    assert isinstance(model.pos_embed, nn.Parameter)
    assert model.pos_embed.shape == (1, 4, 16)


def test_pos_embed_none():
    model = _tiny(pos_embed=None)
    assert model.pos_embed is None
    y = model(torch.randn(2, 3, 8, 8), torch.rand(2))
    assert y.shape == (2, 3, 8, 8)


def test_pos_embed_invalid_raises():
    with pytest.raises(ValueError, match="pos_embed"):
        _tiny(pos_embed="rope")


def test_gradient_flows():
    model = _tiny()
    target = torch.randn(1, 3, 8, 8)
    loss = (model(torch.randn(1, 3, 8, 8), torch.rand(1)) - target).pow(2).sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and (g.abs() > 0).any() for g in grads)


def test_preset_table_matches_paper():
    assert _DIT_CONFIGS == {
        "S": {"embed_dim": 384, "depth": 12, "num_heads": 6},
        "B": {"embed_dim": 768, "depth": 12, "num_heads": 12},
        "L": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
        "XL": {"embed_dim": 1152, "depth": 28, "num_heads": 16},
    }


@pytest.mark.parametrize("size", ["s", "b", "l", "xl"])
@pytest.mark.parametrize("patch", [2, 4, 8])
def test_preset_factories_exported(size, patch):
    assert callable(getattr(models, f"dit_{size}_{patch}"))


def test_preset_factory_configures_model():
    model = dit_s_4(input_size=4, in_channels=1)
    assert model.embed_dim == 384
    assert len(model.blocks) == 12
    assert model.num_heads == 6
    assert model.patch_size == 4


def test_preset_factory_kwargs_override():
    model = dit_s_4(input_size=4, in_channels=1, depth=1)
    assert len(model.blocks) == 1


def _reference_pos_embed(embed_dim, grid_size):
    """Verbatim port of get_2d_sincos_pos_embed from facebookresearch/DiT."""
    np = pytest.importorskip("numpy")

    def get_1d(embed_dim, pos):
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0)
    emb = np.concatenate(
        [get_1d(embed_dim // 2, grid[0]), get_1d(embed_dim // 2, grid[1])], axis=1
    )
    return torch.from_numpy(emb).float()


def test_pos_embed_bitwise_matches_reference():
    from torchebm.models.components import build_2d_sincos_pos_embed

    for embed_dim, grid in [(16, 4), (384, 16)]:
        ours = build_2d_sincos_pos_embed(embed_dim, grid)
        assert torch.equal(ours, _reference_pos_embed(embed_dim, grid))


def test_timestep_embedding_bitwise_matches_reference():
    import math

    from torchebm.models.components import MLPTimestepEmbedder

    t = torch.tensor([0.0, 0.25, 1.0, 999.0])
    dim = 256
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    reference = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    ours = MLPTimestepEmbedder.sinusoidal_embedding(t, dim)
    assert torch.equal(ours, reference)


def test_reference_init_scheme():
    torch.manual_seed(0)
    model = _tiny(num_classes=10)

    assert torch.all(model.patch_embed.proj.bias == 0)
    for block in model.blocks:
        assert torch.all(block.modulation[-1].weight == 0)
        assert torch.all(block.modulation[-1].bias == 0)
        for name in ("attn.qkv", "attn.out_proj"):
            mod = block.get_submodule(name)
            assert torch.all(mod.bias == 0)
            assert (mod.weight != 0).any()
    assert torch.all(model.head.modulation[-1].weight == 0)
    assert torch.all(model.head.proj.weight == 0)
    assert torch.all(model.t_embedder.mlp[0].bias == 0)
    assert 0.01 < model.t_embedder.mlp[0].weight.std().item() < 0.03
    assert 0.01 < model.y_embedder.embedding.weight.std().item() < 0.03


def test_custom_embedder_init_untouched():
    t_emb = nn.Linear(1, 16)
    with torch.no_grad():
        t_emb.weight.fill_(7.0)
    _tiny(t_embedder=t_emb)
    assert torch.all(t_emb.weight == 7.0)


def _fm_steps(model, n, **loss_kwargs):
    """Optimizer steps through FlowMatchingLoss. Zero-init gates gradient
    reach (reference behavior): step 1 only updates the head projection,
    step 2 the modulation weights, so the conditioning embedders receive
    gradient from step 3 onward."""
    from torchebm.losses import FlowMatchingLoss

    loss_fn = FlowMatchingLoss(model=model)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(n):
        loss = loss_fn(torch.randn(4, 3, 8, 8), **loss_kwargs)
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()


def test_flow_matching_trains_dit():
    torch.manual_seed(0)
    model = _tiny()
    _fm_steps(model, 3)
    assert (model.t_embedder.mlp[0].weight.grad.abs() > 0).any()
    assert (model.patch_embed.proj.weight.grad.abs() > 0).any()


def test_flow_matching_trains_class_conditional_dit():
    torch.manual_seed(0)
    model = _tiny(num_classes=4, class_dropout_prob=0.5)
    _fm_steps(model, 3, y=torch.randint(0, 4, (4,)))
    assert (model.y_embedder.embedding.weight.grad.abs() > 0).any()
