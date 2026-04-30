import pytest
import torch

from torchebm.models.conditional_transformer_2d import ConditionalTransformer2D


def _tiny_model(**overrides):
    cfg = dict(
        in_channels=3,
        out_channels=3,
        input_size=8,
        patch_size=4,
        embed_dim=16,
        depth=2,
        num_heads=2,
        cond_dim=16,
    )
    cfg.update(overrides)
    return ConditionalTransformer2D(**cfg)


def test_conditional_transformer_shape():
    model = _tiny_model()
    x = torch.randn(2, 3, 8, 8)
    cond = torch.randn(2, 16)
    y = model(x, cond)
    assert y.shape == (2, 3, 8, 8)


def test_conditional_transformer_without_pos_embed():
    model = _tiny_model(use_sincos_pos_embed=False)
    assert model.pos_embed is None
    x = torch.randn(1, 3, 8, 8)
    cond = torch.randn(1, 16)
    y = model(x, cond)
    assert y.shape == (1, 3, 8, 8)


def test_conditional_transformer_rejects_unaligned_input_size():
    with pytest.raises(ValueError):
        _tiny_model(input_size=9, patch_size=4)


def test_conditional_transformer_zero_init_output():
    model = _tiny_model()
    model.eval()
    x = torch.randn(2, 3, 8, 8)
    cond = torch.randn(2, 16)
    with torch.no_grad():
        y = model(x, cond)
    assert torch.allclose(y, torch.zeros_like(y))


def test_conditional_transformer_different_cond_dim():
    model = _tiny_model(cond_dim=8)
    x = torch.randn(1, 3, 8, 8)
    cond = torch.randn(1, 8)
    y = model(x, cond)
    assert y.shape == (1, 3, 8, 8)


def test_conditional_transformer_gradient_flows():
    model = _tiny_model()
    x = torch.randn(1, 3, 8, 8)
    cond = torch.randn(1, 16)
    target = torch.randn(1, 3, 8, 8)
    # AdaLN-Zero zeros the output at init, so use a non-zero target to exercise backprop
    loss = (model(x, cond) - target).pow(2).sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and (g.abs() > 0).any() for g in grads)
