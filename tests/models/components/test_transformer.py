import pytest
import torch

from torchebm.models.components.transformer import (
    AdaLNZeroBlock,
    FeedForward,
    MultiheadSelfAttention,
    modulate,
)


def test_modulate_identity_when_shift_zero_scale_zero():
    x = torch.randn(2, 4, 8)
    zero = torch.zeros(2, 8)
    out = modulate(x, shift=zero, scale=zero)
    assert torch.allclose(out, x)


def test_modulate_shape():
    x = torch.randn(3, 5, 16)
    shift = torch.randn(3, 16)
    scale = torch.randn(3, 16)
    out = modulate(x, shift, scale)
    assert out.shape == x.shape


def test_multihead_attention_shape_and_finite():
    attn = MultiheadSelfAttention(embed_dim=32, num_heads=4)
    x = torch.randn(2, 10, 32)
    y = attn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_multihead_attention_rejects_bad_head_split():
    with pytest.raises(ValueError):
        MultiheadSelfAttention(embed_dim=30, num_heads=4)


def test_multihead_attention_dropout_zero_when_eval():
    attn = MultiheadSelfAttention(embed_dim=16, num_heads=2, dropout=0.5)
    attn.eval()
    x = torch.randn(1, 4, 16)
    with torch.no_grad():
        y1 = attn(x)
        y2 = attn(x)
    assert torch.allclose(y1, y2)


def test_feedforward_shape_and_grad():
    ff = FeedForward(embed_dim=16, mlp_ratio=2.0)
    x = torch.randn(2, 4, 16, requires_grad=True)
    y = ff(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None


def test_adaln_zero_block_starts_near_identity():
    block = AdaLNZeroBlock(embed_dim=16, num_heads=2, cond_dim=16)
    block.eval()
    x = torch.randn(2, 4, 16)
    cond = torch.randn(2, 16)
    with torch.no_grad():
        y = block(x, cond)
    assert torch.allclose(y, x)


def test_adaln_zero_block_shape_after_training_step():
    block = AdaLNZeroBlock(embed_dim=16, num_heads=2, cond_dim=8)
    x = torch.randn(2, 4, 16)
    cond = torch.randn(2, 8)
    y = block(x, cond)
    assert y.shape == x.shape


def test_adaln_zero_block_default_cond_dim_matches_embed():
    block = AdaLNZeroBlock(embed_dim=16, num_heads=2)
    assert block.cond_dim == 16
