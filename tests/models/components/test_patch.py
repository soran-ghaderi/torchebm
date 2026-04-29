import pytest
import torch

from torchebm.models.components.patch import (
    ConvPatchEmbed2d,
    patchify2d,
    unpatchify2d,
)


@pytest.mark.parametrize("patch_size", [2, 4])
@pytest.mark.parametrize("shape", [(2, 3, 8, 8), (1, 1, 4, 4)])
def test_patchify_unpatchify_round_trip(patch_size, shape):
    b, c, h, w = shape
    x = torch.randn(*shape)
    tokens = patchify2d(x, patch_size)
    recon = unpatchify2d(tokens, patch_size, out_channels=c)
    assert recon.shape == x.shape
    assert torch.allclose(x, recon)


def test_patchify_shape():
    x = torch.randn(2, 3, 8, 8)
    tokens = patchify2d(x, 4)
    assert tokens.shape == (2, 4, 3 * 4 * 4)


def test_patchify_rejects_misaligned_dims():
    x = torch.randn(1, 3, 7, 8)
    with pytest.raises(ValueError):
        patchify2d(x, 4)


def test_unpatchify_rejects_bad_token_dim():
    tokens = torch.randn(1, 4, 10)
    with pytest.raises(ValueError):
        unpatchify2d(tokens, patch_size=4, out_channels=3)


def test_unpatchify_rejects_non_square_token_count():
    tokens = torch.randn(1, 5, 3 * 4 * 4)
    with pytest.raises(ValueError):
        unpatchify2d(tokens, patch_size=4, out_channels=3)


def test_conv_patch_embed_shape():
    m = ConvPatchEmbed2d(in_channels=3, embed_dim=32, patch_size=4)
    x = torch.randn(2, 3, 16, 16)
    out = m(x)
    assert out.shape == (2, 16, 32)
    assert torch.isfinite(out).all()


def test_conv_patch_embed_gradient_flows():
    m = ConvPatchEmbed2d(in_channels=3, embed_dim=16, patch_size=4)
    x = torch.randn(1, 3, 8, 8, requires_grad=True)
    loss = m(x).sum()
    loss.backward()
    assert x.grad is not None
    assert (x.grad.abs() > 0).any()
