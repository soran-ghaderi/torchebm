import torch

from torchebm.models.components.heads import AdaLNZeroPatchHead


def test_patch_head_output_shape():
    head = AdaLNZeroPatchHead(
        embed_dim=32, cond_dim=32, patch_size=4, out_channels=3
    )
    tokens = torch.randn(2, 16, 32)
    cond = torch.randn(2, 32)
    out = head(tokens, cond)
    assert out.shape == (2, 3, 16, 16)


def test_patch_head_zero_init_returns_zeros_on_any_input():
    head = AdaLNZeroPatchHead(
        embed_dim=16, cond_dim=16, patch_size=2, out_channels=3
    )
    head.eval()
    tokens = torch.randn(1, 4, 16)
    cond = torch.randn(1, 16)
    with torch.no_grad():
        out = head(tokens, cond)
    assert torch.allclose(out, torch.zeros_like(out))


def test_patch_head_default_cond_dim_matches_embed():
    head = AdaLNZeroPatchHead(embed_dim=16, patch_size=2, out_channels=3)
    assert head.cond_dim == 16
