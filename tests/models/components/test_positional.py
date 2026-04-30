import pytest
import torch

from torchebm.models.components.positional import (
    _get_1d_sincos_pos_embed_from_grid,
    build_2d_sincos_pos_embed,
)


@pytest.mark.parametrize("embed_dim", [16, 32, 64])
@pytest.mark.parametrize("grid_size", [4, 8])
def test_build_2d_sincos_pos_embed_shape(embed_dim, grid_size):
    pe = build_2d_sincos_pos_embed(embed_dim, grid_size)
    assert pe.shape == (grid_size * grid_size, embed_dim)
    assert torch.isfinite(pe).all()


def test_build_2d_sincos_pos_embed_dtype_device():
    pe = build_2d_sincos_pos_embed(32, 4, dtype=torch.float64)
    assert pe.dtype == torch.float64
    assert pe.device.type == "cpu"


def test_build_2d_sincos_pos_embed_rejects_odd_dim():
    with pytest.raises(ValueError):
        build_2d_sincos_pos_embed(31, 4)


def test_1d_sincos_helper_rejects_odd_dim():
    pos = torch.arange(4)
    with pytest.raises(ValueError):
        _get_1d_sincos_pos_embed_from_grid(5, pos)


def test_pos_embed_distinct_positions_differ():
    pe = build_2d_sincos_pos_embed(16, 4)
    assert not torch.allclose(pe[0], pe[1])
    assert not torch.allclose(pe[0], pe[-1])
