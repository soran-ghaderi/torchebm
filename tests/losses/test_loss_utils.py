
import pytest
import torch
from torchebm.interpolants import (
    LinearInterpolant,
    CosineInterpolant,
    VariancePreservingInterpolant,
)
from torchebm.losses.loss_utils import (
    mean_flat,
    get_interpolant,
    compute_eqm_ct,
    dispersive_loss,
)


def test_mean_flat():
    # Shape: (batch, channels, height, width)
    x = torch.ones(2, 3, 4, 4)
    # Mean over (3, 4, 4) -> 3*4*4 = 48 elements per batch item
    # Since all ones, mean is 1.
    out = mean_flat(x)
    assert out.shape == (2,)
    assert torch.allclose(out, torch.ones(2))

    # Test with varying values
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    # Batch size 2, dim 1.
    # mean_flat over dim 1.
    out = mean_flat(x)
    assert torch.allclose(out, torch.tensor([1.5, 3.5]))


def test_get_interpolant():
    assert isinstance(get_interpolant("linear"), LinearInterpolant)
    assert isinstance(get_interpolant("cosine"), CosineInterpolant)
    assert isinstance(get_interpolant("vp"), VariancePreservingInterpolant)

    with pytest.raises(ValueError):
        get_interpolant("unknown")


def test_compute_eqm_ct():
    # Test formula: c(t) = 4 * min(1, (1-t)/0.2)
    
    t = torch.tensor([0.0, 0.8, 1.0])
    ct = compute_eqm_ct(t)
    
    # t=0: min(1, 5) * 4 = 4.
    # t=0.8: min(1, 1) * 4 = 4.
    # t=1.0: min(1, 0) * 4 = 0.
    
    assert torch.allclose(ct, torch.tensor([4.0, 4.0, 0.0]))

    # Test shape preservation
    t = torch.rand(10, 1)
    ct = compute_eqm_ct(t)
    assert ct.shape == t.shape


def test_dispersive_loss():
    z_same = torch.ones(4, 10)
    # pdist of identical vectors is 0.
    loss_same = dispersive_loss(z_same)
    assert torch.allclose(loss_same, torch.tensor(0.0))

    z_diff = torch.randn(4, 10)
    loss_diff = dispersive_loss(z_diff)
    assert loss_diff.shape == torch.Size([])
