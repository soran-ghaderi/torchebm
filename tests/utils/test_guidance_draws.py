import math

import pytest
import torch
from torch.testing import assert_close

from torchebm.utils import sample_guidance


def _reference_sample_cfg_scale(u, s_max=7.0, cfg_beta=1.0):
    r"""`iMeanFlow.sample_cfg_scale` of Lyy-iiis/imeanflow (bf60cd7c) in PyTorch, from its uniforms."""
    if cfg_beta == 1.0:
        return torch.exp(u * math.log1p(s_max))
    log_base = (1.0 - cfg_beta) * math.log1p(s_max)
    log_inner = torch.log1p(u * math.expm1(log_base))
    return torch.exp(log_inner / (1.0 - cfg_beta))


def _reference_sample_cfg_interval(u_start, u_end, fm_mask):
    r"""`iMeanFlow.sample_cfg_interval` of Lyy-iiis/imeanflow (bf60cd7c) in PyTorch, noise at t = 1."""
    t_min = 0.5 * u_start
    t_max = 0.5 + 0.5 * u_end
    return torch.where(fm_mask, 0.0, t_min), torch.where(fm_mask, 1.0, t_max)


@pytest.mark.parametrize("power", [1.0, 2.0])
def test_scale_matches_imeanflow_reference(power):
    scale, _ = sample_guidance(
        1000, power=power, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
    )
    u = torch.rand(3, 1000, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    assert_close(scale, _reference_sample_cfg_scale(u[0], cfg_beta=power))


def test_interval_matches_imeanflow_reference_after_time_flip():
    r"""The reference draws with the complementary uniforms, which have the same law."""
    n = 1000
    _, interval = sample_guidance(
        n, flow_matching_ratio=0.5, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
    )
    u = torch.rand(3, n, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    t_min, t_max = _reference_sample_cfg_interval(1 - u[2], 1 - u[1], torch.arange(n) < n // 2)
    assert_close(interval, torch.stack([1 - t_max, 1 - t_min], dim=-1))


def test_ranges_and_full_interval_rows():
    scale, interval = sample_guidance(
        1000, low=1.0, high=4.0, flow_matching_ratio=0.25, generator=torch.Generator().manual_seed(0)
    )
    assert scale.shape == (1000,) and interval.shape == (1000, 2)
    assert (scale >= 1.0).all() and (scale <= 4.0).all()
    assert torch.equal(interval[:250], torch.tensor([0.0, 1.0]).expand(250, 2))
    assert (interval[250:, 0] <= 0.5).all() and (interval[250:, 1] >= 0.5).all()


def test_compile():
    compiled = torch.compile(sample_guidance, fullgraph=True)
    scale, interval = compiled(32, generator=None)
    assert scale.shape == (32,) and interval.shape == (32, 2) and (scale >= 1.0).all()
