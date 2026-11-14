import torch
from torch.testing import assert_close

from torchebm.utils import sample_time_pairs


def _reference_sample_tr(batch_size, generator, P_mean=-0.4, P_std=1.0, data_proportion=0.75):
    r"""`MeanFlow.sample_tr` of Gsunshine/meanflow (d70cb55) in PyTorch, noise at t = 1."""
    t = torch.sigmoid(torch.randn(batch_size, generator=generator, dtype=torch.float64) * P_std + P_mean)
    r = torch.sigmoid(torch.randn(batch_size, generator=generator, dtype=torch.float64) * P_std + P_mean)
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    zero_mask = torch.arange(batch_size) < int(batch_size * data_proportion)
    return t, torch.where(zero_mask, t, r)


def test_matches_meanflow_reference_after_time_flip():
    n = 400_000
    s, t = sample_time_pairs(n, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    t_ref, r_ref = _reference_sample_tr(n, torch.Generator().manual_seed(1))
    s_ref, t_flipped = 1 - t_ref, 1 - r_ref
    q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64)
    for x, x_ref in ((s, s_ref), (t, t_flipped), (t - s, t_flipped - s_ref)):
        assert_close(x.quantile(q), x_ref.quantile(q), atol=3e-3, rtol=0)
    assert_close((s == t).double().mean(), (s_ref == t_flipped).double().mean())


def test_pairs_are_ordered_with_flow_matching_share():
    s, t = sample_time_pairs(1000, flow_matching_ratio=0.25, generator=torch.Generator().manual_seed(0))
    assert s.shape == t.shape == (1000,)
    assert (s <= t).all() and (s > 0).all() and (t < 1).all()
    assert torch.equal(s[:250], t[:250])
    assert (s[250:] < t[250:]).all()


def test_uniform_scheme_and_generator():
    pair = sample_time_pairs(64, "uniform", generator=torch.Generator().manual_seed(3))
    again = sample_time_pairs(64, "uniform", generator=torch.Generator().manual_seed(3))
    assert_close(pair, again)
    draws = torch.rand(128, generator=torch.Generator().manual_seed(3)).view(2, 64)
    assert_close(pair[0][48:], draws.amin(0)[48:])
    assert_close(pair[1][48:], draws.amax(0)[48:])


def test_compile():
    compiled = torch.compile(sample_time_pairs, fullgraph=True)
    s, t = compiled(32, generator=None)
    assert s.shape == (32,) and (s <= t).all()
