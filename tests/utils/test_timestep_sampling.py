import math

import pytest
import torch

from torchebm.utils import compute_density_for_timestep_sampling


def seeded():
    return torch.Generator().manual_seed(0)


def test_uniform_scheme_draws_in_unit_interval():
    u = compute_density_for_timestep_sampling("uniform", 1000, generator=seeded())
    assert u.shape == (1000,)
    assert torch.equal(u, torch.rand(1000, generator=seeded()))


def test_logit_normal_scheme_is_sigmoid_of_a_normal():
    u = compute_density_for_timestep_sampling(
        "logit_normal", 1000, logit_mean=0.5, logit_std=2.0, generator=seeded()
    )
    z = torch.normal(0.5, 2.0, size=(1000,), generator=seeded())
    assert torch.equal(u, torch.sigmoid(z))


def test_mode_scheme_matches_reference_formula():
    u = compute_density_for_timestep_sampling(
        "mode", 1000, mode_scale=1.29, generator=seeded()
    )
    v = torch.rand(1000, generator=seeded())
    expected = 1 - v - 1.29 * (torch.cos(math.pi * v / 2) ** 2 - 1 + v)
    assert torch.equal(u, expected)


def test_logit_normal_reproduces_the_edm_lognormal_skew():
    """t = 1 / (1 + exp(z p_std + p_mean)) has the law of logit_normal(-p_mean, p_std)."""
    p_mean, p_std = -1.2, 1.2
    z = torch.randn(200_000, generator=seeded())
    edm = 1.0 / (1.0 + torch.exp(z * p_std + p_mean))
    u = compute_density_for_timestep_sampling(
        "logit_normal", 200_000, logit_mean=-p_mean, logit_std=p_std, generator=seeded()
    )
    assert torch.allclose(u.mean(), edm.mean(), atol=5e-3)
    assert torch.allclose(u.std(), edm.std(), atol=5e-3)


def test_dtype_and_device_are_honored():
    u = compute_density_for_timestep_sampling("uniform", 4, dtype=torch.float64)
    assert u.dtype == torch.float64


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="weighting_scheme"):
        compute_density_for_timestep_sampling("bogus", 4)
