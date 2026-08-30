"""Shared interpolant-loss base: EqM and EnergyMatching sit on one skeleton."""

import pytest
import torch
import torch.nn as nn

from torchebm.core import BaseModel
from torchebm.core.base_loss import BaseInterpolantLoss
from torchebm.losses import EnergyMatchingLoss, EquilibriumMatchingLoss


class QuadraticPotential(BaseModel):
    def forward(self, x, **kwargs):
        return 0.5 * x.flatten(1).square().sum(dim=1)


class LinearField(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x, t=None, **kwargs):
        return self.linear(x)


def test_both_losses_share_the_base():
    eqm = EquilibriumMatchingLoss(model=LinearField())
    em = EnergyMatchingLoss(model=QuadraticPotential(), lambda_cd=0.0)
    assert isinstance(eqm, BaseInterpolantLoss)
    assert isinstance(em, BaseInterpolantLoss)
    assert type(eqm.coupling).__name__ == "IndependentCoupling"
    assert type(em.coupling).__name__ != "IndependentCoupling"


def test_em_accepts_and_dispatches_t_sampler():
    em = EnergyMatchingLoss(
        model=QuadraticPotential(), lambda_cd=0.0, t_sampler="lognormal"
    )
    t_first = em._sample_t(8, torch.Generator().manual_seed(4))
    t_second = em._sample_t(8, torch.Generator().manual_seed(4))
    assert torch.equal(t_first, t_second)
    assert not torch.all(t_first == t_first[0])
    assert torch.all((t_first > 0) & (t_first <= 1))


def test_em_applies_loss_weight_fn():
    model = QuadraticPotential()
    em_weighted = EnergyMatchingLoss(
        model=model, lambda_cd=0.0, loss_weight_fn=lambda t: torch.zeros_like(t)
    )
    loss = em_weighted(torch.randn(8, 4), generator=torch.Generator().manual_seed(2))
    assert torch.equal(loss, torch.zeros(()))


def test_em_invalid_t_sampler_raises():
    with pytest.raises(ValueError, match="t_sampler"):
        EnergyMatchingLoss(model=QuadraticPotential(), t_sampler="bogus")


def test_em_non_callable_loss_weight_fn_raises():
    with pytest.raises(TypeError, match="callable or None"):
        EnergyMatchingLoss(model=QuadraticPotential(), loss_weight_fn=5)


def test_em_lognormal_matches_edm_formula():
    import unittest.mock

    z = torch.tensor([0.0, 1.0, -1.0, 0.5])
    em = EnergyMatchingLoss(
        model=QuadraticPotential(), lambda_cd=0.0, t_sampler="lognormal"
    )
    with unittest.mock.patch(
        "torchebm.core.base_loss.torch.randn", return_value=z
    ):
        t = em._sample_t(4, generator=None)
    sigma = torch.exp(z * 1.2 - 1.2)
    expected = (1.0 / (1.0 + sigma)).clamp(min=1e-4, max=1.0)
    assert torch.allclose(t, expected, atol=1e-6)


def test_em_callable_t_sampler_contract():
    def sampler(batch, *, device, dtype, generator):
        assert dtype == torch.float32
        return torch.full((batch,), 0.3, device=device, dtype=dtype)

    em = EnergyMatchingLoss(
        model=QuadraticPotential(), lambda_cd=0.0, t_sampler=sampler, device="cpu"
    )
    assert torch.allclose(em._sample_t(4, None), torch.full((4,), 0.3))


def test_em_lognormal_t_flows_into_weighted_flow_loss():
    """The drawn lognormal t reaches the flow term, exposed via loss_weight_fn.

    QuadraticPotential gives grad V = xt in closed form, so the flow term is
    computable by hand: mean over pairs of w(t) * ||-xt - ut||^2 * t.
    """
    import unittest.mock

    from torchebm.losses.loss_utils import compute_flow_weight, mean_flat
    from torchebm.losses.loss_utils import get_interpolant

    batch, dim = 2, 4
    x1 = torch.randn(batch, dim)
    x0 = torch.randn(batch, dim)
    z = torch.tensor([0.3, -0.7])

    em = EnergyMatchingLoss(
        model=QuadraticPotential(),
        lambda_cd=0.0,
        sigma=0.0,
        coupling="independent",
        t_sampler="lognormal",
        loss_weight_fn=lambda t: t,
        device="cpu",
    )
    with unittest.mock.patch(
        "torchebm.core.base_loss.torch.randn", return_value=z
    ):
        loss = em(x1, x0=x0)

    t = (1.0 / (1.0 + torch.exp(z * 1.2 - 1.2))).clamp(min=1e-4, max=1.0)
    xt, ut = get_interpolant("linear").interpolate(x0, x1, t)
    w = compute_flow_weight(t, cutoff=em.flow_weight_cutoff)
    expected = (w * mean_flat((-xt - ut).square()) * t).mean()
    assert torch.allclose(loss, expected, atol=1e-5)
