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
