"""FlowMatchingLoss: standard conditional flow matching on the shared base."""

import unittest.mock

import pytest
import torch
import torch.nn as nn

from torchebm.losses import EquilibriumMatchingLoss, FlowMatchingLoss
from torchebm.losses.loss_utils import get_interpolant, mean_flat


class ConstantField(nn.Module):
    def __init__(self, out_val=0.5):
        super().__init__()
        self.out_val = out_val

    def forward(self, x, t=None, **kwargs):
        return torch.full_like(x, self.out_val)


class LearnableField(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x, t=None, **kwargs):
        return self.linear(x)


class TimeRecordingField(nn.Module):
    def __init__(self):
        super().__init__()
        self.received_time = None

    def forward(self, x, t=None, **kwargs):
        self.received_time = t
        return x + t.view(-1, *([1] * (x.ndim - 1)))


class Negate(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x, t=None, **kwargs):
        return -self.inner(x, t, **kwargs)


@pytest.mark.parametrize("interpolant", ["linear", "cosine", "vp"])
def test_fm_loss_finite_scalar(interpolant):
    loss_fn = FlowMatchingLoss(model=ConstantField(), interpolant=interpolant)
    loss = loss_fn(torch.randn(16, 4))
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_fm_manual_verification():
    batch, dim = 2, 4
    x1 = torch.randn(batch, dim)
    x0 = torch.randn(batch, dim)
    t_raw = torch.rand(batch)

    with unittest.mock.patch(
        "torchebm.core.base_loss.torch.rand", return_value=t_raw
    ):
        loss_fn = FlowMatchingLoss(model=ConstantField(0.5), device="cpu")
        loss = loss_fn(x1, x0=x0)

    _, ut = get_interpolant("linear").interpolate(x0, x1, t_raw)
    expected = mean_flat((torch.full_like(ut, 0.5) - ut).square()).mean()
    assert torch.allclose(loss, expected, atol=1e-5)


def test_fm_equals_negated_eqm_constant_endpoint():
    """FM(v) == EqM(-v, ct='constant', ct_multiplier=1): values and gradients.

    EqM conditions the model on zeroed time by construction, so the
    equivalence holds for time-independent fields.
    """
    torch.manual_seed(0)
    batch, dim = 16, 4
    v_model = LearnableField(dim=dim)
    x1 = torch.randn(batch, dim)
    x0 = torch.randn(batch, dim)
    t_raw = torch.rand(batch)

    with unittest.mock.patch(
        "torchebm.core.base_loss.torch.rand", return_value=t_raw
    ):
        fm_loss = FlowMatchingLoss(model=v_model, device="cpu")(x1, x0=x0)
    fm_loss.backward()
    fm_grads = [p.grad.clone() for p in v_model.parameters()]
    v_model.zero_grad()

    with unittest.mock.patch(
        "torchebm.core.base_loss.torch.rand", return_value=t_raw
    ):
        eqm_loss = EquilibriumMatchingLoss(
            model=Negate(v_model), ct="constant", ct_multiplier=1.0, device="cpu"
        )(x1, x0=x0)

    assert torch.allclose(fm_loss, eqm_loss, atol=1e-6)
    eqm_loss.backward()
    for g_fm, p in zip(fm_grads, v_model.parameters()):
        assert torch.allclose(g_fm, p.grad, atol=1e-6)


def test_fm_model_receives_real_time():
    model = TimeRecordingField()
    loss_fn = FlowMatchingLoss(model=model)
    loss_fn(torch.randn(8, 4))
    assert model.received_time is not None
    assert not torch.all(model.received_time == 0)
    assert torch.all((model.received_time >= 0) & (model.received_time <= 1))


def test_fm_loss_weight_fn_applies():
    loss_fn = FlowMatchingLoss(
        model=ConstantField(), loss_weight_fn=lambda t: torch.zeros_like(t)
    )
    assert torch.equal(loss_fn(torch.randn(8, 4)), torch.zeros(()))


def test_fm_lognormal_t_sampler_dispatches():
    loss_fn = FlowMatchingLoss(model=ConstantField(), t_sampler="lognormal")
    t_first = loss_fn._sample_t(8, torch.Generator().manual_seed(6))
    t_second = loss_fn._sample_t(8, torch.Generator().manual_seed(6))
    assert torch.equal(t_first, t_second)
    assert not torch.all(t_first == t_first[0])


def test_fm_samples_through_flow_sampler_without_negation():
    from torchebm.samplers import FlowSampler

    sampler = FlowSampler(LearnableField(dim=2), integrator="euler")
    out = sampler.sample(x=torch.randn(4, 2), n_steps=5)
    assert out.shape == (4, 2)
    assert torch.isfinite(out).all()
