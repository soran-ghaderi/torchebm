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


class LearnableTimeField(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.linear = nn.Linear(dim + 1, dim)

    def forward(self, x, t=None, **kwargs):
        return self.linear(torch.cat([x, t.unsqueeze(-1)], dim=-1))


def _fixed_t(t):
    return lambda batch, *, device, dtype, generator: t.to(device=device, dtype=dtype)


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


def test_fm_equals_negated_eqm_true_clock_for_time_conditioned_field():
    """FM(v) == EqM(-v, ct='constant', ct_multiplier=1, model_time='true')."""
    torch.manual_seed(0)
    batch, dim = 16, 4
    v_model = LearnableTimeField(dim=dim)
    x1, x0, t = torch.randn(batch, dim), torch.randn(batch, dim), torch.rand(batch)

    fm_loss = FlowMatchingLoss(model=v_model, t_sampler=_fixed_t(t))(x1, x0=x0)
    eqm_loss = EquilibriumMatchingLoss(
        model=Negate(v_model),
        ct="constant",
        ct_multiplier=1.0,
        model_time="true",
        t_sampler=_fixed_t(t),
    )(x1, x0=x0)
    assert torch.allclose(fm_loss, eqm_loss, atol=1e-6)

    zeroed = EquilibriumMatchingLoss(
        model=Negate(v_model), ct="constant", ct_multiplier=1.0, t_sampler=_fixed_t(t)
    )(x1, x0=x0)
    assert not torch.allclose(fm_loss, zeroed, atol=1e-6)


# negate_velocity
# ===============


def test_fm_negate_velocity_target():
    batch, dim = 4, 3
    x1, x0, t = torch.randn(batch, dim), torch.randn(batch, dim), torch.rand(batch)
    loss = FlowMatchingLoss(
        model=ConstantField(0.5), negate_velocity=True, t_sampler=_fixed_t(t)
    )(x1, x0=x0)
    _, ut = get_interpolant("linear").interpolate(x0, x1, t)
    expected = mean_flat((torch.full_like(ut, 0.5) + ut).square()).mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_fm_negate_velocity_false_is_default_path():
    batch, dim = 8, 4
    x1, x0, t = torch.randn(batch, dim), torch.randn(batch, dim), torch.rand(batch)
    model = ConstantField(0.5)
    default = FlowMatchingLoss(model=model, t_sampler=_fixed_t(t))(x1, x0=x0)
    explicit = FlowMatchingLoss(
        model=model, negate_velocity=False, t_sampler=_fixed_t(t)
    )(x1, x0=x0)
    assert torch.equal(default, explicit)


def test_fm_negated_is_bitwise_eqm_constant_true_clock():
    """FM(negate_velocity=True) == EqM(ct='constant', ct_multiplier=1, model_time='true')."""
    torch.manual_seed(0)
    batch, dim = 16, 4
    model = LearnableTimeField(dim=dim)
    x1, x0, t = torch.randn(batch, dim), torch.randn(batch, dim), torch.rand(batch)

    fm_loss = FlowMatchingLoss(
        model=model, negate_velocity=True, t_sampler=_fixed_t(t)
    )(x1, x0=x0)
    fm_loss.backward()
    fm_grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()

    eqm_loss = EquilibriumMatchingLoss(
        model=model,
        ct="constant",
        ct_multiplier=1.0,
        model_time="true",
        t_sampler=_fixed_t(t),
    )(x1, x0=x0)
    assert torch.equal(fm_loss, eqm_loss)
    eqm_loss.backward()
    for g_fm, p in zip(fm_grads, model.parameters()):
        assert torch.equal(g_fm, p.grad)


def test_fm_negated_field_descends_through_eqm_energy():
    from torchebm.models import EqMEnergy
    from torchebm.samplers import GradientDescentSampler

    field = LearnableField(dim=2)
    FlowMatchingLoss(model=field, negate_velocity=True)(torch.randn(8, 2)).backward()
    sampler = GradientDescentSampler(EqMEnergy(field, energy_type="implicit"), step_size=0.1)
    out = sampler.sample(x=torch.randn(4, 2), n_steps=5)
    assert out.shape == (4, 2)
    assert torch.isfinite(out).all()


def test_fm_negate_velocity_in_repr():
    assert "negate_velocity=False" in repr(FlowMatchingLoss(model=ConstantField()))
    assert "negate_velocity=True" in repr(
        FlowMatchingLoss(model=ConstantField(), negate_velocity=True)
    )


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
