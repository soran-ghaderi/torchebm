"""Tests for model-induced couplings (BaseModelCoupling, ReflowCoupling)."""

import pytest
import torch
import torch.nn as nn

from torchebm.core import BaseCoupling, BaseModelCoupling, CouplingResult
from torchebm.couplings import IndependentCoupling, ReflowCoupling, resolve_coupling
from torchebm.samplers import FlowSampler


class ConstantVelocity(nn.Module):
    """v(x, t) = 1 everywhere: the probability-flow ODE moves x by +1 over t in [0, 1]."""

    def forward(self, x, t):
        return torch.ones_like(x)


def test_reflow_callable_path_matches_analytic_map():
    phi = lambda x: 2.0 * x + 1.0
    coupling = ReflowCoupling(phi)
    x0 = torch.randn(16, 2)
    res = coupling(x0)
    assert isinstance(res, CouplingResult)
    y0, y1 = res
    assert y0 is x0
    assert torch.allclose(y1, phi(x0))


def test_reflow_ignores_incoming_x1():
    phi = lambda x: x + 3.0
    coupling = ReflowCoupling(phi)
    x0 = torch.randn(8, 2)
    garbage = torch.full((8, 2), 123.0)
    _, y1 = coupling(x0, garbage)
    assert torch.allclose(y1, x0 + 3.0)


def test_reflow_x1_optional():
    coupling = ReflowCoupling(lambda x: x)
    x0 = torch.randn(4, 2)
    y0, y1 = coupling(x0)
    assert y0 is x0 and torch.equal(y1, x0)


def test_reflow_flowsampler_path():
    sampler = FlowSampler(model=ConstantVelocity(), interpolant="linear")
    coupling = ReflowCoupling(sampler, n_steps=20)
    x0 = torch.randn(8, 2)
    _, y1 = coupling(x0)
    assert torch.allclose(y1, x0 + 1.0, atol=1e-3)


def test_reflow_no_grad():
    net = nn.Linear(2, 2)
    coupling = ReflowCoupling(lambda x: net(x))
    x0 = torch.randn(8, 2, requires_grad=True)
    _, y1 = coupling(x0)
    assert y1.grad_fn is None


def test_reflow_invalid_args():
    with pytest.raises(ValueError, match="n_steps must be positive"):
        ReflowCoupling(lambda x: x, n_steps=0)


def test_model_coupling_family_and_resolver():
    coupling = ReflowCoupling(lambda x: x)
    assert isinstance(coupling, BaseModelCoupling)
    assert isinstance(coupling, BaseCoupling)
    assert resolve_coupling(coupling, default="ot", owner="Owner") is coupling
    with pytest.raises(TypeError, match="Owner requires a BaseModelCoupling"):
        resolve_coupling(
            IndependentCoupling(), default="ot", owner="Owner", family=BaseModelCoupling
        )


def test_reflow_repr():
    assert "ReflowCoupling(n_steps=50)" in repr(ReflowCoupling(lambda x: x))
