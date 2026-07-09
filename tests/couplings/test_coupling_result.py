"""Tests for the CouplingResult structured return (pair-iterable + extras)."""

import dataclasses

import pytest
import torch

from torchebm.core import CouplingResult
from torchebm.couplings import IndependentCoupling, SinkhornCoupling


def test_pair_unpack():
    x0 = torch.randn(8, 2)
    x1 = torch.randn(8, 2)
    a, b = CouplingResult(x0, x1)
    assert a is x0 and b is x1


def test_iteration_yields_pair_only():
    res = CouplingResult(torch.randn(4, 2), torch.randn(4, 2), weights=torch.ones(4))
    assert len(tuple(res)) == 2


def test_attribute_access_and_weights_default():
    x0 = torch.randn(4, 2)
    x1 = torch.randn(4, 2)
    res = CouplingResult(x0, x1)
    assert res.x0 is x0 and res.x1 is x1
    assert res.weights is None
    w = torch.ones(4)
    assert CouplingResult(x0, x1, weights=w).weights is w


def test_frozen():
    res = CouplingResult(torch.randn(4, 2), torch.randn(4, 2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.weights = torch.ones(4)


def test_couplings_return_coupling_result():
    x0 = torch.randn(8, 2)
    x1 = torch.randn(8, 2)
    for coupling in (IndependentCoupling(), SinkhornCoupling()):
        res = coupling(x0, x1)
        assert isinstance(res, CouplingResult)
        assert res.weights is None


def test_pairing_families_require_x1():
    with pytest.raises(ValueError, match="x1 must not be None"):
        IndependentCoupling()(torch.randn(8, 2))
    with pytest.raises(ValueError, match="x1 must not be None"):
        SinkhornCoupling()(torch.randn(8, 2))
