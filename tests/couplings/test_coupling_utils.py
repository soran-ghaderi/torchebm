"""Tests for the coupling registry and resolver."""

import pytest

from torchebm.core import BaseCostCoupling, BaseCoupling
from torchebm.couplings import (
    ExactOTCoupling,
    GreedyCoupling,
    IndependentCoupling,
    SinkhornCoupling,
    UnbalancedSinkhornCoupling,
    get_coupling,
    resolve_coupling,
)
from torchebm.couplings.coupling_utils import _COUPLING_NAMES


@pytest.mark.parametrize(
    "name,cls",
    [
        ("independent", IndependentCoupling),
        ("ot", ExactOTCoupling),
        ("exact_ot", ExactOTCoupling),
        ("sinkhorn", SinkhornCoupling),
        ("greedy", GreedyCoupling),
        ("unbalanced_sinkhorn", UnbalancedSinkhornCoupling),
    ],
)
def test_get_coupling_resolves_registry(name, cls):
    assert isinstance(get_coupling(name), cls)


def test_get_coupling_covers_every_registry_name():
    for name in _COUPLING_NAMES:
        assert isinstance(get_coupling(name), BaseCoupling)


def test_get_coupling_unknown_lists_valid_names():
    with pytest.raises(ValueError, match="Unknown coupling") as exc:
        get_coupling("emd")
    message = str(exc.value)
    assert "sinkhorn" in message and "independent" in message


def test_resolve_coupling_none_uses_default():
    coupling = resolve_coupling(None, default="sinkhorn", owner="Owner")
    assert isinstance(coupling, SinkhornCoupling)


def test_resolve_coupling_string_uses_registry():
    coupling = resolve_coupling("independent", default="ot", owner="Owner")
    assert isinstance(coupling, IndependentCoupling)


def test_resolve_coupling_instance_passthrough_is_identity():
    instance = SinkhornCoupling(reg=0.01)
    assert resolve_coupling(instance, default="ot", owner="Owner") is instance


def test_resolve_coupling_wrong_family_raises():
    with pytest.raises(TypeError, match="Owner requires a BaseCostCoupling"):
        resolve_coupling(
            IndependentCoupling(),
            default="ot",
            owner="Owner",
            family=BaseCostCoupling,
        )
