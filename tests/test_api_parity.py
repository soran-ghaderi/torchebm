"""Legacy API against the target API on identical inputs and seeds.

Each step of the standard-API migration (.progress/STANDARD_API.md) adds its
pair here: the legacy call and the target call, one seeded generator for
both, compared with torch.equal wherever the operations are the same. A pair
stays until its legacy path is removed in 1.0.0 and is deleted with it.
"""

import pytest
import torch

from torchebm.core import HarmonicModel
from torchebm.interpolants import (
    CosineInterpolant,
    LinearInterpolant,
    VarianceExplodingInterpolant,
    VariancePreservingInterpolant,
)

INTERPOLANTS = [
    LinearInterpolant,
    CosineInterpolant,
    VariancePreservingInterpolant,
    VarianceExplodingInterpolant,
]


def test_score_prediction_matches_legacy_gradient():
    x = torch.randn(8, 3)
    legacy = -HarmonicModel(k=2.0).gradient(x)
    with torch.no_grad():
        target = HarmonicModel(k=2.0, prediction_type="score")(x)
    assert torch.equal(target, legacy)


@pytest.mark.parametrize("interpolant", INTERPOLANTS)
def test_interpolant_call_matches_legacy_interpolate(interpolant):
    torch.manual_seed(0)
    x0, x1, t = torch.randn(8, 3), torch.randn(8, 3), torch.rand(8)
    legacy = interpolant().interpolate(x0, x1, t)
    target = interpolant(prediction_type="velocity")(x0, x1, t)
    assert all(torch.equal(a, b) for a, b in zip(target, legacy))
    assert interpolant(prediction_type="epsilon")(x0, x1, t)[1] is x0
    assert interpolant(prediction_type="sample")(x0, x1, t)[1] is x1
