"""Tests for the BaseCostCoupling family template (couple pipeline, cost hook)."""

import pytest
import torch

from torchebm.core import BaseCostCoupling


class IdentitySolveCoupling(BaseCostCoupling):
    """Minimal concrete: pair each source with the same-index target."""

    def _solve(self, cost, generator=None):
        return torch.arange(cost.shape[0], device=cost.device)


class CaptureCostCoupling(BaseCostCoupling):
    """Records the kwargs forwarded to compute_cost."""

    def __init__(self):
        self.seen_kwargs = None

    def compute_cost(self, x0, x1, **kwargs):
        self.seen_kwargs = kwargs
        return super().compute_cost(x0, x1, **kwargs)

    def _solve(self, cost, generator=None):
        return torch.arange(cost.shape[0], device=cost.device)


def test_template_batch_mismatch_raises():
    with pytest.raises(ValueError, match="equal batch sizes"):
        IdentitySolveCoupling()(torch.randn(8, 2), torch.randn(4, 2))


def test_template_single_sample_passthrough():
    coupling = IdentitySolveCoupling()
    x0 = torch.randn(1, 2)
    x1 = torch.randn(1, 2)
    y0, y1 = coupling(x0, x1)
    assert y0 is x0 and y1 is x1


def test_template_runs_under_no_grad():
    coupling = IdentitySolveCoupling()
    x0 = torch.randn(8, 2, requires_grad=True)
    x1 = torch.randn(8, 2, requires_grad=True)
    _, y1 = coupling(x0, x1)
    assert y1.grad_fn is None


def test_default_compute_cost_is_normalized_squared_cdist():
    torch.manual_seed(0)
    x0 = torch.randn(6, 3, 4, 4)
    x1 = torch.randn(6, 3, 4, 4)
    cost = IdentitySolveCoupling().compute_cost(x0, x1)

    expected = torch.cdist(x0.reshape(6, -1), x1.reshape(6, -1)).square()
    expected = expected / expected.max().clamp(min=1e-12)
    assert torch.allclose(cost, expected)
    assert cost.shape == (6, 6)
    assert cost.max() == pytest.approx(1.0)


def test_couple_kwargs_reach_compute_cost():
    coupling = CaptureCostCoupling()
    labels = torch.arange(8)
    coupling(torch.randn(8, 2), torch.randn(8, 2), labels=labels)
    assert coupling.seen_kwargs is not None
    assert torch.equal(coupling.seen_kwargs["labels"], labels)
