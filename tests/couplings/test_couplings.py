"""Tests for minibatch couplings (IndependentCoupling, ExactOTCoupling, SinkhornCoupling).

Covers the torch-native exact solver (auction algorithm) against brute-force
optima, the log-domain Sinkhorn plan, and the coupling API contract.
"""

import itertools

import pytest
import torch

from torchebm.couplings import (
    ExactOTCoupling,
    GreedyCoupling,
    IndependentCoupling,
    SinkhornCoupling,
    UnbalancedSinkhornCoupling,
)
from torchebm.couplings.ot import (
    _auction_assignment,
    _greedy_assignment,
    _sinkhorn_log,
    _unbalanced_sinkhorn_log,
)


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ]
)
def device(request):
    return torch.device(request.param)


# IndependentCoupling
# ===================


def test_independent_coupling_is_identity():
    coupling = IndependentCoupling()
    x0 = torch.randn(16, 2)
    x1 = torch.randn(16, 2)
    y0, y1 = coupling(x0, x1)
    assert y0 is x0
    assert y1 is x1


def test_independent_coupling_ignores_kwargs():
    coupling = IndependentCoupling()
    x0 = torch.randn(8, 2)
    x1 = torch.randn(8, 2)
    y0, y1 = coupling(x0, x1, labels=torch.zeros(8))
    assert y0 is x0 and y1 is x1


def test_coupling_batch_mismatch_raises():
    coupling = IndependentCoupling()
    with pytest.raises(ValueError, match="equal batch sizes"):
        coupling(torch.randn(8, 2), torch.randn(4, 2))
    with pytest.raises(ValueError, match="equal batch sizes"):
        ExactOTCoupling()(torch.randn(8, 2), torch.randn(4, 2))


# Auction algorithm (exact assignment)
# ====================================


def test_auction_assignment_known_permutation():
    """Antipodal pairs: the optimal assignment is the swap."""
    cost = torch.tensor([[4.0, 0.1], [0.1, 4.0]])
    perm = _auction_assignment(cost)
    assert perm.tolist() == [1, 0]


def test_auction_assignment_identity_case():
    """Diagonal-dominant costs keep the identity assignment."""
    cost = torch.full((4, 4), 5.0)
    cost.fill_diagonal_(0.1)
    perm = _auction_assignment(cost)
    assert perm.tolist() == [0, 1, 2, 3]


def test_auction_assignment_is_permutation():
    torch.manual_seed(0)
    cost = torch.rand(32, 32)
    perm = _auction_assignment(cost)
    assert perm.sort().values.tolist() == list(range(32))


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_auction_assignment_matches_brute_force(n):
    """Auction total cost matches the exact optimum on small instances."""
    torch.manual_seed(n)
    cost = torch.rand(n, n)
    perm = _auction_assignment(cost, tol=1e-6)
    auction_cost = cost[torch.arange(n), perm].sum().item()
    brute_min = min(
        sum(cost[i, p[i]].item() for i in range(n))
        for p in itertools.permutations(range(n))
    )
    assert auction_cost == pytest.approx(brute_min, abs=1e-4)


def test_auction_assignment_single_element():
    perm = _auction_assignment(torch.tensor([[3.0]]))
    assert perm.tolist() == [0]


# Greedy assignment (approximate)
# ===============================


def test_greedy_assignment_known_permutation():
    """Antipodal pairs: the cheapest-first scan takes the swap."""
    cost = torch.tensor([[4.0, 0.1], [0.1, 4.0]])
    perm = _greedy_assignment(cost)
    assert perm.tolist() == [1, 0]


def test_greedy_assignment_is_permutation():
    torch.manual_seed(0)
    cost = torch.rand(32, 32)
    perm = _greedy_assignment(cost)
    assert perm.sort().values.tolist() == list(range(32))


def test_greedy_assignment_single_element():
    perm = _greedy_assignment(torch.tensor([[3.0]]))
    assert perm.tolist() == [0]


def test_greedy_matches_optimal_on_well_separated():
    """Diagonal-dominant costs: greedy recovers the exact identity optimum."""
    cost = torch.full((4, 4), 5.0)
    cost.fill_diagonal_(0.1)
    perm = _greedy_assignment(cost)
    assert perm.tolist() == [0, 1, 2, 3]


# Sinkhorn plan
# =============


def test_sinkhorn_log_marginals_uniform():
    torch.manual_seed(0)
    n = 16
    cost = torch.cdist(torch.randn(n, 2), torch.randn(n, 2)).square()
    cost = cost / cost.max()  # compute_cost normalizes the same way
    plan = _sinkhorn_log(cost, reg=0.1, n_iters=200)
    assert torch.allclose(plan.sum(dim=1), torch.full((n,), 1.0 / n), atol=1e-4)
    assert torch.allclose(plan.sum(dim=0), torch.full((n,), 1.0 / n), atol=1e-4)
    assert plan.sum() == pytest.approx(1.0, abs=1e-4)


def test_sinkhorn_coupling_seeded_reproducibility():
    coupling = SinkhornCoupling()
    x0 = torch.randn(32, 2)
    x1 = torch.randn(32, 2)

    torch.manual_seed(7)
    a0, a1 = coupling(x0, x1)
    torch.manual_seed(7)
    b0, b1 = coupling(x0, x1)
    assert torch.equal(a0, b0)
    assert torch.equal(a1, b1)


# Unbalanced Sinkhorn (weighted)
# ==============================


def test_unbalanced_reduces_to_balanced_for_large_marginal():
    """rho -> inf recovers balanced Sinkhorn (phi -> 1)."""
    torch.manual_seed(0)
    cost = torch.rand(16, 16)
    balanced = _sinkhorn_log(cost, reg=0.1, n_iters=200)
    relaxed = _unbalanced_sinkhorn_log(cost, reg=0.1, reg_marginal=1e6, n_iters=200)
    assert torch.allclose(balanced, relaxed, atol=1e-4)


def test_unbalanced_weights_uniform_on_matched_clusters():
    torch.manual_seed(0)
    x1 = torch.randn(32, 2) * 0.1
    x0 = torch.randn(32, 2) * 0.1
    res = UnbalancedSinkhornCoupling(reg=0.05, reg_marginal=1.0)(x0, x1)
    assert res.weights is not None
    assert res.weights.shape == (32,)
    assert bool((res.weights > 0).all())
    assert res.weights.mean().item() == pytest.approx(1.0, abs=1e-5)
    # Well-matched clusters: no source is an outlier, weights near uniform.
    assert res.weights.std().item() < 0.2


def test_unbalanced_downweights_outlier_source():
    """A source far from every target loses mass under marginal relaxation."""
    torch.manual_seed(0)
    x1 = torch.randn(16, 2) * 0.1
    x0 = torch.randn(16, 2) * 0.1
    x0[0] = torch.tensor([50.0, 50.0])  # transport cost ~ max for this source
    res = UnbalancedSinkhornCoupling(reg=0.05, reg_marginal=0.1)(x0, x1)
    assert res.weights[0] < res.weights[1:].min()


def test_unbalanced_seeded_reproducibility():
    coupling = UnbalancedSinkhornCoupling()
    x0 = torch.randn(16, 2)
    x1 = torch.randn(16, 2)
    torch.manual_seed(3)
    a = coupling(x0, x1)
    torch.manual_seed(3)
    b = coupling(x0, x1)
    assert torch.equal(a.x1, b.x1)
    assert torch.equal(a.weights, b.weights)


def test_unbalanced_invalid_args():
    with pytest.raises(ValueError, match="reg must be positive"):
        UnbalancedSinkhornCoupling(reg=0.0)
    with pytest.raises(ValueError, match="reg_marginal must be positive"):
        UnbalancedSinkhornCoupling(reg_marginal=0.0)
    with pytest.raises(ValueError, match="n_iters must be positive"):
        UnbalancedSinkhornCoupling(n_iters=0)


def test_unbalanced_single_sample_passthrough():
    coupling = UnbalancedSinkhornCoupling()
    x0 = torch.randn(1, 2)
    x1 = torch.randn(1, 2)
    res = coupling(x0, x1)
    assert res.x0 is x0 and res.x1 is x1 and res.weights is None


# OT couplings end-to-end
# =======================


def _transport_cost(x0, x1):
    return (x0 - x1).flatten(1).square().sum(dim=1).mean()


@pytest.mark.parametrize(
    "coupling_cls", [ExactOTCoupling, SinkhornCoupling, GreedyCoupling]
)
def test_ot_coupling_reduces_transport_cost(coupling_cls, device):
    """OT pairing beats the adversarial identity pairing on swapped clusters."""
    torch.manual_seed(0)
    n = 32
    cluster_a = torch.randn(n // 2, 2, device=device) * 0.1 + 5.0
    cluster_b = torch.randn(n // 2, 2, device=device) * 0.1 - 5.0
    x0 = torch.cat([cluster_a, cluster_b])
    x1 = torch.cat([cluster_b + 0.5, cluster_a + 0.5])  # crossed pairing

    coupling = coupling_cls()
    y0, y1 = coupling(x0, x1)
    assert _transport_cost(y0, y1) < _transport_cost(x0, x1)


def test_ot_coupling_exact_preserves_shape_dtype_device(device):
    coupling = ExactOTCoupling()
    x0 = torch.randn(8, 3, 4, 4, device=device, dtype=torch.float32)
    x1 = torch.randn(8, 3, 4, 4, device=device, dtype=torch.float32)
    y0, y1 = coupling(x0, x1)
    assert y0.shape == x0.shape and y1.shape == x1.shape
    assert y1.dtype == x1.dtype and y1.device == x1.device


def test_ot_coupling_exact_is_permutation_of_targets():
    torch.manual_seed(0)
    x0 = torch.randn(16, 2)
    x1 = torch.randn(16, 2)
    y0, y1 = ExactOTCoupling()(x0, x1)
    assert y0 is x0
    # Every original target appears exactly once.
    matches = (y1.unsqueeze(1) - x1.unsqueeze(0)).abs().sum(-1) == 0
    assert bool(matches.any(dim=1).all())
    assert bool(matches.any(dim=0).all())


@pytest.mark.parametrize(
    "coupling_cls", [ExactOTCoupling, SinkhornCoupling, GreedyCoupling]
)
def test_ot_coupling_single_sample_passthrough(coupling_cls):
    coupling = coupling_cls()
    x0 = torch.randn(1, 2)
    x1 = torch.randn(1, 2)
    y0, y1 = coupling(x0, x1)
    assert y0 is x0 and y1 is x1


def test_ot_coupling_no_grad():
    """Coupled tensors carry no graph even for differentiable inputs."""
    x0 = torch.randn(8, 2, requires_grad=True)
    x1 = torch.randn(8, 2, requires_grad=True)
    _, y1 = ExactOTCoupling()(x0, x1)
    assert y1.grad_fn is None


def test_sinkhorn_coupling_invalid_args():
    with pytest.raises(ValueError, match="reg must be positive"):
        SinkhornCoupling(reg=0.0)
    with pytest.raises(ValueError, match="n_iters must be positive"):
        SinkhornCoupling(n_iters=0)


def test_greedy_coupling_is_permutation_of_targets():
    torch.manual_seed(0)
    x0 = torch.randn(16, 2)
    x1 = torch.randn(16, 2)
    y0, y1 = GreedyCoupling()(x0, x1)
    assert y0 is x0
    matches = (y1.unsqueeze(1) - x1.unsqueeze(0)).abs().sum(-1) == 0
    assert bool(matches.any(dim=1).all())
    assert bool(matches.any(dim=0).all())


def test_ot_coupling_repr():
    assert "ExactOTCoupling" in repr(ExactOTCoupling())
    assert "SinkhornCoupling" in repr(SinkhornCoupling())
    assert "GreedyCoupling" in repr(GreedyCoupling())
