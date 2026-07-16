r"""Minibatch optimal-transport couplings.

All solvers are torch-native (no scipy/POT dependency):

- `ExactOTCoupling`: Bertsekas auction algorithm with epsilon scaling, solving
  the discrete assignment problem to (float) optimality; a deterministic
  permutation of the target batch.
- `SinkhornCoupling`: log-domain Sinkhorn iterations producing an entropic
  transport plan, from which target indices are drawn row-conditionally.
- `GreedyCoupling`: nearest-free-pair heuristic; a fast, deterministic
  approximation to the assignment problem.

All (approximately) minimize the squared-Euclidean transport cost

\[
\min_{\pi} \sum_i \|x_0^{(i)} - x_1^{(\pi(i))}\|^2
\]

which straightens interpolation paths (OT-CFM, Energy Matching warm-up).
"""

from __future__ import annotations

import math

import torch

from torchebm.core import BaseCostCoupling, CouplingResult


def _sinkhorn_log(C: torch.Tensor, reg: float, n_iters: int) -> torch.Tensor:
    r"""Log-domain Sinkhorn iterations with uniform marginals.

    Args:
        C: Cost matrix of shape (n, m).
        reg: Entropic regularization strength.
        n_iters: Number of Sinkhorn iterations.

    Returns:
        Transport plan of shape (n, m) summing to 1 with (approximately)
        uniform marginals 1/n and 1/m.
    """
    n, m = C.shape
    M = -C / reg
    log_mu = -math.log(n)
    log_nu = -math.log(m)
    f = torch.zeros(n, dtype=C.dtype, device=C.device)
    g = torch.zeros(m, dtype=C.dtype, device=C.device)
    for _ in range(n_iters):
        f = log_mu - torch.logsumexp(M + g.unsqueeze(0), dim=1)
        g = log_nu - torch.logsumexp(M + f.unsqueeze(1), dim=0)
    return (M + f.unsqueeze(1) + g.unsqueeze(0)).exp()


def _unbalanced_sinkhorn_log(
    C: torch.Tensor, reg: float, reg_marginal: float, n_iters: int
) -> torch.Tensor:
    r"""Log-domain unbalanced Sinkhorn with KL-relaxed marginals.

    Applies the damping factor \(\phi = \rho / (\rho + \varepsilon)\)
    (Chizat et al., 2018) to both potential updates, where \(\rho\) is the
    marginal-relaxation strength and \(\varepsilon\) the entropic
    regularization. \(\rho \to \infty\) (\(\phi \to 1\)) recovers balanced
    Sinkhorn; small \(\rho\) lets the plan create/destroy mass, down-weighting
    samples that are expensive to transport (outliers).

    Args:
        C: Cost matrix of shape (n, m).
        reg: Entropic regularization strength \(\varepsilon\).
        reg_marginal: KL marginal-relaxation strength \(\rho\).
        n_iters: Number of Sinkhorn iterations.

    Returns:
        Transport plan of shape (n, m); marginals are only softly enforced,
        so row/column masses need not be uniform.
    """
    n, m = C.shape
    M = -C / reg
    log_mu = -math.log(n)
    log_nu = -math.log(m)
    fi = reg_marginal / (reg_marginal + reg)
    f = torch.zeros(n, dtype=C.dtype, device=C.device)
    g = torch.zeros(m, dtype=C.dtype, device=C.device)
    for _ in range(n_iters):
        f = fi * (log_mu - torch.logsumexp(M + g.unsqueeze(0), dim=1))
        g = fi * (log_nu - torch.logsumexp(M + f.unsqueeze(1), dim=0))
    return (M + f.unsqueeze(1) + g.unsqueeze(0)).exp()


def _auction_assignment(
    cost: torch.Tensor,
    tol: float = 1e-4,
    scale_factor: float = 8.0,
    max_rounds: int = 0,
) -> torch.Tensor:
    r"""Solve the min-cost assignment problem via the auction algorithm.

    Vectorized Bertsekas forward auction with epsilon scaling. Each round,
    all unassigned rows bid simultaneously (`topk(2)` for best/second-best
    margins); conflicts are resolved with `scatter_reduce`. Prices persist
    across scaling phases. The final assignment cost is within `tol` of the
    optimum (exact for well-separated costs).

    Args:
        cost: Square cost matrix of shape (n, n).
        tol: Optimality tolerance on the total assignment cost, relative to
            the cost scale.
        scale_factor: Epsilon division factor between scaling phases.
        max_rounds: Safety cap on total bidding rounds; 0 selects
            `max(200, 100 * n)`. On hitting the cap, remaining rows are
            paired with remaining columns arbitrarily (valid, near-optimal).

    Returns:
        Long tensor `perm` of shape (n,) such that row `i` is assigned to
        column `perm[i]`.
    """
    n = cost.shape[0]
    device = cost.device
    if n == 1:
        return torch.zeros(1, dtype=torch.long, device=device)

    benefit = -cost.double()
    prices = torch.zeros(n, dtype=benefit.dtype, device=device)
    assign_row = torch.full((n,), -1, dtype=torch.long, device=device)
    assign_col = torch.full((n,), -1, dtype=torch.long, device=device)

    eps_final = tol / n
    spread = float(benefit.max() - benefit.min())
    eps = max(spread / 4.0, eps_final)
    if max_rounds <= 0:
        max_rounds = max(200, 100 * n)

    rounds = 0
    while True:
        assign_row.fill_(-1)
        assign_col.fill_(-1)
        while bool((assign_row < 0).any()):
            rounds += 1
            if rounds > max_rounds:
                break
            unassigned = (assign_row < 0).nonzero(as_tuple=True)[0]
            values = benefit[unassigned] - prices.unsqueeze(0)
            top2 = values.topk(2, dim=1)
            best_col = top2.indices[:, 0]
            margin = top2.values[:, 0] - top2.values[:, 1]
            bids = prices[best_col] + margin + eps

            # Highest bid per contested column, then lowest row index wins ties.
            bid_max = torch.full(
                (n,), float("-inf"), dtype=benefit.dtype, device=device
            )
            bid_max.scatter_reduce_(0, best_col, bids, reduce="amax")
            is_winner = bids >= bid_max[best_col]
            winner_row = torch.full((n,), n, dtype=torch.long, device=device)
            winner_row.scatter_reduce_(
                0, best_col[is_winner], unassigned[is_winner], reduce="amin"
            )
            cols = (winner_row < n).nonzero(as_tuple=True)[0]
            rows = winner_row[cols]

            # Evict previous owners, assign winners, raise prices.
            prev = assign_col[cols]
            assign_row[prev[prev >= 0]] = -1
            assign_row[rows] = cols
            assign_col[cols] = rows
            prices[cols] = bid_max[cols]

        if rounds > max_rounds or eps <= eps_final:
            break
        eps = max(eps / scale_factor, eps_final)

    # Safety-valve completion: pair leftover rows/columns arbitrarily.
    if bool((assign_row < 0).any()):
        free_rows = (assign_row < 0).nonzero(as_tuple=True)[0]
        free_cols = (assign_col < 0).nonzero(as_tuple=True)[0]
        assign_row[free_rows] = free_cols

    return assign_row


def _greedy_assignment(cost: torch.Tensor) -> torch.Tensor:
    r"""Greedy nearest-free-pair assignment on a square cost matrix.

    Sorts all pairwise costs once (O(n^2 log n)) and scans them ascending,
    committing the cheapest still-free (row, col) pair until every row is
    matched. Deterministic; an approximation to the optimal assignment, far
    cheaper than the auction algorithm for large batches.

    Args:
        cost: Square cost matrix of shape (n, n).

    Returns:
        Long tensor `perm` of shape (n,) such that row `i` is assigned to
        column `perm[i]`; always a permutation.
    """
    n = cost.shape[0]
    device = cost.device
    if n == 1:
        return torch.zeros(1, dtype=torch.long, device=device)

    # Greedy nearest-free-pair is inherently sequential: the scan must run on the
    # host, so the sorted order crosses the device boundary in a single transfer.
    order = torch.argsort(cost.reshape(-1)).cpu().tolist()
    perm = [0] * n
    row_used = bytearray(n)
    col_used = bytearray(n)
    filled = 0
    for v in order:
        i, j = divmod(v, n)
        if not row_used[i] and not col_used[j]:
            perm[i] = j
            row_used[i] = col_used[j] = 1
            filled += 1
            if filled == n:
                break
    return torch.tensor(perm, dtype=torch.long, device=device)


class ExactOTCoupling(BaseCostCoupling):
    r"""
    Exact minibatch optimal-transport coupling.

    Solves the assignment problem on the squared-Euclidean cost with the
    Bertsekas auction algorithm and returns a deterministic permutation of the
    target batch (paper-faithful minibatch EMD). Round-bound: roughly 0.15 s
    (batch 128) to 0.5 s (batch 256) per call on CPU or GPU; prefer
    `SinkhornCoupling` inside training loops.

    Args:
        tol: Optimality tolerance, relative to the normalized cost scale.

    Example:
        ```python
        from torchebm.couplings import ExactOTCoupling

        coupling = ExactOTCoupling()
        x0 = torch.randn_like(x1)
        x0, x1 = coupling(x0, x1)             # transport-aligned pairs
        xt, ut = interpolant.interpolate(x0, x1, t)
        ```
    """

    def __init__(self, tol: float = 1e-4):
        self.tol = tol

    def _solve(self, cost: torch.Tensor) -> torch.Tensor:
        return _auction_assignment(cost, tol=self.tol)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tol={self.tol})"


class SinkhornCoupling(BaseCostCoupling):
    r"""
    Entropic minibatch optimal-transport coupling.

    Computes an entropic transport plan with log-domain Sinkhorn iterations,
    then draws one target index per source sample from the row-conditional
    plan. This preserves the source marginal exactly (joint sampling with
    replacement would not) and is stochastic. Milliseconds per call; the
    preferred choice inside training loops.

    Args:
        reg: Entropic regularization strength.
        n_iters: Number of Sinkhorn iterations.

    Example:
        ```python
        from torchebm.couplings import SinkhornCoupling

        coupling = SinkhornCoupling(reg=0.01)
        x0, x1 = coupling(torch.randn_like(x1), x1)
        ```
    """

    def __init__(self, reg: float = 0.05, n_iters: int = 100):
        if reg <= 0:
            raise ValueError(f"reg must be positive, got {reg}")
        if n_iters <= 0:
            raise ValueError(f"n_iters must be positive, got {n_iters}")
        self.reg = reg
        self.n_iters = n_iters

    def _solve(self, cost: torch.Tensor) -> torch.Tensor:
        plan = _sinkhorn_log(cost, reg=self.reg, n_iters=self.n_iters)
        return torch.multinomial(plan.clamp(min=0), 1).squeeze(-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reg={self.reg}, n_iters={self.n_iters})"


class UnbalancedSinkhornCoupling(BaseCostCoupling):
    r"""
    Unbalanced entropic optimal-transport coupling with per-pair weights.

    Computes a KL-marginal-relaxed transport plan (unbalanced Sinkhorn), pairs
    each source with a row-conditional draw, and returns the row masses as
    per-pair importance `weights` on the `CouplingResult` (normalized so the
    balanced case is all-ones). Sources that are expensive to transport
    (outliers, unmatched mass) receive low weight, which weight-aware
    consumers (e.g. `EnergyMatchingLoss`'s flow term) use to down-weight
    those pairs. Used by UOT-FM and single-cell trajectory settings where
    mass is not conserved.

    Args:
        reg: Entropic regularization strength \(\varepsilon\).
        reg_marginal: KL marginal-relaxation strength \(\rho\); larger values
            approach balanced Sinkhorn (uniform weights).
        n_iters: Number of Sinkhorn iterations.

    Example:
        ```python
        from torchebm.couplings import UnbalancedSinkhornCoupling

        coupling = UnbalancedSinkhornCoupling(reg=0.05, reg_marginal=1.0)
        res = coupling(torch.randn_like(x1), x1)
        x0, x1 = res
        weights = res.weights            # (batch,), outliers down-weighted
        ```
    """

    def __init__(self, reg: float = 0.05, reg_marginal: float = 1.0, n_iters: int = 100):
        if reg <= 0:
            raise ValueError(f"reg must be positive, got {reg}")
        if reg_marginal <= 0:
            raise ValueError(f"reg_marginal must be positive, got {reg_marginal}")
        if n_iters <= 0:
            raise ValueError(f"n_iters must be positive, got {n_iters}")
        self.reg = reg
        self.reg_marginal = reg_marginal
        self.n_iters = n_iters

    @torch.no_grad()
    def couple(self, x0, x1=None, **kwargs):
        x1 = self._require_x1(x1)
        self._check_batch(x0, x1)
        if x0.shape[0] == 1:
            return CouplingResult(x0, x1)
        cost = self.compute_cost(x0, x1, **kwargs)
        plan = _unbalanced_sinkhorn_log(
            cost, reg=self.reg, reg_marginal=self.reg_marginal, n_iters=self.n_iters
        )
        mass = plan.sum(dim=1)
        weights = mass / mass.mean().clamp(min=1e-12)
        idx = torch.multinomial(plan.clamp(min=1e-30), 1).squeeze(-1)
        return CouplingResult(x0, x1[idx], weights=weights)

    def _solve(self, cost: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError(
            "UnbalancedSinkhornCoupling overrides couple() to attach weights"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(reg={self.reg}, "
            f"reg_marginal={self.reg_marginal}, n_iters={self.n_iters})"
        )


class GreedyCoupling(BaseCostCoupling):
    r"""
    Greedy approximate optimal-transport coupling.

    Pairs each source with a target by repeatedly committing the globally
    cheapest still-free pair (nearest-free-pair heuristic) on the
    squared-Euclidean cost. Deterministic and O(n^2 log n): a fast,
    non-optimal approximation to the assignment problem, useful when the
    exact auction solver (~0.5 s per call at batch 256) is too slow and the
    stochasticity of `SinkhornCoupling` is unwanted.

    Example:
        ```python
        from torchebm.couplings import GreedyCoupling

        coupling = GreedyCoupling()
        x0, x1 = coupling(torch.randn_like(x1), x1)
        ```
    """

    def _solve(self, cost: torch.Tensor) -> torch.Tensor:
        return _greedy_assignment(cost)


__all__ = [
    "ExactOTCoupling",
    "SinkhornCoupling",
    "GreedyCoupling",
    "UnbalancedSinkhornCoupling",
]
