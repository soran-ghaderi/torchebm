r"""Global-batch entropic OT coupling across ranks.

With a process group, `SinkhornCoupling` all_gathers both batches, every rank
solves the identical pooled problem, the row-conditional draw is broadcast
from rank 0, and each rank keeps its own rows. The pinned contract: the
2-rank global coupling equals the single-process coupling on the concatenated
batch. Assignment solvers reject a process group at construction (quadratic
pooled cost).
"""

import sys

import pytest
import torch
import torch.distributed as dist

from torchebm.couplings import (
    ExactOTCoupling,
    GreedyCoupling,
    SinkhornCoupling,
    UnbalancedSinkhornCoupling,
)

from dist_harness import save_result, spawn_dist

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable"),
    pytest.mark.skipif(sys.platform == "win32", reason="gloo spawn harness is POSIX-only"),
]

BATCH = 8
DIM = 2
DRAW_SEED = 7
REG = 0.05
N_ITERS = 200


def _rank_batches(rank):
    x0 = torch.randn(BATCH, DIM, generator=torch.Generator().manual_seed(100 + rank))
    x1 = torch.randn(BATCH, DIM, generator=torch.Generator().manual_seed(200 + rank))
    return x0, x1


def _global_worker(rank, world_size, tmpdir):
    x0, x1 = _rank_batches(rank)
    coupling = SinkhornCoupling(
        reg=REG, n_iters=N_ITERS, process_group=dist.group.WORLD
    )
    # rank-offset seeds: only rank 0's generator may influence the draw
    res = coupling(
        x0, x1, generator=torch.Generator().manual_seed(DRAW_SEED + rank)
    )
    save_result(tmpdir, rank, {"x0": res.x0, "x1": res.x1})


def test_two_rank_global_coupling_matches_single_process():
    results = spawn_dist(_global_worker)
    full0 = torch.cat([_rank_batches(0)[0], _rank_batches(1)[0]])
    full1 = torch.cat([_rank_batches(0)[1], _rank_batches(1)[1]])

    # workers run single-threaded; pin the reference solve to match bitwise
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        ref = SinkhornCoupling(reg=REG, n_iters=N_ITERS)(
            full0, full1, generator=torch.Generator().manual_seed(DRAW_SEED)
        )
    finally:
        torch.set_num_threads(prev_threads)

    assert torch.equal(torch.cat([results[0]["x0"], results[1]["x0"]]), full0)
    assert torch.equal(torch.cat([results[0]["x1"], results[1]["x1"]]), ref.x1)


def test_assignment_and_unbalanced_couplings_reject_process_group():
    for cls in (ExactOTCoupling, GreedyCoupling, UnbalancedSinkhornCoupling):
        with pytest.raises(ValueError, match="does not support process_group"):
            cls(process_group=object())


def test_single_process_with_group_degrades_to_local():
    x0, x1 = _rank_batches(0)
    grouped = SinkhornCoupling(reg=REG, n_iters=N_ITERS, process_group=object())(
        x0, x1, generator=torch.Generator().manual_seed(3)
    )
    local = SinkhornCoupling(reg=REG, n_iters=N_ITERS)(
        x0, x1, generator=torch.Generator().manual_seed(3)
    )
    assert torch.equal(grouped.x1, local.x1)


def test_conditioning_kwargs_rejected_on_global_path():
    coupling = SinkhornCoupling(process_group=object())
    x0, x1 = _rank_batches(0)
    with pytest.raises(NotImplementedError, match="Conditioning"):
        coupling(x0, x1, labels=torch.zeros(BATCH))
