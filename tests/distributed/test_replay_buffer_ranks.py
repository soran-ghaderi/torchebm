r"""Rank-local replay buffers and explicit cross-rank mixing."""

import sys

import pytest
import torch
import torch.distributed as dist

from torchebm.utils import ReplayBuffer

from dist_harness import dist_device, make_generator, save_result, spawn_dist

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable"),
    pytest.mark.skipif(
        sys.platform == "win32", reason="gloo spawn harness is POSIX-only"
    ),
]

CAPACITY = 16
DIM = 3


def _buffer(rank):
    return ReplayBuffer(
        CAPACITY, (DIM,), device=dist_device(), generator=make_generator(100 + rank)
    )


def _rows(t):
    return sorted(map(tuple, t.reshape(t.shape[0], -1).tolist()))


def _mix_worker(rank, world_size, tmpdir):
    buffer_a, buffer_b = _buffer(rank), _buffer(rank)
    pre = buffer_a.storage.clone()
    buffer_a.mix_across_ranks(generator=torch.Generator().manual_seed(500 + rank))
    buffer_b.mix_across_ranks(generator=torch.Generator().manual_seed(500 + 7 * rank))
    save_result(
        tmpdir,
        rank,
        {
            "pre": pre,
            "post_a": buffer_a.storage.clone(),
            "post_b": buffer_b.storage.clone(),
            "position": buffer_a.get_extra_state()["position"],
        },
    )


def test_buffers_rank_local_and_mixing_partitions_pool():
    results = spawn_dist(_mix_worker)
    pre0, pre1 = results[0]["pre"], results[1]["pre"]
    post0, post1 = results[0]["post_a"], results[1]["post_a"]

    assert not torch.equal(pre0, pre1)
    assert _rows(torch.cat([post0, post1])) == _rows(torch.cat([pre0, pre1]))

    pre0_rows = set(map(tuple, pre0.tolist()))
    from_rank0 = sum(tuple(r) in pre0_rows for r in post0.tolist())
    assert 0 < from_rank0 < CAPACITY

    assert torch.equal(results[0]["post_a"], results[0]["post_b"])
    assert torch.equal(results[1]["post_a"], results[1]["post_b"])
    assert results[0]["position"] == 0 and results[1]["position"] == 0
