r"""Shim behavior single-process and under a 2-process gloo group."""

import sys

import pytest
import torch
import torch.distributed as dist

from torchebm import distributed as ebm_dist

from dist_harness import save_result, spawn_dist

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable"),
    pytest.mark.skipif(sys.platform == "win32", reason="gloo spawn harness is POSIX-only"),
]


def test_single_process_identity():
    assert not ebm_dist.is_distributed()
    assert ebm_dist.get_rank() == 0
    assert ebm_dist.get_world_size() == 1
    x = torch.randn(4, 3)
    assert ebm_dist.all_gather_cat(x) is x
    obj = {"a": 1}
    assert ebm_dist.broadcast_object(obj) is obj


def _shim_worker(rank, world_size, tmpdir):
    assert ebm_dist.is_distributed()
    assert ebm_dist.get_rank() == rank
    assert ebm_dist.get_world_size() == world_size
    gathered = ebm_dist.all_gather_cat(torch.full((2, 3), float(rank)))
    payload = ebm_dist.broadcast_object({"seed": 1234} if rank == 0 else None)
    save_result(tmpdir, rank, {"gathered": gathered, "payload": payload})


def test_shim_collectives_two_ranks():
    results = spawn_dist(_shim_worker)
    world = len(results)
    expected = torch.cat([torch.full((2, 3), float(r)) for r in range(world)])
    for res in results:
        assert torch.equal(res["gathered"], expected)
        assert res["payload"] == {"seed": 1234}
