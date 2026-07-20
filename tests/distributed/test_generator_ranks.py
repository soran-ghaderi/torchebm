r"""Per-rank RNG control with explicit generators.

The distributed contract for randomness: the library holds no hidden RNG state,
so ranks decorrelate exactly when the caller seeds them differently. A shared
seed must reproduce identical chains on every rank; a rank-offset seed must not.
"""

import sys

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchebm.core import BaseModel
from torchebm.samplers import LangevinDynamics

from dist_harness import save_result, spawn_dist

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable"),
    pytest.mark.skipif(sys.platform == "win32", reason="gloo spawn harness is POSIX-only"),
]

BASE_SEED = 1234


class QuadraticEnergy(BaseModel):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x):
        return self.scale * (x**2).sum(dim=-1)


def _worker(rank, world_size, tmpdir):
    sampler = LangevinDynamics(model=QuadraticEnergy(), step_size=1e-2)
    kwargs = dict(dim=2, n_samples=4, n_steps=5)
    shared = sampler.sample(
        **kwargs, generator=torch.Generator().manual_seed(BASE_SEED)
    )
    per_rank = sampler.sample(
        **kwargs, generator=torch.Generator().manual_seed(BASE_SEED + rank)
    )
    save_result(tmpdir, rank, {"shared": shared, "per_rank": per_rank})


def test_shared_seed_matches_and_rank_offset_decorrelates():
    results = spawn_dist(_worker)
    assert torch.equal(results[0]["shared"], results[1]["shared"])
    assert not torch.equal(results[0]["per_rank"], results[1]["per_rank"])
