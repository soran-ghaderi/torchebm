r"""Per-rank PCD replay buffers and explicit cross-rank mixing.

Replay buffers are rank-local by design: independent chains per rank multiply
chain diversity by the world size, and no default path issues a collective.
`mix_buffer_across_ranks` re-deals the pooled chains with one shared
permutation broadcast from rank 0, so rank-offset generators cannot
desynchronize the shuffle, and no chain is duplicated or lost.
"""

import sys

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchebm.core import BaseModel
from torchebm.losses import ContrastiveDivergence
from torchebm.samplers import LangevinDynamics

from dist_harness import dist_device, make_generator, save_result, spawn_dist

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable"),
    pytest.mark.skipif(sys.platform == "win32", reason="gloo spawn harness is POSIX-only"),
]

BUFFER = 16
DIM = 3


class QuadraticEnergy(BaseModel):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x):
        return self.scale * (x**2).sum(dim=-1)


def _make_loss():
    model = QuadraticEnergy().to(dist_device())
    return ContrastiveDivergence(
        model=model,
        sampler=LangevinDynamics(model=model, step_size=1e-2),
        k_steps=1,
        persistent=True,
        buffer_size=BUFFER,
        init_steps=0,
        new_sample_ratio=0.0,
        device=dist_device(),
    )


def _rows(t: torch.Tensor):
    return sorted(map(tuple, t.reshape(t.shape[0], -1).tolist()))


def _mix_worker(rank, world_size, tmpdir):
    loss_a = _make_loss()
    loss_b = _make_loss()
    loss_a.initialize_buffer((DIM,), generator=make_generator(100 + rank))
    loss_b.initialize_buffer((DIM,), generator=make_generator(100 + rank))
    pre = loss_a.replay_buffer.clone()
    # rank-offset seeds: only rank 0's generator may influence the permutation
    loss_a.mix_buffer_across_ranks(generator=torch.Generator().manual_seed(500 + rank))
    loss_b.mix_buffer_across_ranks(
        generator=torch.Generator().manual_seed(500 + 7 * rank)
    )
    save_result(
        tmpdir,
        rank,
        {
            "pre": pre,
            "post_a": loss_a.replay_buffer.clone(),
            "post_b": loss_b.replay_buffer.clone(),
            "ptr": loss_a._buffer_ptr_int,
        },
    )


def test_buffers_rank_local_and_mixing_partitions_pool():
    results = spawn_dist(_mix_worker)
    pre0, pre1 = results[0]["pre"], results[1]["pre"]
    post0, post1 = results[0]["post_a"], results[1]["post_a"]

    # rank-local by default: independently seeded buffers differ
    assert not torch.equal(pre0, pre1)

    # exact partition: the pooled rows are re-dealt, none duplicated or lost
    assert _rows(torch.cat([post0, post1])) == _rows(torch.cat([pre0, pre1]))

    # each rank's shard draws from both source ranks
    pre0_rows = set(map(tuple, pre0.tolist()))
    from_rank0 = sum(tuple(r) in pre0_rows for r in post0.tolist())
    assert 0 < from_rank0 < BUFFER

    # rank 0's seed alone fixes the permutation: identical pre-state mixed
    # under different non-rank-0 seeds lands identically
    assert torch.equal(results[0]["post_a"], results[0]["post_b"])
    assert torch.equal(results[1]["post_a"], results[1]["post_b"])

    # FIFO pointer is untouched
    assert results[0]["ptr"] == 0 and results[1]["ptr"] == 0


def test_mix_requires_persistent():
    model = QuadraticEnergy()
    loss = ContrastiveDivergence(
        model=model,
        sampler=LangevinDynamics(model=model, step_size=1e-2),
        persistent=False,
    )
    with pytest.raises(RuntimeError, match="persistent"):
        loss.mix_buffer_across_ranks()


def test_mix_requires_initialized_buffer():
    with pytest.raises(RuntimeError, match="not initialized"):
        _make_loss().mix_buffer_across_ranks()


def test_mix_is_noop_single_process():
    loss = _make_loss()
    loss.initialize_buffer((DIM,), generator=make_generator(0))
    before = loss.replay_buffer.clone()
    loss.mix_buffer_across_ranks()
    assert torch.equal(loss.replay_buffer, before)


def test_buffer_checkpoint_roundtrip_requires_init():
    loss = _make_loss()
    loss.initialize_buffer((DIM,), generator=make_generator(1))
    loss.update_buffer(torch.randn(4, DIM))
    state = loss.state_dict()

    fresh = _make_loss()
    with pytest.raises(RuntimeError, match="replay_buffer"):
        fresh.load_state_dict(state)

    fresh.initialize_buffer((DIM,), generator=make_generator(2))
    fresh.load_state_dict(state)
    assert torch.equal(fresh.replay_buffer, loss.replay_buffer)
    assert fresh._buffer_ptr_int == 4
