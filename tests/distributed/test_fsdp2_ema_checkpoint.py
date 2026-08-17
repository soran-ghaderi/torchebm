r"""EMA updates and checkpointing with sharded parameters.

`update_ema` on identically sharded models updates each local DTensor shard
in place with no collective; after N optimizer steps the sharded EMA must
match an unsharded reference EMA. Checkpointing follows the split contract:
sharded model parameters round-trip through `torch.distributed.checkpoint`
(`get_model_state_dict`/`set_model_state_dict`), while rank-local loss state
(the PCD replay buffer) rides in one plain file per rank holding only the
buffer entries, restored with `strict=False`.
"""

import copy
import os
import sys

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchebm.core import BaseModel
from torchebm.losses import ContrastiveDivergence
from torchebm.samplers import LangevinDynamics
from torchebm.utils.training import update_ema

from dist_harness import cpu_mesh, save_result, spawn_dist

fsdp = pytest.importorskip("torch.distributed.fsdp")

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable"),
    pytest.mark.skipif(
        sys.platform == "win32", reason="gloo spawn harness is POSIX-only"
    ),
    pytest.mark.skipif(not hasattr(fsdp, "fully_shard"), reason="FSDP2 unavailable"),
]

DIM = 4
BATCH = 8
BUFFER = 16


class MLPEnergy(BaseModel):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, 16), nn.SiLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _shard_net(model, world_size):
    mesh = cpu_mesh(world_size)
    for m in model.net:
        if isinstance(m, nn.Linear):
            fsdp.fully_shard(m, mesh=mesh)
    fsdp.fully_shard(model.net, mesh=mesh)
    return model


def _full(param):
    from torch.distributed.tensor import DTensor

    return param.full_tensor() if isinstance(param, DTensor) else param


def _ema_worker(rank, world_size, tmpdir):
    torch.manual_seed(0)
    model = MLPEnergy()
    ema = copy.deepcopy(model)
    model_ref = copy.deepcopy(model)
    ema_ref = copy.deepcopy(model)
    _shard_net(model, world_size)
    _shard_net(ema, world_size)

    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    opt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.05)
    for step in range(4):
        # identical batch on every rank: sharded (averaged) grads == ref grads
        x = torch.randn(BATCH, DIM, generator=torch.Generator().manual_seed(50 + step))
        for m, o in ((model, opt), (model_ref, opt_ref)):
            o.zero_grad()
            m(x).sum().backward()
            o.step()
        update_ema(ema, model, decay=0.9)
        update_ema(ema_ref, model_ref, decay=0.9)

    ref_params = dict(ema_ref.named_parameters())
    err = max(
        (_full(p) - ref_params[n]).abs().max().item()
        for n, p in ema.named_parameters()
    )
    save_result(tmpdir, rank, {"err": err})


def test_ema_on_identically_sharded_models_matches_unsharded():
    for res in spawn_dist(_ema_worker):
        assert res["err"] < 1e-6, res


def _dcp_worker(rank, world_size, tmpdir):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        set_model_state_dict,
    )

    torch.manual_seed(0)
    model = _shard_net(MLPEnergy(), world_size)
    loss = ContrastiveDivergence(
        model=model,
        sampler=LangevinDynamics(model=model, step_size=0.01),
        persistent=True,
        buffer_size=BUFFER,
        init_steps=0,
    )
    loss.initialize_buffer((DIM,), generator=torch.Generator().manual_seed(100 + rank))
    loss.update_buffer(torch.randn(4, DIM))

    saved_params = {n: _full(p).clone() for n, p in model.named_parameters()}
    saved_buffer = loss.replay_buffer.clone()

    ckpt = os.path.join(tmpdir, "ckpt")
    dcp.save({"model": get_model_state_dict(model)}, checkpoint_id=ckpt)
    torch.save(
        {"replay_buffer": loss.replay_buffer, "buffer_ptr": loss.buffer_ptr},
        os.path.join(tmpdir, f"loss_rank{rank}.pt"),
    )

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    loss.replay_buffer.zero_()
    loss.buffer_ptr.fill_(0)
    loss._buffer_ptr_int = 0

    state = {"model": get_model_state_dict(model)}
    dcp.load(state, checkpoint_id=ckpt)
    set_model_state_dict(model, state["model"])
    loss.load_state_dict(
        torch.load(os.path.join(tmpdir, f"loss_rank{rank}.pt")), strict=False
    )

    err = max(
        (_full(p) - saved_params[n]).abs().max().item()
        for n, p in model.named_parameters()
    )
    save_result(
        tmpdir,
        rank,
        {
            "param_err": err,
            "buffer_restored": torch.equal(loss.replay_buffer, saved_buffer),
            "ptr": loss._buffer_ptr_int,
            "buffer_sum": loss.replay_buffer.sum().item(),
        },
    )


def test_dcp_model_roundtrip_with_rank_local_loss_state():
    results = spawn_dist(_dcp_worker)
    for res in results:
        assert res["param_err"] < 1e-7, res
        assert res["buffer_restored"], res
        assert res["ptr"] == 4, res
    # buffers stay rank-local through the round-trip
    assert results[0]["buffer_sum"] != results[1]["buffer_sum"]
