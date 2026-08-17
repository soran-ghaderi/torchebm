r"""Loss-family audit under FSDP2 (shard-inside pattern).

First-order objectives run through FSDP2 hooks unchanged and must match an
unsharded per-rank replica: contrastive divergence (k no-grad sampler
forwards, two grad-tracked forwards, one backward) and implicit EqM. Their
parameter gradients follow the data-parallel convention: the sharded gradient
equals the cross-rank mean of the per-rank reference gradients.

Objectives that backpropagate through an input-gradient built with
`create_graph=True` (the Energy Matching flow term, explicit EqM energies)
cannot run under FSDP2 hooks (post-backward resharding frees storage the
second-order backward references) and must fail fast with an actionable
error in training mode; their eval-mode paths are first-order and must still
match the reference.
"""

import copy
import sys

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchebm.core import BaseModel
from torchebm.losses import (
    ContrastiveDivergence,
    EnergyMatchingLoss,
    EquilibriumMatchingLoss,
)
from torchebm.samplers import LangevinDynamics

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


class MLPEnergy(BaseModel):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, 16), nn.SiLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FieldNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM + 1, 16), nn.SiLU(), nn.Linear(16, DIM)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t.reshape(-1, 1)], dim=-1))


def _shard_net(model, world_size):
    mesh = cpu_mesh(world_size)
    for m in model.net:
        if isinstance(m, nn.Linear):
            fsdp.fully_shard(m, mesh=mesh)
    fsdp.fully_shard(model.net, mesh=mesh)
    return model


def _pair(cls, world_size):
    torch.manual_seed(0)
    model = cls()
    ref = copy.deepcopy(model)
    return _shard_net(model, world_size), ref


def _local_batch(rank):
    torch.manual_seed(100 + rank)
    return torch.randn(BATCH, DIM)


def _grad_err(model, ref, world_size):
    r"""Max |sharded grad - cross-rank mean of reference grads| over params."""
    from torch.distributed.tensor import DTensor

    ref_params = dict(ref.named_parameters())
    err = 0.0
    for name, p in model.named_parameters():
        rg = ref_params[name].grad.detach().clone()
        dist.all_reduce(rg)
        rg /= world_size
        sg = p.grad
        if isinstance(sg, DTensor):
            sg = sg.full_tensor()
        err = max(err, (sg - rg).abs().max().item())
    return err


def _cd_loss(model):
    return ContrastiveDivergence(
        model=model,
        sampler=LangevinDynamics(model=model, step_size=0.01),
        k_steps=3,
        persistent=False,
    )


def _cd_worker(rank, world_size, tmpdir):
    model, ref = _pair(MLPEnergy, world_size)
    x = _local_batch(rank)
    loss, _ = _cd_loss(model)(x, generator=torch.Generator().manual_seed(9))
    loss.backward()
    ref_loss, _ = _cd_loss(ref)(x, generator=torch.Generator().manual_seed(9))
    ref_loss.backward()
    save_result(
        tmpdir,
        rank,
        {
            "loss_err": abs(loss.item() - ref_loss.item()),
            "grad_err": _grad_err(model, ref, world_size),
        },
    )


def test_contrastive_divergence_matches_unsharded_reference():
    for res in spawn_dist(_cd_worker):
        assert res["loss_err"] < 1e-6, res
        assert res["grad_err"] < 1e-5, res


def _eqm_worker(rank, world_size, tmpdir):
    model, ref = _pair(FieldNet, world_size)
    x = _local_batch(rank)
    loss = EquilibriumMatchingLoss(model=model)(
        x, generator=torch.Generator().manual_seed(9)
    )
    loss.backward()
    ref_loss = EquilibriumMatchingLoss(model=ref)(
        x, generator=torch.Generator().manual_seed(9)
    )
    ref_loss.backward()
    save_result(
        tmpdir,
        rank,
        {
            "loss_err": abs(loss.item() - ref_loss.item()),
            "grad_err": _grad_err(model, ref, world_size),
        },
    )


def test_implicit_eqm_matches_unsharded_reference():
    for res in spawn_dist(_eqm_worker):
        assert res["loss_err"] < 1e-6, res
        assert res["grad_err"] < 1e-5, res


def _em_worker(rank, world_size, tmpdir):
    model, ref = _pair(MLPEnergy, world_size)
    x = _local_batch(rank)
    out = {}

    em = EnergyMatchingLoss(model=model, lambda_cd=0.0, n_langevin_steps=2)
    try:
        em(x, generator=torch.Generator().manual_seed(5))
        out["train_error"] = ""
    except RuntimeError as e:
        out["train_error"] = str(e)

    model.eval()
    ref.eval()
    em_ref = EnergyMatchingLoss(model=ref, lambda_cd=0.0, n_langevin_steps=2)
    val = em(x, generator=torch.Generator().manual_seed(5))
    ref_val = em_ref(x, generator=torch.Generator().manual_seed(5))
    out["eval_err"] = abs(val.item() - ref_val.item())
    save_result(tmpdir, rank, out)


def test_energy_matching_fails_fast_in_training_and_matches_in_eval():
    for res in spawn_dist(_em_worker):
        assert "DTensor" in res["train_error"], res
        assert res["eval_err"] < 1e-6, res


def _eqm_explicit_worker(rank, world_size, tmpdir):
    model, ref = _pair(FieldNet, world_size)
    x = _local_batch(rank)
    out = {}

    eqm = EquilibriumMatchingLoss(model=model, energy_type="dot")
    try:
        eqm(x, generator=torch.Generator().manual_seed(5))
        out["train_error"] = ""
    except RuntimeError as e:
        out["train_error"] = str(e)

    model.eval()
    ref.eval()
    eqm_ref = EquilibriumMatchingLoss(model=ref, energy_type="dot")
    val = eqm(x, generator=torch.Generator().manual_seed(5))
    ref_val = eqm_ref(x, generator=torch.Generator().manual_seed(5))
    out["eval_err"] = abs(val.item() - ref_val.item())
    save_result(tmpdir, rank, out)


def test_explicit_eqm_fails_fast_in_training_and_matches_in_eval():
    for res in spawn_dist(_eqm_explicit_worker):
        assert "DTensor" in res["train_error"], res
        assert res["eval_err"] < 1e-6, res
