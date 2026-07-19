r"""Pins the FSDP2 facts the distributed score-matching contract is built on.

1. The hook path (`fully_shard` wrappers) cannot run the double backward that
   score matching needs: the post-backward hook reshards parameter storage the
   second-order graph still references, independent of `reshard_after_forward`.
   If torch ever lifts this, the failure test breaks loudly and the
   functional-path requirement can be revisited.
2. The functional path (`functional_call` on a hook-free skeleton with the
   sharded DTensor params and a batch-sharded input) runs the double backward
   through differentiable DTensor collectives and reproduces the gradients of
   an unsharded reference with global-batch-mean semantics.
"""

import copy
import sys

import pytest
import torch
import torch.distributed as dist
from torch import nn

from dist_harness import cpu_mesh, save_result, spawn_dist

fsdp = pytest.importorskip("torch.distributed.fsdp")

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable"),
    pytest.mark.skipif(sys.platform == "win32", reason="gloo spawn harness is POSIX-only"),
    pytest.mark.skipif(not hasattr(fsdp, "fully_shard"), reason="FSDP2 unavailable"),
]

DIM = 8
BATCH = 8
NOISE = 0.1


def _make_net():
    return nn.Sequential(
        nn.Linear(DIM, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 1)
    )


def _hook_path_worker(rank, world_size, tmpdir):
    from torchebm.core import BaseModel
    from torchebm.losses import DenoisingScoreMatching

    class MLPEnergy(BaseModel):
        def __init__(self):
            super().__init__()
            self.net = _make_net()

        def forward(self, x):
            return self.net(x).squeeze(-1)

    torch.manual_seed(0)
    model = MLPEnergy()
    mesh = cpu_mesh(world_size)
    for m in model.net:
        if isinstance(m, nn.Linear):
            fsdp.fully_shard(m, mesh=mesh)
    fsdp.fully_shard(model.net, mesh=mesh)

    torch.manual_seed(100 + rank)
    x = torch.randn(BATCH, DIM)
    loss = DenoisingScoreMatching(model=model, noise_scale=NOISE)(x)
    try:
        loss.backward()
        raised = None
    except RuntimeError as e:
        raised = str(e)
    save_result(tmpdir, rank, {"raised": raised})


def test_hook_path_double_backward_still_unsupported():
    results = spawn_dist(_hook_path_worker)
    for res in results:
        assert res["raised"] is not None, (
            "The FSDP2 hook path ran a score-matching double backward. The "
            "upstream limitation this suite is designed around has been lifted;"
            " revisit the functional-path requirement."
        )


def _functional_worker(rank, world_size, tmpdir):
    from torch.distributed.tensor import DTensor, Shard
    from torch.func import functional_call

    torch.manual_seed(0)
    net = _make_net()
    ref = copy.deepcopy(net)
    mesh = cpu_mesh(world_size)
    fsdp.fully_shard(net, mesh=mesh)
    params = dict(net.named_parameters())

    torch.manual_seed(100 + rank)
    x = torch.randn(BATCH, DIM)
    noise = torch.randn_like(x) * NOISE
    x_pert = (x + noise).detach()
    target = -noise / (NOISE**2)

    x_dt = DTensor.from_local(x_pert, mesh, [Shard(0)], run_check=False)
    x_dt.requires_grad_(True)
    energy = functional_call(ref, params, (x_dt,)).squeeze(-1)
    score = torch.autograd.grad(energy.sum(), x_dt, create_graph=True)[0]
    target_dt = DTensor.from_local(target, mesh, [Shard(0)], run_check=False)
    loss = 0.5 * (score - target_dt).square().sum(dim=1).mean()
    loss.backward()
    loss_val = loss.detach()
    if isinstance(loss_val, DTensor):
        loss_val = loss_val.full_tensor()

    x_ref = x_pert.clone().requires_grad_(True)
    e_ref = ref(x_ref).squeeze(-1)
    score_ref = torch.autograd.grad(e_ref.sum(), x_ref, create_graph=True)[0]
    loss_ref = 0.5 * (score_ref - target).square().sum(dim=1).mean()
    loss_ref.backward()

    ref_grads = {n: p.grad for n, p in ref.named_parameters()}
    max_err = 0.0
    for n, p in params.items():
        if p.grad is None:
            # score-independent params (a final-layer bias is a constant
            # energy offset) legitimately get no grad; the reference agrees
            assert ref_grads[n] is None, f"missing sharded grad for {n}"
            continue
        full = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
        mean = ref_grads[n].detach().clone()
        dist.all_reduce(mean)
        mean /= world_size
        max_err = max(max_err, (full - mean).abs().max().item())
    save_result(tmpdir, rank, {"max_err": max_err, "loss": float(loss_val)})


def test_functional_score_path_matches_reference():
    results = spawn_dist(_functional_worker)
    losses = {round(r["loss"], 6) for r in results}
    assert len(losses) == 1, f"global loss differs across ranks: {losses}"
    for res in results:
        assert res["max_err"] < 1e-5, res
