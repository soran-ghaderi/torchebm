r"""FSDP2 facts that score-matching support relies on.

1. The hook path (`fully_shard` wrappers) cannot run the double backward that
   score matching needs: the post-backward hook reshards parameter storage the
   second-order graph still references, independent of `reshard_after_forward`.
   If a torch release lifts this, the failure test here breaks loudly and the
   functional-path requirement can be revisited.
2. The functional path (`functional_call` on an unwrapped module with the
   sharded DTensor params and a batch-sharded input) runs the double backward
   through differentiable DTensor collectives; `DenoisingScoreMatching` with
   `use_autograd=False` reproduces the gradients of an unsharded reference
   with global-batch-mean semantics.
3. The autograd path fails fast with an actionable error under DTensor params.
"""

import copy
import sys

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchebm.core import BaseModel

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


class MLPEnergy(BaseModel):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _shard(model, world_size):
    mesh = cpu_mesh(world_size)
    for m in model.net:
        if isinstance(m, nn.Linear):
            fsdp.fully_shard(m, mesh=mesh)
    fsdp.fully_shard(model.net, mesh=mesh)
    return model


def _hook_path_worker(rank, world_size, tmpdir):
    torch.manual_seed(0)
    model = _shard(MLPEnergy(), world_size)

    torch.manual_seed(100 + rank)
    x = torch.randn(BATCH, DIM).requires_grad_(True)
    energy = model(x)
    score = torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
    loss = 0.5 * score.square().sum(dim=1).mean()
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
            "The FSDP2 hook path ran a score-matching double backward; this "
            "upstream limitation appears lifted. Revisit the functional-path "
            "requirement for score matching under fully_shard."
        )


def _functional_dsm_worker(rank, world_size, tmpdir):
    from torchebm.losses import DenoisingScoreMatching

    torch.manual_seed(0)
    model = MLPEnergy()
    ref_model = copy.deepcopy(model)
    model = _shard(model, world_size)

    loss_fn = DenoisingScoreMatching(
        model=model,
        noise_scale=NOISE,
        use_autograd=False,
        functional_model=MLPEnergy(),
    )
    ref_loss_fn = DenoisingScoreMatching(model=ref_model, noise_scale=NOISE)

    torch.manual_seed(100 + rank)
    x = torch.randn(BATCH, DIM)
    torch.manual_seed(7 + rank)
    loss = loss_fn(x)
    loss.backward()
    torch.manual_seed(7 + rank)
    ref_loss = ref_loss_fn(x)
    ref_loss.backward()

    ref_grads = {n: p.grad for n, p in ref_model.named_parameters()}
    max_err = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            assert ref_grads[n] is None, f"missing sharded grad for {n}"
            continue
        full = p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad
        mean = ref_grads[n].detach().clone()
        dist.all_reduce(mean)
        mean /= world_size
        max_err = max(max_err, (full - mean).abs().max().item())
    save_result(
        tmpdir,
        rank,
        {
            "max_err": max_err,
            "loss_matches_local_ref": bool(
                torch.allclose(loss.detach(), ref_loss.detach(), atol=1e-6)
            ),
        },
    )


def test_functional_dsm_matches_unsharded_reference():
    results = spawn_dist(_functional_dsm_worker)
    for res in results:
        assert res["loss_matches_local_ref"], res
        assert res["max_err"] < 1e-5, res


def _autograd_fail_fast_worker(rank, world_size, tmpdir):
    from torchebm.losses import DenoisingScoreMatching

    torch.manual_seed(0)
    model = _shard(MLPEnergy(), world_size)
    loss_fn = DenoisingScoreMatching(model=model, noise_scale=NOISE)
    try:
        loss_fn(torch.randn(BATCH, DIM))
        msg = None
    except RuntimeError as e:
        msg = str(e)
    save_result(tmpdir, rank, {"msg": msg})


def test_autograd_path_fails_fast_with_dtensor_params():
    results = spawn_dist(_autograd_fail_fast_worker)
    for res in results:
        assert res["msg"] is not None and "use_autograd=False" in res["msg"], res
