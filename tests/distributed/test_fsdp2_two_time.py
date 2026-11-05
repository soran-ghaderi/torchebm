r"""Two-time DiT (flow map) under FSDP2.

Every block and the root are sharded with `fully_shard`; the call
``model(x_s, t, s=s)`` runs through FSDP2's hooks and must match an unsharded
per-rank replica in output and in the data-parallel mean of the parameter
gradients, including those of the interval embedder.
"""

import copy
import sys

import pytest
import torch
import torch.distributed as dist

from torchebm.models import DiT

from dist_harness import dist_device, dist_mesh, save_result, spawn_dist

fsdp = pytest.importorskip("torch.distributed.fsdp")

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable"),
    pytest.mark.skipif(
        sys.platform == "win32", reason="gloo spawn harness is POSIX-only"
    ),
    pytest.mark.skipif(not hasattr(fsdp, "fully_shard"), reason="FSDP2 unavailable"),
]


def _flow_map_worker(rank, world_size, tmpdir):
    from torch.distributed.tensor import DTensor

    torch.manual_seed(0)
    model = DiT(
        input_size=8,
        in_channels=2,
        patch_size=2,
        embed_dim=16,
        depth=2,
        num_heads=2,
        num_classes=4,
        prediction_type="flow_map",
    ).to(dist_device())
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    ref = copy.deepcopy(model)
    mesh = dist_mesh(world_size)
    for block in model.blocks:
        fsdp.fully_shard(block, mesh=mesh)
    fsdp.fully_shard(model, mesh=mesh)

    torch.manual_seed(100 + rank)
    x = torch.randn(4, 2, 8, 8).to(dist_device())
    s = torch.rand(4).to(dist_device())
    t = s + (1 - s) * torch.rand(4).to(dist_device())
    y = torch.randint(0, 4, (4,)).to(dist_device())

    out = model(x, t, y, s=s)
    out.square().mean().backward()
    ref_out = ref(x, t, y, s=s)
    ref_out.square().mean().backward()

    ref_params = dict(ref.named_parameters())
    grad_err = 0.0
    for name, p in model.named_parameters():
        rg = ref_params[name].grad.detach().clone()
        dist.all_reduce(rg)
        rg /= world_size
        sg = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
        grad_err = max(grad_err, (sg - rg).abs().max().item())
    save_result(
        tmpdir,
        rank,
        {
            "out_err": (out - ref_out).abs().max().item(),
            "grad_err": grad_err,
            "dt_grad": ref.dt_embedder.mlp[0].weight.grad.abs().sum().item(),
        },
    )


def test_flow_map_dit_matches_unsharded_reference():
    for res in spawn_dist(_flow_map_worker):
        assert res["out_err"] < 1e-5, res
        assert res["grad_err"] < 1e-5, res
        assert res["dt_grad"] > 0, res
