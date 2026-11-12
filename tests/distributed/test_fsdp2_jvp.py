r"""JVP through a DiT under FSDP2.

Every block and the root are sharded with `fully_shard`; ``torch.func.jvp``
runs through FSDP2's hooks and must match an unsharded per-rank replica in
the output, its tangent, and the data-parallel mean of the parameter
gradients of the output.
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


def _jvp_worker(rank, world_size, tmpdir):
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
    t = torch.rand(4).to(dist_device())
    y = torch.randint(0, 4, (4,)).to(dist_device())
    v = torch.randn_like(x)
    dt = torch.ones_like(t)

    out, out_t = torch.func.jvp(lambda x, t: model(x, t, y), (x, t), (v, dt))
    out.square().mean().backward()
    ref_out, ref_out_t = torch.func.jvp(lambda x, t: ref(x, t, y), (x, t), (v, dt))
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
            "tangent_err": (out_t - ref_out_t).abs().max().item(),
            "tangent_norm": out_t.norm().item(),
            "grad_err": grad_err,
        },
    )


def test_jvp_dit_matches_unsharded_reference():
    for res in spawn_dist(_jvp_worker):
        assert res["out_err"] < 1e-5, res
        assert res["tangent_err"] < 1e-5, res
        assert res["tangent_norm"] > 0, res
        assert res["grad_err"] < 1e-5, res
