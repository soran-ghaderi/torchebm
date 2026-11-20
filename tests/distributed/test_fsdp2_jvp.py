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


BLOCK_OPTIONS = {
    "dit": {},
    "lightningdit": dict(qk_norm=True, swiglu=True, rope="2d", rms_norm=True),
    "in_context": dict(
        conditioning="in_context",
        pos_embed=None,
        qk_norm=True,
        swiglu=True,
        rope="1d",
        rms_norm=True,
        block_bias=False,
    ),
}


def _jvp_worker(rank, world_size, tmpdir, options):
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
        **options,
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


@pytest.mark.parametrize("options", BLOCK_OPTIONS.values(), ids=BLOCK_OPTIONS.keys())
def test_jvp_dit_matches_unsharded_reference(options):
    for res in spawn_dist(_jvp_worker, args=(options,)):
        assert res["out_err"] < 1e-5, res
        assert res["tangent_err"] < 1e-5, res
        assert res["tangent_norm"] > 0, res
        assert res["grad_err"] < 1e-5, res


def _average_velocity_jvp_worker(rank, world_size, tmpdir, guidance_conditioning, aux_head):
    from torch.distributed.tensor import DTensor

    from torchebm.losses import MeanFlowLoss
    from torchebm.utils import average_velocity_jvp, sample_guidance

    torch.manual_seed(0)
    options = dict(BLOCK_OPTIONS["in_context"], time_invariant=True, aux_head_depth=1) if aux_head else {}
    model = DiT(
        input_size=8,
        in_channels=2,
        patch_size=2,
        embed_dim=16,
        depth=2,
        num_heads=2,
        num_classes=4,
        prediction_type="average_velocity",
        guidance_conditioning=guidance_conditioning,
        **options,
    ).to(dist_device())
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    ref = copy.deepcopy(model)
    mesh = dist_mesh(world_size)
    for block in (*model.blocks, *model.aux_blocks):
        fsdp.fully_shard(block, mesh=mesh)
    fsdp.fully_shard(model, mesh=mesh)

    torch.manual_seed(100 + rank)
    x = torch.randn(4, 2, 8, 8).to(dist_device())
    s = torch.rand(4).to(dist_device())
    t = torch.cat([s[:2], s[2:] + (1 - s[2:]) * torch.rand(2).to(dist_device())])
    y = torch.randint(0, 4, (4,)).to(dist_device())
    v = torch.randn_like(x)
    loss_fn = MeanFlowLoss()
    model_kwargs = {}
    if guidance_conditioning is not None:
        scale, interval = sample_guidance(4, flow_matching_ratio=0.5, device=dist_device())
        model_kwargs = {"guidance_scale": scale, "guidance_interval": interval}

    u, du_ds, *aux = average_velocity_jvp(
        model, x, s, t, v, y, flow_matching_ratio=0.5, model_kwargs=model_kwargs, aux_head=aux_head
    )
    loss_fn(u, du_ds, v, s, t, aux_input=aux[0] if aux_head else None).backward()
    ref_u, ref_du_ds, *ref_aux = average_velocity_jvp(
        ref, x, s, t, v, y, flow_matching_ratio=0.5, model_kwargs=model_kwargs, aux_head=aux_head
    )
    loss_fn(ref_u, ref_du_ds, v, s, t, aux_input=ref_aux[0] if aux_head else None).backward()

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
            "out_err": max((a - b).abs().max().item() for a, b in zip([u, *aux], [ref_u, *ref_aux])),
            "tangent_err": (du_ds - ref_du_ds).abs().max().item(),
            "tangent_norm": du_ds[2:].norm().item(),
            "grad_err": grad_err,
        },
    )


@pytest.mark.parametrize(
    "guidance_conditioning, aux_head",
    [(None, False), ("scale_and_interval", False), ("scale_and_interval", True)],
)
def test_average_velocity_jvp_matches_unsharded_reference(guidance_conditioning, aux_head):
    for res in spawn_dist(_average_velocity_jvp_worker, args=(guidance_conditioning, aux_head)):
        assert res["out_err"] < 1e-5, res
        assert res["tangent_err"] < 1e-5, res
        assert res["tangent_norm"] > 0, res
        assert res["grad_err"] < 1e-5, res
