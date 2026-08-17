r"""First-order paths of an energy model whose inner net is sharded with FSDP2.

The shard-inside pattern keeps `BaseModel` the user-facing type and applies
`fully_shard` to the network it owns, so samplers and losses stay unaware of
sharding. Every first-order path (`gradient`, and the MCMC chains built on it)
runs through FSDP2's hooks unchanged; only score matching's double backward
needs the functional path.

Each rank holds a full replica of the reference model and its own local batch,
so the sharded result must match its own-rank unsharded reference exactly.
"""

import copy
import sys

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchebm.core import BaseModel
from torchebm.distributed import unsharded
from torchebm.samplers import HamiltonianMonteCarlo, LangevinDynamics

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

DIM = 8
BATCH = 8
STEPS = 5


class MLPEnergy(BaseModel):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _shard(model, world_size, **kwargs):
    mesh = cpu_mesh(world_size)
    for m in model.net:
        if isinstance(m, nn.Linear):
            fsdp.fully_shard(m, mesh=mesh, **kwargs)
    fsdp.fully_shard(model.net, mesh=mesh, **kwargs)
    return model


def _pair(rank, world_size, **kwargs):
    r"""A sharded model and an unsharded replica of the same initialization."""
    torch.manual_seed(0)
    model = MLPEnergy()
    ref = copy.deepcopy(model)
    return _shard(model, world_size, **kwargs), ref


def _local_batch(rank):
    torch.manual_seed(100 + rank)
    return torch.randn(BATCH, DIM)


def _gradient_worker(rank, world_size, tmpdir):
    model, ref = _pair(rank, world_size)
    x = _local_batch(rank)
    err = (model.gradient(x) - ref.gradient(x)).abs().max().item()
    save_result(tmpdir, rank, {"err": err})


def test_gradient_matches_unsharded_reference():
    for res in spawn_dist(_gradient_worker):
        assert res["err"] < 1e-6, res


def _sampler_worker(rank, world_size, tmpdir):
    model, ref = _pair(rank, world_size)
    x = _local_batch(rank)
    errs = {}
    for name, build in (
        ("langevin", lambda m: LangevinDynamics(model=m, step_size=0.01)),
        (
            "hmc",
            lambda m: HamiltonianMonteCarlo(
                model=m, step_size=0.01, n_leapfrog_steps=3
            ),
        ),
    ):
        g = torch.Generator().manual_seed(1234)
        out = build(model).sample(x=x, n_steps=STEPS, generator=g)
        g = torch.Generator().manual_seed(1234)
        ref_out = build(ref).sample(x=x, n_steps=STEPS, generator=g)
        errs[name] = (out - ref_out).abs().max().item()
    save_result(tmpdir, rank, errs)


def test_seeded_samplers_match_unsharded_reference():
    for res in spawn_dist(_sampler_worker):
        assert res["langevin"] < 1e-6, res
        assert res["hmc"] < 1e-6, res


def _unsharded_context_worker(rank, world_size, tmpdir):
    model, ref = _pair(rank, world_size)
    x = _local_batch(rank)
    sampler = LangevinDynamics(model=model, step_size=0.01)

    g = torch.Generator().manual_seed(1234)
    with unsharded(model.net):
        out = sampler.sample(x=x, n_steps=STEPS, generator=g)

    g = torch.Generator().manual_seed(1234)
    ref_out = LangevinDynamics(model=ref, step_size=0.01).sample(
        x=x, n_steps=STEPS, generator=g
    )

    g = torch.Generator().manual_seed(1234)
    after = sampler.sample(x=x, n_steps=STEPS, generator=g)

    save_result(
        tmpdir,
        rank,
        {
            "err": (out - ref_out).abs().max().item(),
            "err_after": (after - ref_out).abs().max().item(),
        },
    )


def test_unsharded_context_preserves_sampling_results():
    for res in spawn_dist(_unsharded_context_worker):
        assert res["err"] < 1e-6, res
        assert res["err_after"] < 1e-6, res


def _device_dtype_worker(rank, world_size, tmpdir):
    model, _ = _pair(rank, world_size)
    save_result(
        tmpdir,
        rank,
        {
            "is_dtensor": isinstance(
                next(model.parameters()), torch.distributed.tensor.DTensor
            ),
            "device": str(model.device),
            "dtype": str(model.dtype),
        },
    )


def test_device_and_dtype_resolve_through_dtensor_parameters():
    for res in spawn_dist(_device_dtype_worker):
        assert res["is_dtensor"], res
        assert res["device"] == "cpu", res
        assert res["dtype"] == "torch.float32", res


def _meta_init_worker(rank, world_size, tmpdir):
    with torch.device("meta"):
        model = MLPEnergy()
    model = _shard(model, world_size)
    model.to_empty(device="cpu")
    torch.manual_seed(0)
    for m in model.net:
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    x = _local_batch(rank)
    grad = model.gradient(x)
    save_result(
        tmpdir,
        rank,
        {"finite": bool(torch.isfinite(grad).all()), "shape": tuple(grad.shape)},
    )


def test_meta_device_init_then_materialization():
    for res in spawn_dist(_meta_init_worker):
        assert res["finite"], res
        assert res["shape"] == (BATCH, DIM), res


def _bf16_worker(rank, world_size, tmpdir):
    policy = fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    model, _ = _pair(rank, world_size, mp_policy=policy)
    x = _local_batch(rank)

    grad = model.gradient(x)
    model.force_fp32_gradient = True
    grad_fp32 = model.gradient(x)

    save_result(
        tmpdir,
        rank,
        {
            "dtype": str(grad.dtype),
            "dtype_fp32": str(grad_fp32.dtype),
            "err": (grad - grad_fp32).abs().max().item(),
        },
    )


def test_bf16_policy_and_force_fp32_gradient():
    for res in spawn_dist(_bf16_worker):
        assert res["dtype"] == "torch.float32", res
        assert res["dtype_fp32"] == "torch.float32", res
        assert res["err"] < 1e-1, res


def _ddp_worker(rank, world_size, tmpdir):
    torch.manual_seed(0)
    model = MLPEnergy()
    ref = copy.deepcopy(model)
    model.net = nn.parallel.DistributedDataParallel(model.net)

    x = _local_batch(rank)
    err = (model.gradient(x) - ref.gradient(x)).abs().max().item()
    save_result(tmpdir, rank, {"err": err})


def test_ddp_inner_module_also_runs():
    for res in spawn_dist(_ddp_worker):
        assert res["err"] < 1e-6, res
