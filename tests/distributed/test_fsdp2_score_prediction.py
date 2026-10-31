r"""`BaseModel(prediction_type="score")` with its inner net sharded by FSDP2."""

import copy
import sys

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchebm.core import BaseModel

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

DIM = 8
BATCH = 8


class MLPEnergy(BaseModel):
    def __init__(self, output_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(DIM, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.Linear(16, 1, bias=output_bias),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _replicas(output_bias=True, prediction_type="score", functional_model=None):
    torch.manual_seed(0)
    model = MLPEnergy(
        output_bias,
        prediction_type=prediction_type,
        functional_model=functional_model,
    ).to(dist_device())
    return model, copy.deepcopy(model)


def _pair(world_size, **kwargs):
    model, ref = _replicas(**kwargs)
    mesh = dist_mesh(world_size)
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            fsdp.fully_shard(layer, mesh=mesh)
    fsdp.fully_shard(model.net, mesh=mesh)
    return model, ref


def _local_batch(rank):
    torch.manual_seed(100 + rank)
    return torch.randn(BATCH, DIM).to(dist_device())


def _no_grad_worker(rank, world_size, tmpdir):
    model, ref = _pair(world_size)
    x = _local_batch(rank)
    with torch.no_grad():
        err = (model(x) - ref(x)).abs().max().item()
    save_result(tmpdir, rank, {"err": err})


def test_score_under_no_grad_matches_unsharded_reference():
    for res in spawn_dist(_no_grad_worker):
        assert res["err"] < 1e-6, res


def _loss(output):
    if isinstance(output, tuple):
        energy, score = output
        return score.square().sum(dim=1).mean() + energy.mean()
    return output.square().sum(dim=1).mean()


def _fsdp_training_worker(rank, world_size, tmpdir, prediction_type):
    with torch.device("meta"):
        template = MLPEnergy()
    model, ref = _pair(
        world_size, prediction_type=prediction_type, functional_model=template
    )
    x = _local_batch(rank)
    max_err = 0.0
    missing = []
    for _ in range(2):
        model.zero_grad()
        ref.zero_grad()
        _loss(model(x)).backward()
        _loss(ref(x)).backward()
        for (name, param), ref_param in zip(model.named_parameters(), ref.parameters()):
            if ref_param.grad is None:
                missing.append(name if param.grad is not None else None)
                continue
            ref_grad = ref_param.grad.clone()
            dist.all_reduce(ref_grad)
            ref_grad /= world_size
            max_err = max(
                max_err, (param.grad.full_tensor() - ref_grad).abs().max().item()
            )
    placements_match = all(
        tuple(p.grad.placements) == tuple(p.placements)
        for p in model.parameters()
        if p.grad is not None
    )
    torch.optim.Adam(model.parameters(), lr=1e-3).step()
    stepped_finite = all(
        bool(torch.isfinite(p.full_tensor()).all()) for p in model.parameters()
    )
    save_result(
        tmpdir,
        rank,
        {
            "max_err": max_err,
            "unexpected_grads": [n for n in missing if n is not None],
            "placements_match": placements_match,
            "stepped_finite": stepped_finite,
        },
    )


@pytest.mark.parametrize("prediction_type", ["score", ("energy", "score")])
def test_training_through_score_matches_unsharded_reference(prediction_type):
    for res in spawn_dist(_fsdp_training_worker, args=(prediction_type,)):
        assert res["max_err"] < 1e-5, res
        assert not res["unexpected_grads"], res
        assert res["placements_match"], res
        assert res["stepped_finite"], res


def _missing_template_worker(rank, world_size, tmpdir):
    model, _ = _pair(world_size)
    try:
        model(_local_batch(rank))
        message = None
    except RuntimeError as error:
        message = str(error)
    save_result(tmpdir, rank, {"message": message})


def test_training_through_score_without_functional_model_raises():
    for res in spawn_dist(_missing_template_worker):
        assert res["message"] is not None and "functional_model" in res["message"], res


def _ddp_training_worker(rank, world_size, tmpdir, static_graph):
    model, ref = _replicas(output_bias=False)
    model.net = nn.parallel.DistributedDataParallel(
        model.net, static_graph=static_graph
    )
    x = _local_batch(rank)
    max_err = 0.0
    for _ in range(2):
        model.zero_grad()
        ref.zero_grad()
        model(x).square().sum(dim=1).mean().backward()
        ref(x).square().sum(dim=1).mean().backward()
        for param, ref_param in zip(
            model.net.module.parameters(), ref.net.parameters()
        ):
            if ref_param.grad is None:
                continue
            ref_grad = ref_param.grad.clone()
            dist.all_reduce(ref_grad)
            ref_grad /= world_size
            max_err = max(max_err, (param.grad - ref_grad).abs().max().item())
    save_result(tmpdir, rank, {"max_err": max_err})


def test_training_through_score_under_ddp_matches_unsharded_reference():
    for res in spawn_dist(_ddp_training_worker, args=(False,)):
        assert res["max_err"] < 1e-6, res


def test_ddp_static_graph_still_breaks_gradient_sync_through_score():
    for res in spawn_dist(_ddp_training_worker, args=(True,)):
        assert res["max_err"] > 1e-6, (
            "DDP static_graph now syncs gradients through a score prediction; "
            "drop the static_graph caveat from BaseModel.__call__."
        )
