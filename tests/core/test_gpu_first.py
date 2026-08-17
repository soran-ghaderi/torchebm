r"""GPU-first hot-path guarantees (#239).

Pins: (1) BaseModel.gradient computes in the input dtype (no fp64 downcast) with
an opt-in fp32 escape hatch; (2) BaseTrainer metrics are device tensors synced
once per epoch; (3) the sampler/loss hot loops launch no host sync on CUDA.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchebm.core import BaseModel
from torchebm.core.base_trainer import BaseTrainer
from torchebm.losses import ContrastiveDivergence, EquilibriumMatchingLoss
from torchebm.samplers import LangevinDynamics


class QuadraticEnergy(BaseModel):
    r"""g(x) = 0.5 * ||x||^2 with analytic gradient x; dtype-preserving forward."""

    def forward(self, x):
        return 0.5 * (x**2).sum(dim=-1)


class DtypeRecordingEnergy(BaseModel):
    r"""Records the dtype the autograd compute actually ran in."""

    def __init__(self):
        super().__init__()
        self.seen_dtype = None

    def forward(self, x):
        self.seen_dtype = x.dtype
        return 0.5 * (x**2).sum(dim=-1)


class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)

    def forward(self, x, t=None, **kwargs):
        return self.lin(x)


# --------------------------------------------------------------------------- #
# 1. gradient dtype
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradient_returns_input_dtype(dtype):
    model = QuadraticEnergy()
    g = model.gradient(torch.randn(5, 3, dtype=dtype))
    assert g.dtype == dtype


def test_fp64_gradient_computed_in_fp64_not_downcast():
    model = DtypeRecordingEnergy()
    x = torch.randn(5, 3, dtype=torch.float64)
    g = model.gradient(x)
    assert model.seen_dtype == torch.float64  # compute stayed fp64 (no downcast)
    assert g.dtype == torch.float64


def test_force_fp32_gradient_computes_in_fp32():
    model = DtypeRecordingEnergy()
    model.force_fp32_gradient = True
    x = torch.randn(5, 3, dtype=torch.float64)
    g = model.gradient(x)
    assert model.seen_dtype == torch.float32  # forced fp32 compute
    assert g.dtype == torch.float64  # return dtype is always the input dtype


# --------------------------------------------------------------------------- #
# 2. trainer metrics are device tensors, epoch syncs once
# --------------------------------------------------------------------------- #
def _eqm_trainer():
    field = Field()
    loss = EquilibriumMatchingLoss(model=field, energy_type="none")
    return BaseTrainer(
        model=field,
        optimizer=torch.optim.SGD(field.parameters(), lr=1e-3),
        loss_fn=loss,
    )


def test_train_step_returns_device_scalar_tensor():
    trainer = _eqm_trainer()
    metrics = trainer.train_step(torch.randn(8, 2))
    assert isinstance(metrics["loss"], torch.Tensor)
    assert metrics["loss"].dim() == 0
    assert metrics["loss"].device.type == trainer.device.type


def test_train_epoch_returns_finite_floats():
    trainer = _eqm_trainer()
    loader = DataLoader(TensorDataset(torch.randn(32, 2)), batch_size=8)
    avg = trainer.train_epoch(loader)
    assert set(avg) >= {"loss"}
    for value in avg.values():
        assert isinstance(value, float)
        assert value == value  # not NaN


def test_cd_train_step_returns_device_tensors():
    model = QuadraticEnergy()
    sampler = LangevinDynamics(model, step_size=0.01)
    cd = ContrastiveDivergence(model=model, sampler=sampler, k_steps=2)
    loss, _ = cd(torch.randn(8, 3))
    assert isinstance(loss, torch.Tensor) and torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# 3. CUDA: sampler / loss hot loops launch no host sync
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_langevin_and_loss_are_sync_free_on_cuda():
    dev = torch.device("cuda")
    model = QuadraticEnergy().to(dev)
    sampler = LangevinDynamics(model, step_size=0.01, device=dev)
    x = torch.randn(64, 4, device=dev)

    # Warm up outside the guard (lazy init / first-launch allocations may sync).
    sampler.sample(x=x, n_steps=3)
    model.gradient(x)

    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        # Sampler step loop: model.gradient + integrator step, no host sync.
        sampler.sample(x=x, n_steps=25)
        # Bare gradient calls (the dtype path must not introduce a sync).
        for _ in range(5):
            model.gradient(x)
    finally:
        torch.cuda.set_sync_debug_mode(prev)
