r"""Library-wide `model_kwargs` conditioning contract (GitHub #237).

Guards that conditioning reaches the model through every sampler and loss, that
negatives are conditioned on the same energy as positives, that the deprecated
bare-kwargs paths warn, and that conditioning tensors are moved device-only
(never dtype-cast) so integer labels stay integral.
"""

import warnings

import pytest
import torch
import torch.nn as nn

from torchebm.core import BaseModel
from torchebm.core.base_module import _WARNED_ONCE
from torchebm.core.base_trainer import BaseTrainer
from torchebm.losses import (
    ContrastiveDivergence,
    DenoisingScoreMatching,
    EnergyMatchingLoss,
    EquilibriumMatchingLoss,
)
from torchebm.models import InteractionModel
from torchebm.samplers import (
    FlowSampler,
    GradientDescentSampler,
    HamiltonianMonteCarlo,
    LangevinDynamics,
    NesterovSampler,
    RiemannianManifoldHMC,
)


@pytest.fixture(autouse=True)
def _reset_warn_once():
    # warn_once dedups per-process; clear so each deprecation assertion sees it.
    _WARNED_ONCE.clear()
    yield
    _WARNED_ONCE.clear()


class RecordingEnergy(BaseModel):
    r"""Scalar energy that records the conditioning it receives per call."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.seen = []  # per call: None, or (dtype, shape) of the `y` kwarg

    def forward(self, x, y=None):
        self.seen.append(None if y is None else (y.dtype, tuple(y.shape)))
        return self.scale * (x**2).sum(dim=-1)


class RecordingField(nn.Module):
    r"""Vector field f(x, t) that records conditioning; EqM/flow style."""

    def __init__(self, dim=2):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.seen = []

    def forward(self, x, t=None, y=None):
        self.seen.append(None if y is None else (y.dtype, tuple(y.shape)))
        return self.lin(x)


class PlainEnergy(BaseModel):
    r"""Unconditional x-only energy: must keep working untouched."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x):
        return self.scale * (x**2).sum(dim=-1)


def _identity_metric(x):
    dim = x.shape[-1]
    eye = torch.eye(dim, dtype=x.dtype, device=x.device)
    return eye.expand(x.shape[0], dim, dim).contiguous()


def _labels(n):
    return torch.randint(0, 4, (n,), dtype=torch.long)


# --------------------------------------------------------------------------- #
# Conditioning reaches the model through every sampler
# --------------------------------------------------------------------------- #
def test_basemodel_gradient_threads_and_preserves_label_dtype():
    model = RecordingEnergy()
    x = torch.randn(6, 2)
    y = _labels(6)
    model.gradient(x, model_kwargs={"y": y})
    assert model.seen and model.seen[-1] is not None
    # GPU-first: label stays long despite gradient()'s float32 cast of x.
    assert model.seen[-1][0] == torch.long


GRADIENT_SAMPLERS = [
    ("langevin", lambda m: LangevinDynamics(m, step_size=0.01)),
    ("gd", lambda m: GradientDescentSampler(m, step_size=0.01)),
    ("nesterov", lambda m: NesterovSampler(m, step_size=0.01)),
    ("hmc", lambda m: HamiltonianMonteCarlo(m, step_size=0.02, n_leapfrog_steps=2)),
    (
        "rmhmc",
        lambda m: RiemannianManifoldHMC(
            m, metric_fn=_identity_metric, step_size=0.02, n_leapfrog_steps=2
        ),
    ),
]


@pytest.mark.parametrize("name,factory", GRADIENT_SAMPLERS, ids=[n for n, _ in GRADIENT_SAMPLERS])
def test_gradient_samplers_thread_conditioning(name, factory):
    model = RecordingEnergy()
    sampler = factory(model)
    x = torch.randn(5, 2)
    y = _labels(5)
    sampler.sample(x=x, n_steps=3, model_kwargs={"y": y}, return_diagnostics=True)
    assert model.seen, f"{name}: model never called"
    assert all(rec is not None for rec in model.seen), f"{name}: a call missed conditioning"
    assert all(rec[0] == torch.long for rec in model.seen), f"{name}: label dtype changed"


@pytest.mark.parametrize("name,factory", GRADIENT_SAMPLERS, ids=[n for n, _ in GRADIENT_SAMPLERS])
def test_gradient_samplers_unconditional_unchanged(name, factory):
    model = RecordingEnergy()
    sampler = factory(model)
    sampler.sample(x=torch.randn(4, 2), n_steps=3)
    assert model.seen and all(rec is None for rec in model.seen)


def test_flow_sampler_threads_conditioning():
    field = RecordingField()
    sampler = FlowSampler(field, interpolant="linear", integrator="euler")
    sampler.sample(x=torch.randn(5, 2), n_steps=3, model_kwargs={"y": _labels(5)})
    assert field.seen and all(rec is not None for rec in field.seen)


def test_interaction_model_threads_conditioning():
    inner = RecordingEnergy()
    repulsive = InteractionModel(inner, sigma_w=4.0, strength=0.1)
    LangevinDynamics(repulsive, step_size=0.01).sample(
        x=torch.randn(6, 2), n_steps=3, model_kwargs={"y": _labels(6)}
    )
    assert inner.seen and all(rec is not None for rec in inner.seen)


# --------------------------------------------------------------------------- #
# Conditional negatives: the correctness trap
# --------------------------------------------------------------------------- #
def test_em_conditions_positives_and_negatives():
    model = RecordingEnergy()
    loss_fn = EnergyMatchingLoss(
        model=model, lambda_cd=1.0, n_langevin_steps=2, noise_fraction=0.5
    )
    x = torch.randn(8, 2)
    loss = loss_fn(x, model_kwargs={"y": _labels(8)})
    assert torch.isfinite(loss)
    # Every model call (flow term, pos energy, Langevin negatives, neg energy)
    # must have received conditioning, or the contrastive term is mismatched.
    assert model.seen and all(rec is not None for rec in model.seen)


def test_cd_conditions_positives_and_negatives():
    model = RecordingEnergy()
    sampler = LangevinDynamics(model, step_size=0.01)
    cd = ContrastiveDivergence(model=model, sampler=sampler, k_steps=2, persistent=False)
    loss, _ = cd(torch.randn(8, 2), model_kwargs={"y": _labels(8)})
    assert torch.isfinite(loss)
    assert model.seen and all(rec is not None for rec in model.seen)


def test_dsm_threads_conditioning():
    model = RecordingEnergy()
    dsm = DenoisingScoreMatching(model=model, noise_scale=0.1)
    loss = dsm(torch.randn(8, 2), model_kwargs={"y": _labels(8)})
    assert torch.isfinite(loss)
    assert model.seen and all(rec is not None for rec in model.seen)


# --------------------------------------------------------------------------- #
# Deprecation of the bare-kwargs paths
# --------------------------------------------------------------------------- #
def test_flow_bare_kwargs_deprecated_but_works():
    field = RecordingField()
    sampler = FlowSampler(field, interpolant="linear", integrator="euler")
    with pytest.warns(DeprecationWarning):
        sampler.sample(x=torch.randn(4, 2), n_steps=2, y=_labels(4))
    assert field.seen and all(rec is not None for rec in field.seen)


def test_eqm_bare_kwargs_deprecated():
    field = RecordingField()
    loss_fn = EquilibriumMatchingLoss(model=field, energy_type="none")
    with pytest.warns(DeprecationWarning):
        loss_fn(torch.randn(6, 2), y=_labels(6))


def test_cd_option_kwargs_deprecated():
    model = RecordingEnergy()
    cd = ContrastiveDivergence(
        model=model, sampler=LangevinDynamics(model, step_size=0.01), k_steps=1
    )
    with pytest.warns(DeprecationWarning):
        cd(torch.randn(4, 2), energy_reg_weight=0.01)


def test_no_warning_on_proper_model_kwargs():
    field = RecordingField()
    loss_fn = EquilibriumMatchingLoss(model=field, energy_type="none")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        loss_fn(torch.randn(6, 2), model_kwargs={"y": _labels(6)})


# --------------------------------------------------------------------------- #
# Back-compat: unconditional x-only models raise loudly on stray conditioning
# --------------------------------------------------------------------------- #
def test_plain_model_unconditional_ok():
    sampler = LangevinDynamics(PlainEnergy(), step_size=0.01)
    out = sampler.sample(x=torch.randn(4, 2), n_steps=3)
    assert out.shape == (4, 2)


def test_plain_model_conditioning_raises_not_silently_dropped():
    sampler = LangevinDynamics(PlainEnergy(), step_size=0.01)
    with pytest.raises(TypeError):
        sampler.sample(x=torch.randn(4, 2), n_steps=3, model_kwargs={"y": _labels(4)})


# --------------------------------------------------------------------------- #
# BaseTrainer batch splitting + conditioning path
# --------------------------------------------------------------------------- #
def test_split_batch_forms():
    field = RecordingField()
    loss_fn = EquilibriumMatchingLoss(model=field, energy_type="none")
    trainer = BaseTrainer(
        model=field, optimizer=torch.optim.SGD(field.parameters(), lr=0.0), loss_fn=loss_fn
    )
    x = torch.randn(4, 2)
    y = _labels(4)

    data, mk = trainer._split_batch(x)
    assert mk == {} and data.shape == (4, 2)

    data, mk = trainer._split_batch((x, y))
    assert set(mk) == {"y"} and mk["y"].dtype == torch.long

    data, mk = trainer._split_batch({"x": x, "y": y})
    assert set(mk) == {"y"}


def test_trainer_train_step_conditions_model():
    field = RecordingField()
    loss_fn = EquilibriumMatchingLoss(model=field, energy_type="none")
    trainer = BaseTrainer(
        model=field, optimizer=torch.optim.SGD(field.parameters(), lr=0.0), loss_fn=loss_fn
    )
    trainer.train_step((torch.randn(8, 2), _labels(8)))
    assert field.seen and all(rec is not None for rec in field.seen)
