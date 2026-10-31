"""The sampler call: sampler(model, sample, n_steps=...)."""

import warnings

import pytest
import torch
from torch import nn

from torchebm._deprecation import _WARNED_ONCE, TorchEBMDeprecationWarning
from torchebm.core import HarmonicModel, LinearScheduler
from torchebm.samplers import (
    FlowSampler,
    GradientDescentSampler,
    HamiltonianMonteCarlo,
    LangevinDynamics,
    NesterovSampler,
    PredictionType,
    RiemannianManifoldHMC,
)


class Recorder(nn.Module):
    """Velocity field that records every call."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.calls = []

    def forward(self, x, t, **kwargs):
        self.calls.append((x.detach().clone(), t.detach().clone(), kwargs))
        return self.linear(x)


class EnergyRecorder(HarmonicModel):
    def __init__(self, **kwargs):
        super().__init__(k=2.0, **kwargs)
        self.inputs = []

    def forward(self, x, y=None):
        self.inputs.append(x.detach().clone())
        energy = super().forward(x)
        return energy if y is None else energy + y


class Epsilon(nn.Module):
    prediction_type = "epsilon"

    def forward(self, x, t, **kwargs):
        return x


def identity_metric(x):
    eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    return eye.expand(x.shape[0], *eye.shape).contiguous()


def seeded():
    return torch.Generator().manual_seed(11)


def test_runs_on_the_sample_device_and_dtype():
    x = torch.randn(5, 2, 3, dtype=torch.float64, generator=seeded())
    out = LangevinDynamics(step_size=0.1)(HarmonicModel(), x, n_steps=4)
    assert out.shape == x.shape
    assert out.dtype == torch.float64
    assert not out.requires_grad


def test_trajectory_has_one_entry_per_update():
    sampler = LangevinDynamics(step_size=0.1, integrator="heun")
    x = torch.randn(4, 2, generator=seeded())
    trajectory = sampler(
        HarmonicModel(),
        x,
        n_steps=6,
        generator=seeded(),
        return_trajectory=True,
    )
    final = sampler(HarmonicModel(), x, n_steps=6, generator=seeded())
    assert trajectory.shape == (4, 6, 2)
    assert torch.equal(trajectory[:, -1], final)


def test_sde_last_step_is_the_final_trajectory_entry():
    sampler = FlowSampler(mode="sde", sample_eps=0.05)
    model = Recorder()
    x = torch.randn(3, 2, generator=seeded())
    trajectory = sampler(
        model, x, n_steps=4, generator=seeded(), return_trajectory=True
    )
    assert trajectory.shape == (3, 4, 2)
    final = sampler(model, x, n_steps=4, generator=seeded())
    assert torch.equal(trajectory[:, -1], final)


def test_energy_model_is_differentiated_into_the_score():
    x = torch.randn(4, 2, generator=seeded())
    sampler = LangevinDynamics(step_size=0.1)
    from_energy = sampler(HarmonicModel(k=2.0), x, n_steps=5, generator=seeded())
    from_score = sampler(
        HarmonicModel(k=2.0, prediction_type="score"),
        x,
        n_steps=5,
        generator=seeded(),
    )
    assert torch.equal(from_energy, from_score)


def test_tuple_output_is_indexed():
    x = torch.randn(4, 2, generator=seeded())
    sampler = GradientDescentSampler(step_size=0.1)
    both = sampler(
        HarmonicModel(k=2.0, prediction_type=("energy", "score")),
        x,
        n_steps=3,
    )
    score = sampler(HarmonicModel(k=2.0, prediction_type="score"), x, n_steps=3)
    assert torch.equal(both, score)


@pytest.mark.parametrize(
    "model, sampler",
    [
        (HarmonicModel(), FlowSampler(integrator="euler")),
        (Epsilon(), LangevinDynamics(step_size=0.1)),
        (HarmonicModel(prediction_type=("energy", "score")), FlowSampler()),
        (HarmonicModel(prediction_type="score"), HamiltonianMonteCarlo(step_size=0.1)),
    ],
    ids=[
        "energy-to-velocity",
        "epsilon-to-score",
        "tuple-without-velocity",
        "hmc-needs-energy",
    ],
)
def test_mismatched_prediction_types_raise(model, sampler):
    with pytest.raises(ValueError, match="outputs"):
        sampler(model, torch.zeros(2, 2), n_steps=1)


def test_model_without_prediction_type_is_taken_as_is():
    sampler = FlowSampler(prediction_type="score", integrator="euler")
    x = sampler(Recorder(), torch.randn(2, 2, generator=seeded()), n_steps=3)
    assert x.shape == (2, 2)


def test_time_conditioned_model_receives_each_stage_time_per_sample():
    model = Recorder()
    FlowSampler(integrator="midpoint")(
        model,
        torch.randn(3, 2, generator=seeded()),
        n_steps=4,
        model_kwargs={"y": 1},
    )
    grid = torch.linspace(0.0, 1.0, 5)
    assert len(model.calls) == 8
    for i, (x, t, kwargs) in enumerate(model.calls):
        assert t.shape == (3,)
        assert kwargs == {"y": 1}
        if i % 2 == 0:
            assert torch.equal(t, grid[i // 2].expand(3))


def test_custom_time_grid_defines_the_steps():
    model = Recorder()
    sampler = FlowSampler(integrator="euler")
    grid = torch.tensor([0.0, 0.1, 0.3, 0.6, 1.0])
    trajectory = sampler(
        model,
        torch.randn(3, 2, generator=seeded()),
        n_steps=99,
        t=grid,
        return_trajectory=True,
    )
    assert trajectory.shape == (3, 4, 2)
    assert torch.equal(torch.stack([t[0] for _, t, _ in model.calls]), grid[:-1])
    with pytest.raises(ValueError, match="t must"):
        sampler(model, torch.zeros(2, 2), t=grid[:1])
    with pytest.raises(ValueError, match="t must"):
        sampler(model, torch.zeros(2, 2), t=grid.view(1, -1))


def test_custom_uniform_grid_matches_n_steps():
    model = Recorder()
    x = torch.randn(3, 2, generator=seeded())
    sampler = FlowSampler(integrator="midpoint")
    uniform = sampler(model, x, n_steps=5)
    custom = sampler(model, x, t=torch.linspace(0.0, 1.0, 6))
    assert torch.equal(custom, uniform)


def test_energy_model_receives_no_timestep_and_the_kwargs():
    model = EnergyRecorder()
    GradientDescentSampler(step_size=0.1)(
        model,
        torch.randn(4, 2, generator=seeded()),
        n_steps=2,
        model_kwargs={"y": torch.zeros(4)},
    )
    assert len(model.inputs) == 2


def test_nesterov_evaluates_at_the_lookahead_point():
    model = EnergyRecorder(prediction_type="score")
    sampler = NesterovSampler(step_size=0.1, momentum=0.9)
    x = torch.randn(4, 2, generator=seeded())
    trajectory = sampler(model, x, n_steps=2, return_trajectory=True)
    assert torch.equal(model.inputs[0], x)
    v1 = trajectory[:, 0] - x
    assert torch.allclose(model.inputs[1], trajectory[:, 0] + 0.9 * v1)


def test_generator_makes_sampling_reproducible():
    sampler = LangevinDynamics(step_size=0.1)
    x = torch.randn(4, 2, generator=seeded())
    a = sampler(HarmonicModel(), x, n_steps=5, generator=seeded())
    b = sampler(HarmonicModel(), x, n_steps=5, generator=seeded())
    c = sampler(HarmonicModel(), x, n_steps=5)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_no_gradients_are_left_behind():
    model = Recorder()
    x = FlowSampler(integrator="euler")(
        model, torch.randn(2, 2, generator=seeded()), n_steps=3
    )
    assert not x.requires_grad
    assert all(p.grad is None for p in model.parameters())


def test_schedulers_reset_at_every_call_and_clamp_holds():
    step_size = LinearScheduler(start_value=0.1, end_value=0.0, n_steps=10)
    sampler = LangevinDynamics(step_size=step_size, clamp=(-0.5, 0.5))
    x = torch.full((4, 2), 2.0)
    for _ in range(2):
        out = sampler(HarmonicModel(), x, n_steps=3)
        assert sampler.schedulers["step_size"].step_count == 3
    assert out.abs().max() <= 0.5


def test_adaptive_integrator_solves_and_returns_the_final_state():
    sampler = FlowSampler()
    assert type(sampler.integrator).__name__ == "Dopri5Integrator"
    model = Recorder()
    x = torch.randn(3, 2, generator=seeded())
    out = sampler(model, x, n_steps=4)
    assert out.shape == x.shape
    assert len(model.calls) > 4
    with pytest.raises(NotImplementedError, match="fixed-step"):
        sampler(model, x, n_steps=4, return_trajectory=True)
    with pytest.raises(ValueError, match="positive"):
        sampler(model, x, n_steps=0)


def test_implicit_integrator_runs_through_the_call():
    sampler = LangevinDynamics(step_size=0.1, integrator="backward_euler_maruyama")
    model = EnergyRecorder()
    x = torch.randn(3, 2, generator=seeded())
    trajectory = sampler(model, x, n_steps=2, return_trajectory=True)
    assert trajectory.shape == (3, 2, 2)
    assert len(model.inputs) == 2 * sampler.integrator.solver_max_iter


@pytest.mark.parametrize(
    "factory",
    [
        lambda: HamiltonianMonteCarlo(step_size=0.1, n_leapfrog_steps=3),
        lambda: RiemannianManifoldHMC(
            metric_fn=identity_metric, step_size=0.1, n_leapfrog_steps=2
        ),
    ],
    ids=["hmc", "rmhmc"],
)
def test_hmc_trajectory_records_every_transition(factory):
    sampler = factory()
    x = torch.randn(4, 2, generator=seeded())
    trajectory = sampler(
        HarmonicModel(),
        x,
        n_steps=3,
        generator=seeded(),
        return_trajectory=True,
    )
    final = sampler(HarmonicModel(), x, n_steps=3, generator=seeded())
    assert trajectory.shape == (4, 3, 2)
    assert torch.equal(trajectory[:, -1], final)
    assert sampler.schedulers["step_size"].step_count == 3


def test_flow_prediction_type_names_match_the_models():
    assert (
        FlowSampler(prediction_type="epsilon").prediction_type is PredictionType.NOISE
    )
    assert PredictionType.NOISE == "epsilon"
    with pytest.raises(ValueError, match="prediction_type"):
        FlowSampler(prediction_type="vector")
    with pytest.warns(TorchEBMDeprecationWarning):
        _WARNED_ONCE.clear()
        legacy = FlowSampler(prediction="noise")
    assert legacy.prediction_type == "epsilon"


@pytest.mark.parametrize(
    "sampler",
    [
        LangevinDynamics(step_size=0.1),
        GradientDescentSampler(step_size=0.1),
        NesterovSampler(step_size=0.1),
        FlowSampler(),
        HamiltonianMonteCarlo(step_size=0.1),
        RiemannianManifoldHMC(metric_fn=identity_metric, step_size=0.1),
    ],
    ids=["langevin", "gd", "nesterov", "flow", "hmc", "rmhmc"],
)
def test_sample_needs_a_bound_model(sampler):
    with pytest.raises(RuntimeError, match=r"sampler\(model"):
        sampler.sample(x=torch.zeros(2, 2), n_steps=1)


@pytest.mark.parametrize(
    "construct",
    [
        lambda: LangevinDynamics(HarmonicModel(), step_size=0.1),
        lambda: GradientDescentSampler(HarmonicModel(), step_size=0.1),
        lambda: NesterovSampler(HarmonicModel(), step_size=0.1),
        lambda: FlowSampler(Recorder()),
        lambda: FlowSampler(prediction="velocity"),
        lambda: FlowSampler(train_eps=0.0),
        lambda: FlowSampler(negate_velocity=True),
        lambda: HamiltonianMonteCarlo(HarmonicModel(), step_size=0.1),
        lambda: RiemannianManifoldHMC(HarmonicModel(), metric_fn=identity_metric),
    ],
)
def test_model_bound_construction_warns_once(construct):
    _WARNED_ONCE.clear()
    with pytest.warns(TorchEBMDeprecationWarning, match=r"sampler\(model"):
        construct()


@pytest.mark.parametrize(
    "construct",
    [
        lambda: LangevinDynamics(step_size=0.1),
        lambda: GradientDescentSampler(step_size=0.1),
        lambda: NesterovSampler(step_size=0.1),
        lambda: FlowSampler(prediction_type="score", mode="sde"),
        lambda: HamiltonianMonteCarlo(step_size=0.1),
        lambda: RiemannianManifoldHMC(metric_fn=identity_metric),
    ],
)
def test_target_construction_is_silent(construct):
    _WARNED_ONCE.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchEBMDeprecationWarning)
        construct()
