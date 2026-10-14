"""The sampler step contract: set_timesteps, timesteps, order, step."""

import warnings

import pytest
import torch
from torch import nn

from torchebm._deprecation import _WARNED_ONCE, TorchEBMDeprecationWarning
from torchebm.core import GaussianModel, HarmonicModel, LinearScheduler
from torchebm.samplers import (
    FlowSampler,
    GradientDescentSampler,
    HamiltonianMonteCarlo,
    LangevinDynamics,
    NesterovSampler,
    PredictionType,
    RiemannianManifoldHMC,
)


class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x, t, **kwargs):
        return self.linear(x)


def identity_metric(x):
    eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    return eye.expand(x.shape[0], *eye.shape).contiguous()


def seeded():
    return torch.Generator().manual_seed(7)


@pytest.mark.parametrize("integrator, order", [("euler_maruyama", 1), ("heun", 2)])
def test_langevin_timesteps_list_every_stage(integrator, order):
    sampler = LangevinDynamics(step_size=0.1, integrator=integrator)
    sampler.set_timesteps(5)
    assert sampler.order == order
    assert sampler.num_inference_steps == 5
    assert sampler.timesteps.shape == (5 * order,)
    assert not sampler.timestep_conditioned
    assert sampler.prediction_type == "score"


def test_langevin_multistage_step_returns_stage_point_then_update():
    sampler = LangevinDynamics(step_size=0.1, noise_scale=1.0, integrator="heun")
    sampler.set_timesteps(1)
    x = torch.randn(4, 2, generator=seeded())
    score = -2.0 * x
    predictor = sampler.step(score, sampler.timesteps[0], x)
    assert torch.equal(predictor, x + 0.1 * score)
    update = sampler.step(-2.0 * predictor, sampler.timesteps[1], predictor)
    assert update.shape == x.shape
    assert not torch.equal(update, predictor)


def test_langevin_step_clamps_and_steps_schedulers():
    step_size = LinearScheduler(start_value=0.1, end_value=0.0, n_steps=10)
    sampler = LangevinDynamics(step_size=step_size, clamp=(-0.5, 0.5))
    sampler.set_timesteps(3)
    x = torch.full((4, 2), 2.0)
    for t in sampler.timesteps:
        x = sampler.step(torch.zeros_like(x), t, x)
    assert x.abs().max() <= 0.5
    assert sampler.schedulers["step_size"].step_count == 3
    sampler.set_timesteps(3)
    assert sampler.schedulers["step_size"].step_count == 0


def test_gradient_descent_step():
    sampler = GradientDescentSampler(step_size=0.1)
    sampler.set_timesteps(2)
    x = torch.randn(4, 2, generator=seeded())
    assert torch.allclose(sampler.step(-x, sampler.timesteps[0], x), x - 0.1 * x)


def test_nesterov_evaluates_at_the_lookahead_point():
    sampler = NesterovSampler(step_size=0.1, momentum=0.9)
    sampler.set_timesteps(2)
    x = torch.randn(4, 2, generator=seeded())
    t = sampler.timesteps[0]
    assert torch.equal(sampler.scale_model_input(x, t), x)
    score = -x
    x1 = sampler.step(score, t, x)
    assert torch.allclose(x1, x + 0.1 * score)
    assert torch.allclose(sampler.scale_model_input(x1, t), x1 + 0.9 * (0.1 * score))
    sampler.set_timesteps(2)
    assert torch.equal(sampler.scale_model_input(x1, t), x1)


def test_flow_timesteps_are_the_stage_times():
    sampler = FlowSampler(sample_eps=0.0, integrator="midpoint")
    sampler.set_timesteps(4)
    grid = torch.linspace(0.0, 1.0, 5)
    dt = grid[1:] - grid[:-1]
    assert sampler.order == 2
    assert torch.equal(sampler.timesteps[0::2], grid[:-1])
    assert torch.equal(sampler.timesteps[1::2], grid[:-1] + 0.5 * dt)
    assert sampler.timestep_conditioned


@pytest.mark.parametrize("last_step, n_evaluations", [("Mean", 5), (None, 4)])
def test_flow_sde_last_step_is_one_more_evaluation(last_step, n_evaluations):
    sampler = FlowSampler(mode="sde", last_step=last_step, sample_eps=0.05)
    sampler.set_timesteps(4)
    assert sampler.timesteps.shape == (n_evaluations,)
    if last_step is not None:
        assert sampler.timesteps[-1] == 1.0 - sampler.last_step_size


def test_flow_step_consumes_each_prediction_type():
    x = torch.randn(4, 2, generator=seeded())
    t = torch.tensor(0.5)
    outputs = {}
    for prediction_type in ("velocity", "score", "epsilon"):
        sampler = FlowSampler(prediction_type=prediction_type, sample_eps=0.01)
        assert sampler.prediction_type == prediction_type
        sampler.set_timesteps(2)
        outputs[prediction_type] = sampler.step(torch.ones_like(x), t, x)
    assert outputs["velocity"].shape == x.shape
    assert not torch.equal(outputs["velocity"], outputs["score"])
    assert not torch.equal(outputs["score"], outputs["epsilon"])


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


def test_flow_default_integrator_is_euler_without_a_model():
    assert type(FlowSampler().integrator).__name__ == "EulerMaruyamaIntegrator"
    assert (
        type(FlowSampler(mode="sde").integrator).__name__ == "EulerMaruyamaIntegrator"
    )


@pytest.mark.parametrize(
    "sampler",
    [
        FlowSampler(integrator="dopri5"),
        LangevinDynamics(integrator="backward_euler_maruyama"),
    ],
    ids=["adaptive", "implicit"],
)
def test_solver_side_integrators_are_rejected_by_set_timesteps(sampler):
    with pytest.raises(NotImplementedError, match="integrate"):
        sampler.set_timesteps(4)


def test_flow_set_timesteps_rejects_zero_steps():
    with pytest.raises(ValueError, match="positive"):
        FlowSampler().set_timesteps(0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: HamiltonianMonteCarlo(
            HarmonicModel(k=2.0), step_size=0.1, n_leapfrog_steps=3
        ),
        lambda: RiemannianManifoldHMC(
            HarmonicModel(k=2.0),
            metric_fn=identity_metric,
            step_size=0.1,
            n_leapfrog_steps=2,
        ),
    ],
    ids=["hmc", "rmhmc"],
)
def test_hmc_step_is_one_transition_of_sample(factory):
    x = torch.randn(6, 2, generator=seeded())
    expected = factory().sample(x=x, n_steps=1, generator=seeded())
    sampler = factory()
    actual = sampler.step(x, generator=seeded())
    assert torch.equal(actual, expected)
    assert sampler.schedulers["step_size"].step_count == 1
    assert not hasattr(sampler, "set_timesteps")


@pytest.mark.parametrize(
    "sampler",
    [
        LangevinDynamics(step_size=0.1),
        GradientDescentSampler(step_size=0.1),
        NesterovSampler(step_size=0.1),
        FlowSampler(),
    ],
    ids=["langevin", "gd", "nesterov", "flow"],
)
def test_sample_needs_a_bound_model(sampler):
    with pytest.raises(RuntimeError, match="Pipeline"):
        sampler.sample(x=torch.zeros(2, 2), n_steps=1)


@pytest.mark.parametrize(
    "construct",
    [
        lambda: LangevinDynamics(HarmonicModel(), step_size=0.1),
        lambda: GradientDescentSampler(HarmonicModel(), step_size=0.1),
        lambda: NesterovSampler(HarmonicModel(), step_size=0.1),
        lambda: FlowSampler(Field()),
        lambda: FlowSampler(prediction="velocity"),
        lambda: FlowSampler(train_eps=0.0),
        lambda: FlowSampler(negate_velocity=True),
    ],
)
def test_model_bound_construction_warns_once(construct):
    _WARNED_ONCE.clear()
    with pytest.warns(TorchEBMDeprecationWarning, match="Pipeline"):
        construct()


@pytest.mark.parametrize(
    "construct",
    [
        lambda: LangevinDynamics(step_size=0.1),
        lambda: GradientDescentSampler(step_size=0.1),
        lambda: NesterovSampler(step_size=0.1),
        lambda: FlowSampler(prediction_type="score", mode="sde"),
        lambda: HamiltonianMonteCarlo(GaussianModel(torch.zeros(2), torch.eye(2))),
    ],
)
def test_target_construction_is_silent(construct):
    _WARNED_ONCE.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchEBMDeprecationWarning)
        construct()
