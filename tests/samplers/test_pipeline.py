"""Pipeline: a model run through a sampler's step contract."""

import pytest
import torch
from torch import nn

from torchebm.core import HarmonicModel
from torchebm.samplers import (
    FlowSampler,
    GradientDescentSampler,
    HamiltonianMonteCarlo,
    LangevinDynamics,
    NesterovSampler,
    Pipeline,
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


def seeded():
    return torch.Generator().manual_seed(11)


def test_draws_the_initial_state_from_the_sampler_device_and_dtype():
    pipe = Pipeline(
        HarmonicModel(), LangevinDynamics(step_size=0.1, dtype=torch.float64)
    )
    x = pipe(
        batch_size=5, sample_shape=(2, 3), num_inference_steps=4, generator=seeded()
    )
    assert x.shape == (5, 2, 3)
    assert x.dtype == torch.float64
    assert not x.requires_grad


def test_initial_state_needs_a_shape():
    pipe = Pipeline(HarmonicModel(), LangevinDynamics(step_size=0.1))
    with pytest.raises(ValueError, match="sample_shape"):
        pipe(batch_size=4)


def test_trajectory_has_one_entry_per_update():
    pipe = Pipeline(HarmonicModel(), LangevinDynamics(step_size=0.1, integrator="heun"))
    x = torch.randn(4, 2, generator=seeded())
    trajectory = pipe(
        x, num_inference_steps=6, generator=seeded(), return_trajectory=True
    )
    final = pipe(x, num_inference_steps=6, generator=seeded())
    assert trajectory.shape == (4, 6, 2)
    assert torch.equal(trajectory[:, -1], final)


def test_sde_last_step_is_the_final_trajectory_entry():
    sampler = FlowSampler(mode="sde", sample_eps=0.05)
    pipe = Pipeline(Recorder(), sampler)
    x = torch.randn(3, 2, generator=seeded())
    trajectory = pipe(
        x, num_inference_steps=4, generator=seeded(), return_trajectory=True
    )
    assert trajectory.shape == (3, 5, 2)
    assert torch.equal(
        trajectory[:, -1], pipe(x, num_inference_steps=4, generator=seeded())
    )


def test_energy_model_is_differentiated_into_the_score():
    x = torch.randn(4, 2, generator=seeded())
    sampler = LangevinDynamics(step_size=0.1)
    from_energy = Pipeline(HarmonicModel(k=2.0), sampler)(
        x, num_inference_steps=5, generator=seeded()
    )
    from_score = Pipeline(HarmonicModel(k=2.0, prediction_type="score"), sampler)(
        x, num_inference_steps=5, generator=seeded()
    )
    assert torch.equal(from_energy, from_score)


def test_tuple_output_is_indexed():
    x = torch.randn(4, 2, generator=seeded())
    sampler = GradientDescentSampler(step_size=0.1)
    both = Pipeline(HarmonicModel(k=2.0, prediction_type=("energy", "score")), sampler)
    score = Pipeline(HarmonicModel(k=2.0, prediction_type="score"), sampler)
    assert torch.equal(both(x, num_inference_steps=3), score(x, num_inference_steps=3))


class Epsilon(nn.Module):
    prediction_type = "epsilon"

    def forward(self, x, t, **kwargs):
        return x


@pytest.mark.parametrize(
    "model, sampler",
    [
        (HarmonicModel(), FlowSampler()),
        (Epsilon(), LangevinDynamics(step_size=0.1)),
        (HarmonicModel(prediction_type=("energy", "score")), FlowSampler()),
    ],
    ids=["energy-to-velocity", "epsilon-to-score", "tuple-without-velocity"],
)
def test_mismatched_prediction_types_raise(model, sampler):
    with pytest.raises(ValueError, match="consumes"):
        Pipeline(model, sampler)


def test_model_without_prediction_type_is_taken_as_is():
    pipe = Pipeline(Recorder(), FlowSampler(prediction_type="score"))
    x = pipe(batch_size=2, sample_shape=(2,), num_inference_steps=3)
    assert x.shape == (2, 2)


def test_bound_model_samplers_are_rejected():
    with pytest.raises(TypeError, match="step contract"):
        Pipeline(HarmonicModel(), HamiltonianMonteCarlo(HarmonicModel()))


def test_time_conditioned_model_receives_each_timestep_per_sample():
    model = Recorder()
    sampler = FlowSampler(integrator="midpoint")
    pipe = Pipeline(model, sampler)
    pipe(batch_size=3, sample_shape=(2,), num_inference_steps=4, model_kwargs={"y": 1})
    assert len(model.calls) == sampler.timesteps.shape[0]
    for (x, t, kwargs), expected in zip(model.calls, sampler.timesteps):
        assert t.shape == (3,)
        assert torch.equal(t, expected.expand(3))
        assert kwargs == {"y": 1}


def test_energy_model_receives_no_timestep_and_the_kwargs():
    model = EnergyRecorder()
    y = torch.zeros(4)
    Pipeline(model, GradientDescentSampler(step_size=0.1))(
        torch.randn(4, 2, generator=seeded()),
        num_inference_steps=2,
        model_kwargs={"y": y},
    )
    assert len(model.inputs) == 2
    assert (
        all(not p.grad for p in model.parameters())
        if list(model.parameters())
        else True
    )


def test_nesterov_evaluates_at_the_lookahead_point():
    model = EnergyRecorder(prediction_type="score")
    sampler = NesterovSampler(step_size=0.1, momentum=0.9)
    x = torch.randn(4, 2, generator=seeded())
    pipe = Pipeline(model, sampler)
    trajectory = pipe(x, num_inference_steps=2, return_trajectory=True)
    assert torch.equal(model.inputs[0], x)
    v1 = trajectory[:, 0] - x
    assert torch.allclose(model.inputs[1], trajectory[:, 0] + 0.9 * v1)


def test_generator_makes_sampling_reproducible():
    pipe = Pipeline(HarmonicModel(), LangevinDynamics(step_size=0.1))
    a = pipe(batch_size=4, sample_shape=(2,), num_inference_steps=5, generator=seeded())
    b = pipe(batch_size=4, sample_shape=(2,), num_inference_steps=5, generator=seeded())
    c = pipe(batch_size=4, sample_shape=(2,), num_inference_steps=5)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_no_gradients_are_left_behind():
    model = Recorder()
    pipe = Pipeline(model, FlowSampler())
    x = pipe(batch_size=2, sample_shape=(2,), num_inference_steps=3)
    assert not x.requires_grad
    assert all(p.grad is None for p in model.parameters())


def test_repr():
    pipe = Pipeline(HarmonicModel(), LangevinDynamics(step_size=0.1))
    assert repr(pipe) == "Pipeline(model=HarmonicModel, sampler=LangevinDynamics)"
