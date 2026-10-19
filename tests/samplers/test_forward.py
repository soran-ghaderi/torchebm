"""The sampler call: sampler(model, sample, num_inference_steps=...)."""

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


def seeded():
    return torch.Generator().manual_seed(11)


def test_runs_on_the_sample_device_and_dtype():
    x = torch.randn(5, 2, 3, dtype=torch.float64, generator=seeded())
    out = LangevinDynamics(step_size=0.1)(HarmonicModel(), x, num_inference_steps=4)
    assert out.shape == x.shape
    assert out.dtype == torch.float64
    assert not out.requires_grad


def test_trajectory_has_one_entry_per_update():
    sampler = LangevinDynamics(step_size=0.1, integrator="heun")
    x = torch.randn(4, 2, generator=seeded())
    trajectory = sampler(
        HarmonicModel(),
        x,
        num_inference_steps=6,
        generator=seeded(),
        return_trajectory=True,
    )
    final = sampler(HarmonicModel(), x, num_inference_steps=6, generator=seeded())
    assert trajectory.shape == (4, 6, 2)
    assert torch.equal(trajectory[:, -1], final)


def test_sde_last_step_is_the_final_trajectory_entry():
    sampler = FlowSampler(mode="sde", sample_eps=0.05)
    model = Recorder()
    x = torch.randn(3, 2, generator=seeded())
    trajectory = sampler(
        model, x, num_inference_steps=4, generator=seeded(), return_trajectory=True
    )
    assert trajectory.shape == (3, 5, 2)
    final = sampler(model, x, num_inference_steps=4, generator=seeded())
    assert torch.equal(trajectory[:, -1], final)


def test_energy_model_is_differentiated_into_the_score():
    x = torch.randn(4, 2, generator=seeded())
    sampler = LangevinDynamics(step_size=0.1)
    from_energy = sampler(
        HarmonicModel(k=2.0), x, num_inference_steps=5, generator=seeded()
    )
    from_score = sampler(
        HarmonicModel(k=2.0, prediction_type="score"),
        x,
        num_inference_steps=5,
        generator=seeded(),
    )
    assert torch.equal(from_energy, from_score)


def test_tuple_output_is_indexed():
    x = torch.randn(4, 2, generator=seeded())
    sampler = GradientDescentSampler(step_size=0.1)
    both = sampler(
        HarmonicModel(k=2.0, prediction_type=("energy", "score")),
        x,
        num_inference_steps=3,
    )
    score = sampler(
        HarmonicModel(k=2.0, prediction_type="score"), x, num_inference_steps=3
    )
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
        sampler(model, torch.zeros(2, 2), num_inference_steps=1)


def test_model_without_prediction_type_is_taken_as_is():
    sampler = FlowSampler(prediction_type="score", integrator="euler")
    x = sampler(
        Recorder(), torch.randn(2, 2, generator=seeded()), num_inference_steps=3
    )
    assert x.shape == (2, 2)


def test_time_conditioned_model_receives_each_timestep_per_sample():
    model = Recorder()
    sampler = FlowSampler(integrator="midpoint")
    sampler(
        model,
        torch.randn(3, 2, generator=seeded()),
        num_inference_steps=4,
        model_kwargs={"y": 1},
    )
    assert len(model.calls) == sampler.timesteps.shape[0]
    for (x, t, kwargs), expected in zip(model.calls, sampler.timesteps):
        assert t.shape == (3,)
        assert torch.equal(t, expected.expand(3))
        assert kwargs == {"y": 1}


def test_energy_model_receives_no_timestep_and_the_kwargs():
    model = EnergyRecorder()
    GradientDescentSampler(step_size=0.1)(
        model,
        torch.randn(4, 2, generator=seeded()),
        num_inference_steps=2,
        model_kwargs={"y": torch.zeros(4)},
    )
    assert len(model.inputs) == 2


def test_nesterov_evaluates_at_the_lookahead_point():
    model = EnergyRecorder(prediction_type="score")
    sampler = NesterovSampler(step_size=0.1, momentum=0.9)
    x = torch.randn(4, 2, generator=seeded())
    trajectory = sampler(model, x, num_inference_steps=2, return_trajectory=True)
    assert torch.equal(model.inputs[0], x)
    v1 = trajectory[:, 0] - x
    assert torch.allclose(model.inputs[1], trajectory[:, 0] + 0.9 * v1)


def test_generator_makes_sampling_reproducible():
    sampler = LangevinDynamics(step_size=0.1)
    x = torch.randn(4, 2, generator=seeded())
    a = sampler(HarmonicModel(), x, num_inference_steps=5, generator=seeded())
    b = sampler(HarmonicModel(), x, num_inference_steps=5, generator=seeded())
    c = sampler(HarmonicModel(), x, num_inference_steps=5)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_no_gradients_are_left_behind():
    model = Recorder()
    x = FlowSampler(integrator="euler")(
        model, torch.randn(2, 2, generator=seeded()), num_inference_steps=3
    )
    assert not x.requires_grad
    assert all(p.grad is None for p in model.parameters())


def test_adaptive_integrator_solves_and_returns_the_final_state():
    sampler = FlowSampler()
    model = Recorder()
    x = torch.randn(3, 2, generator=seeded())
    out = sampler(model, x, num_inference_steps=4)
    assert out.shape == x.shape
    assert len(model.calls) > 4
    with pytest.raises(NotImplementedError, match="fixed-step"):
        sampler(model, x, num_inference_steps=4, return_trajectory=True)
    with pytest.raises(ValueError, match="positive"):
        sampler(model, x, num_inference_steps=0)


def test_implicit_integrator_runs_through_the_call():
    sampler = LangevinDynamics(step_size=0.1, integrator="backward_euler_maruyama")
    model = EnergyRecorder()
    x = torch.randn(3, 2, generator=seeded())
    trajectory = sampler(model, x, num_inference_steps=2, return_trajectory=True)
    assert trajectory.shape == (3, 2, 2)
    assert len(model.inputs) == 2 * sampler.integrator.solver_max_iter


def test_hmc_trajectory_records_every_transition():
    sampler = HamiltonianMonteCarlo(step_size=0.1, n_leapfrog_steps=3)
    x = torch.randn(4, 2, generator=seeded())
    trajectory = sampler(
        HarmonicModel(),
        x,
        num_inference_steps=3,
        generator=seeded(),
        return_trajectory=True,
    )
    final = sampler(HarmonicModel(), x, num_inference_steps=3, generator=seeded())
    assert trajectory.shape == (4, 3, 2)
    assert torch.equal(trajectory[:, -1], final)
