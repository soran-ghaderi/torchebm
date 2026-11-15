import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from torchebm.models import TimeConditionedMLP
from torchebm.samplers import FlowMapSampler, FlowSampler

MEAN = torch.tensor([2.0, -1.0], dtype=torch.float64)
STD = 0.5


def _marginal(t):
    t = t.unsqueeze(-1)
    return t * MEAN, torch.sqrt(t.square() * STD**2 + (1 - t).square())


class GaussianFlowMap(nn.Module):
    r"""Exact flow map of the linear path from N(0, I) to N(MEAN, STD^2 I)."""

    def __init__(self, prediction_type):
        super().__init__()
        self.prediction_type = prediction_type

    def forward(self, x, t, s):
        mean_s, std_s = _marginal(s)
        mean_t, std_t = _marginal(t)
        flow_map = mean_t + std_t / std_s * (x - mean_s)
        if self.prediction_type == "flow_map":
            return flow_map
        return (flow_map - x) / (t - s).unsqueeze(-1)


class GaussianVelocity(nn.Module):
    prediction_type = "velocity"

    def forward(self, x, t):
        mean_t, std_t = _marginal(t)
        t = t.unsqueeze(-1)
        d_std = (t * STD**2 - (1 - t)) / std_t
        return MEAN + d_std / std_t * (x - mean_t)


@pytest.fixture
def noise():
    return torch.randn(256, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize("prediction_type", ["flow_map", "average_velocity"])
@pytest.mark.parametrize("n_steps", [1, 4])
def test_matches_flow_sampler(noise, prediction_type, n_steps):
    sampler = FlowMapSampler(prediction_type=prediction_type, dtype=torch.float64)
    out = sampler(GaussianFlowMap(prediction_type), noise, n_steps=n_steps)
    reference = FlowSampler(integrator="rk4", dtype=torch.float64)(
        GaussianVelocity(), noise, n_steps=200
    )
    assert_close(out, reference, atol=1e-8, rtol=1e-8)


def test_custom_grid_and_trajectory(noise):
    sampler = FlowMapSampler(dtype=torch.float64)
    model = GaussianFlowMap("flow_map")
    grid = torch.tensor([0.0, 0.3, 0.7, 1.0])
    trajectory = sampler(model, noise, t=grid, return_trajectory=True)
    assert trajectory.shape == (256, 3, 2)
    mean_t, std_t = _marginal(grid[1:].double())
    assert_close(trajectory[:, 0], mean_t[0] + std_t[0] * noise)
    assert_close(trajectory[:, -1], sampler(model, noise, n_steps=1))


def test_stochastic_one_step_is_deterministic(noise):
    model = GaussianFlowMap("flow_map")
    stochastic = FlowMapSampler(stochastic=True, dtype=torch.float64)(model, noise)
    assert_close(stochastic, FlowMapSampler(dtype=torch.float64)(model, noise))


def test_stochastic_multistep_samples_the_data_distribution():
    x = torch.randn(100_000, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    sampler = FlowMapSampler(stochastic=True, dtype=torch.float64)
    model = GaussianFlowMap("flow_map")
    out = sampler(model, x, n_steps=3, generator=torch.Generator().manual_seed(2))
    again = sampler(model, x, n_steps=3, generator=torch.Generator().manual_seed(2))
    assert_close(out, again)
    assert_close(out.mean(0), MEAN, atol=1e-2, rtol=0)
    assert_close(out.std(0), torch.full((2,), STD, dtype=torch.float64), atol=1e-2, rtol=0)


@pytest.mark.parametrize("prediction_type", ["flow_map", "average_velocity"])
def test_two_time_mlp(prediction_type):
    model = TimeConditionedMLP(
        in_dim=2, hidden_dim=32, depth=2, num_classes=3, prediction_type=prediction_type
    )
    sampler = FlowMapSampler(prediction_type=prediction_type)
    x = torch.randn(8, 2)
    y = torch.randint(0, 3, (8,))
    for stochastic in (False, True):
        sampler.stochastic = stochastic
        assert sampler(model, x, n_steps=2, model_kwargs={"y": y}).shape == (8, 2)


def test_compile():
    model = TimeConditionedMLP(in_dim=2, hidden_dim=32, depth=2, prediction_type="flow_map")
    sampler = FlowMapSampler()
    x = torch.randn(8, 2)
    compiled = torch.compile(sampler, fullgraph=True)
    assert_close(compiled(model, x, n_steps=2), sampler(model, x, n_steps=2))


def test_prediction_type_mismatch_raises(noise):
    with pytest.raises(ValueError, match="consumes 'flow_map'"):
        FlowMapSampler()(GaussianFlowMap("average_velocity"), noise)


def test_invalid_arguments_raise(noise):
    with pytest.raises(ValueError, match="prediction_type"):
        FlowMapSampler(prediction_type="velocity")
    sampler = FlowMapSampler(dtype=torch.float64)
    model = GaussianFlowMap("flow_map")
    with pytest.raises(ValueError, match="n_steps"):
        sampler(model, noise, n_steps=0)
    with pytest.raises(ValueError, match="1-D"):
        sampler(model, noise, t=torch.tensor([0.0]))
    with pytest.raises(NotImplementedError):
        sampler.sample()
