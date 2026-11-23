import pytest
import torch
from torch import nn
from torch.testing import assert_close

from torchebm.samplers import MixedFieldSampler

A, LAM, R = 0.8, 4.0, 0.5


class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, t[:, None]], dim=-1))


class Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.times = []

    def forward(self, x, t):
        self.times.append(t.clone())
        return torch.zeros_like(x)


def _reference_sample(model, x, n_steps, n_refinement_steps, heun):
    r"""Port of ``_mixed_sample`` (mixed-field e2e04f00, ``eval/image_metrics.py``)
    on the reference image field, which has the EqM sign (away from the data);
    its ``eqm_descent(..., n_refinement_steps + 1, eta)`` (``sampling/descent.py``)
    runs ``n_refinement_steps`` updates."""

    def field(x, t):
        return -model(x, t)

    dt = A / n_steps
    for i in range(n_steps):
        t0 = i * dt
        tt = torch.full((x.shape[0],), t0, dtype=x.dtype)
        v0 = -field(x, tt) / LAM
        if heun:
            t1 = torch.full_like(tt, min(t0 + dt, A))
            v1 = -field(x + dt * v0, t1) / LAM
            x = x + 0.5 * dt * (v0 + v1)
        else:
            x = x + dt * v0
    eta = R * (1.0 - A) / LAM
    tau = torch.full((x.shape[0],), A, dtype=x.dtype)
    for _ in range(n_refinement_steps):
        x = x - eta * field(x, tau)
    return x


@pytest.mark.parametrize("integrator", ["euler", "heun"])
def test_matches_reference(integrator):
    torch.manual_seed(0)
    model = Field().double()
    x = torch.randn(64, 2, dtype=torch.float64)
    sampler = MixedFieldSampler(dtype=torch.float64, integrator=integrator)
    samples = sampler(model, x, n_steps=23, n_refinement_steps=7)
    expected = _reference_sample(model, x, 23, 7, heun=integrator == "heun")
    assert_close(samples, expected, rtol=1e-12, atol=1e-12)


def test_refinement_contracts_by_one_minus_rate():
    data = torch.tensor([[0.3, -0.7]], dtype=torch.float64)

    def field(x, t):
        return LAM / (1 - A) * (data - x) * (t >= A)[:, None]

    class Ideal(nn.Module):
        def forward(self, x, t):
            return field(x, t)

    x = torch.randn(16, 2, dtype=torch.float64)
    sampler = MixedFieldSampler(dtype=torch.float64)
    trajectory = sampler(
        Ideal(), x, n_steps=1, n_refinement_steps=5, return_trajectory=True
    )
    residual = (trajectory - data).norm(dim=-1)
    assert_close(residual[:, 2:], residual[:, 1:-1] * (1 - R), rtol=1e-12, atol=0)


def test_time_inputs_and_trajectory_shape():
    recorder = Recorder()
    sampler = MixedFieldSampler(dtype=torch.float64)
    trajectory = sampler(
        recorder,
        torch.zeros(5, 2, dtype=torch.float64),
        n_steps=4,
        n_refinement_steps=3,
        return_trajectory=True,
    )
    times = torch.stack([t[0] for t in recorder.times])
    assert trajectory.shape == (5, 7, 2)
    assert_close(
        times, torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 0.8, 0.8], dtype=torch.float64)
    )


def test_ct_threshold_must_lie_inside_the_unit_interval():
    with pytest.raises(ValueError, match="ct_threshold"):
        MixedFieldSampler(ct_threshold=0.0)
