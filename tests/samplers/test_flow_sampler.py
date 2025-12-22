import pytest
import torch

from torchebm.samplers import FlowSampler


class ZeroModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.zeros_like(x)


@pytest.mark.parametrize("device", ["cpu"])
def test_flow_sampler_sample_ode_dispatch(device):
    model = ZeroModel().to(device)
    sampler = FlowSampler(model=model, prediction="velocity", device=device, dtype=torch.float32)

    out = sampler.sample(
        dim=4,
        n_samples=3,
        n_steps=5,
        mode="ode",
        ode_method="euler",
    )
    assert out.shape == (3, 4)
    assert out.device.type == device
    assert out.dtype == torch.float32
    assert torch.all(torch.isfinite(out))


@pytest.mark.parametrize("device", ["cpu"])
def test_flow_sampler_sample_sde_dispatch(device):
    model = ZeroModel().to(device)
    sampler = FlowSampler(model=model, prediction="score", device=device, dtype=torch.float32)

    out = sampler.sample(
        dim=4,
        n_samples=3,
        n_steps=5,
        mode="sde",
        sde_method="euler",
        last_step=None,
    )
    assert out.shape == (3, 4)
    assert out.device.type == device
    assert out.dtype == torch.float32
    assert torch.all(torch.isfinite(out))


@pytest.mark.parametrize("device", ["cpu"])
def test_flow_sampler_sample_shape(device):
    model = ZeroModel().to(device)
    sampler = FlowSampler(model=model, prediction="velocity", device=device, dtype=torch.float32)

    out = sampler.sample(
        shape=(2, 3, 8, 8),
        n_steps=3,
        mode="ode",
        ode_method="euler",
    )
    assert out.shape == (2, 3, 8, 8)
    assert out.device.type == device
    assert out.dtype == torch.float32
    assert torch.all(torch.isfinite(out))


