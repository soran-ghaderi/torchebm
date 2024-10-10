import pytest
import torch
from torchebm.core.energy_function import EnergyFunction
from torchebm.samplers.langevin_dynamics import LangevinDynamics

class QuadraticEnergy(EnergyFunction):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(x**2, dim=-1)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return x

@pytest.fixture
def energy_function():
    return QuadraticEnergy()

@pytest.fixture
def langevin_sampler(energy_function):
    return LangevinDynamics(energy_function, step_size=0.1, noise_scale=0.1)

def test_langevin_dynamics_initialization():
    energy_func = QuadraticEnergy()
    sampler = LangevinDynamics(energy_func, step_size=0.1, noise_scale=0.1)
    assert isinstance(sampler, LangevinDynamics)
    assert sampler.step_size == 0.1
    assert sampler.noise_scale == 0.1

