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

def test_langevin_dynamics_initialization_invalid_params():
    energy_func = QuadraticEnergy()
    with pytest.raises(ValueError):
        LangevinDynamics(energy_func, step_size=-0.1, noise_scale=0.1)
    with pytest.raises(ValueError):
        LangevinDynamics(energy_func, step_size=0.1, noise_scale=-0.1)

def test_langevin_dynamics_sample(langevin_sampler):
    initial_state = torch.tensor([1.0, 1.0])
    n_steps = 100
    final_state = langevin_sampler.sample(initial_state, n_steps)
    assert final_state.shape == initial_state.shape
    assert torch.all(torch.isfinite(final_state))

def test_langevin_dynamics_sample_trajectory(langevin_sampler):
    initial_state = torch.tensor([1.0, 1.0])
    n_steps = 100
    trajectory = langevin_sampler.sample(initial_state, n_steps, return_trajectory=True)
    assert trajectory.shape == (n_steps + 1, *initial_state.shape)
    assert torch.all(torch.isfinite(trajectory))

def test_langevin_dynamics_sample_chain(langevin_sampler):
    initial_state = torch.tensor([1.0, 1.0])
    n_steps = 100
    n_samples = 10
    samples = langevin_sampler.sample_chain(initial_state, n_steps, n_samples)
    assert samples.shape == (n_samples, *initial_state.shape)
    assert torch.all(torch.isfinite(samples))

def test_langevin_dynamics_sample_parallel(langevin_sampler):
    n_chains = 5
    initial_states = torch.randn(n_chains, 2)
    n_steps = 100
    final_states = langevin_sampler.sample_parallel(initial_states, n_steps)
    assert final_states.shape == initial_states.shape
    assert torch.all(torch.isfinite(final_states))

@pytest.mark.parametrize("shape", [(2,), (3,), (2, 3), (3, 2)])
def test_langevin_dynamics_different_shapes(energy_function, shape):
    sampler = LangevinDynamics(energy_function, step_size=0.1, noise_scale=0.1)
    initial_state = torch.randn(*shape)
    n_steps = 100
    final_state = sampler.sample(initial_state, n_steps)
    assert final_state.shape == initial_state.shape
    assert torch.all(torch.isfinite(final_state))

def test_langevin_dynamics_reproducibility(energy_function):
    torch.manual_seed(42)
    sampler1 = LangevinDynamics(energy_function, step_size=0.1, noise_scale=0.1)
    initial_state = torch.tensor([1.0, 1.0])
    n_steps = 100
    result1 = sampler1.sample(initial_state, n_steps)

    torch.manual_seed(42)
    sampler2 = LangevinDynamics(energy_function, step_size=0.1, noise_scale=0.1)
    result2 = sampler2.sample(initial_state, n_steps)

    assert torch.allclose(result1, result2)

def test_langevin_dynamics_gradient_calls(energy_function):
    class GradientCountingEnergy(EnergyFunction):
        def __init__(self, energy_function):
            super().__init__()
            self.energy_function = energy_function
            self.gradient_calls = 0

        def forward(self, x):
            return self.energy_function.forward(x)

        def gradient(self, x):
            self.gradient_calls += 1
            return self.energy_function.gradient(x)

    counting_energy = GradientCountingEnergy(energy_function)
    sampler = LangevinDynamics(counting_energy, step_size=0.1, noise_scale=0.1)
    initial_state = torch.tensor([1.0, 1.0])
    n_steps = 100
    sampler.sample(initial_state, n_steps)
    assert counting_energy.gradient_calls == n_steps
