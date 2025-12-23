
import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from torchebm.samplers.gradient_descent import GradientDescentSampler, NesterovSampler
from torchebm.core import BaseModel, BaseScheduler

class QuadraticEnergy(BaseModel):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 0.5 * (x ** 2).sum(dim=1)
        
    def gradient(self, x):
        return x

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def dtype():
    return torch.float32

class TestGradientDescentSampler:
    
    def test_initialization(self, device, dtype):
        model = QuadraticEnergy()
        sampler = GradientDescentSampler(model, step_size=0.1, device=device, dtype=dtype)
        assert sampler.model is model
        assert sampler.schedulers["step_size"].get_value() == 0.1

    def test_manual_step(self, device, dtype):
        # E = 0.5 x^2 <=> grad = x
        # x_0 = 1.0
        # eta = 0.1
        # x_1 = x_0 - eta * grad(x_0) = 1.0 - 0.1 * 1.0 = 0.9
        
        model = QuadraticEnergy()
        sampler = GradientDescentSampler(model, step_size=0.1, device=device, dtype=dtype)
        
        x = torch.ones(1, 1, device=device, dtype=dtype)
        # Run 1 step
        samples = sampler.sample(x, n_steps=1)
        
        assert_close(samples, torch.full_like(samples, 0.9))

    def test_convergence(self, device, dtype):
        # Should converge to 0
        model = QuadraticEnergy()
        sampler = GradientDescentSampler(model, step_size=0.1, device=device, dtype=dtype)
        
        x = torch.ones(1, 1, device=device, dtype=dtype) * 10.0
        samples = sampler.sample(x, n_steps=200)
        
        # 10 * (0.9)^100 is very small
        assert_close(samples, torch.zeros_like(samples), atol=1e-4, rtol=1e-4)

    def test_trajectory(self, device, dtype):
        model = QuadraticEnergy()
        sampler = GradientDescentSampler(model, step_size=0.1, device=device, dtype=dtype)
        
        x = torch.ones(1, 1, device=device, dtype=dtype)
        # n_steps=2 -> initial + 2 steps = 3 states
        samples = sampler.sample(x, n_steps=2, return_trajectory=True)
        
        assert samples.shape == (1, 3, 1) # (B, T, D)
        assert_close(samples[0, 0], torch.tensor([1.0], device=device))
        assert_close(samples[0, 1], torch.tensor([0.9], device=device))
        assert_close(samples[0, 2], torch.tensor([0.81], device=device))

class TestNesterovSampler:
    
    def test_initialization(self, device, dtype):
        model = QuadraticEnergy()
        sampler = NesterovSampler(model, step_size=0.1, momentum=0.9, device=device, dtype=dtype)
        assert sampler.momentum == 0.9
        
    def test_manual_step(self, device, dtype):
        # x_0 = 1.0, v_0 = 0.0
        # mu = 0.9, eta = 0.1
        
        # Step 1:
        # lookahead = 1.0 + 0.9*0 = 1.0
        # grad = 1.0
        # v_1 = 0.9*0 - 0.1*1.0 = -0.1
        # x_1 = 1.0 + (-0.1) = 0.9
        
        # Step 2:
        # lookahead = 0.9 + 0.9*(-0.1) = 0.81
        # grad = 0.81
        # v_2 = 0.9*(-0.1) - 0.1*(0.81) = -0.09 - 0.081 = -0.171
        # x_2 = 0.9 + (-0.171) = 0.729
        
        model = QuadraticEnergy()
        sampler = NesterovSampler(model, step_size=0.1, momentum=0.9, device=device, dtype=dtype)
        
        x = torch.ones(1, 1, device=device, dtype=dtype)
        samples = sampler.sample(x, n_steps=2, return_trajectory=True)
        
        assert_close(samples[0, 1], torch.tensor([0.9], device=device))
        assert_close(samples[0, 2], torch.tensor([0.729], device=device))

    def test_scheduler(self, device, dtype):
        # Test with decaying step size
        class DecayingScheduler(BaseScheduler):
            def __init__(self, initial):
                super().__init__(initial)
            def _compute_value(self):
                # val = initial * 0.1^step_count
                return self.start_value * (0.1 ** self.step_count)
                
        model = QuadraticEnergy()
        scheduler = DecayingScheduler(0.1)
        sampler = GradientDescentSampler(model, step_size=scheduler, device=device, dtype=dtype)
        
        x = torch.ones(1, 1, device=device, dtype=dtype)
        
        # Step 1: eta=0.01. x_1 = 1 - 0.01*1 = 0.99.
        # Step 2: eta=0.001. x_2 = 0.99 - 0.001*0.99 = 0.98901
        
        samples = sampler.sample(x, n_steps=2, return_trajectory=True)
        
        assert_close(samples[0, 1], torch.tensor([0.99], device=device))
        assert_close(samples[0, 2], torch.tensor([0.98901], device=device))
