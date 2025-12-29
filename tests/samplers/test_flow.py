
import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.testing import assert_close

from torchebm.samplers.flow import FlowSampler, PredictionType
from torchebm.interpolants import LinearInterpolant, VariancePreservingInterpolant

class MockModel(nn.Module):
    def __init__(self, mode="constant", val=0.0):
        super().__init__()
        self.mode = mode
        self.val = val

    def forward(self, x, t, **kwargs):
        if self.mode == "constant":
            # Return constant value
            return torch.full_like(x, self.val)
        elif self.mode == "linear":
            # Return x * val
            return x * self.val
        return torch.zeros_like(x)

class QuadraticEnergy(nn.Module):
    def gradient(self, x):
        return x  # E = 0.5 * x^2 -> grad = x

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def dtype():
    return torch.float32

class TestFlowSampler:
    
    def test_initialization(self, device, dtype):
        model = MockModel()
        sampler = FlowSampler(model, interpolant="linear", prediction="velocity", device=device, dtype=dtype)
        assert sampler.model is model
        assert isinstance(sampler.interpolant, LinearInterpolant)
        assert sampler.prediction_type == PredictionType.VELOCITY
        assert sampler.device == device
        assert sampler.dtype == dtype

    def test_get_drift_velocity(self, device, dtype):
        # v(x,t) = 2.0
        model = MockModel(mode="constant", val=2.0).to(device)
        sampler = FlowSampler(model, interpolant="linear", prediction="velocity", device=device, dtype=dtype)
        drift_fn = sampler._get_drift()
        
        x = torch.zeros(1, 1, device=device, dtype=dtype)
        t = torch.zeros(1, device=device, dtype=dtype)
        
        # Drift should be just the model output for velocity prediction
        out = drift_fn(x, t)
        assert_close(out, torch.full_like(x, 2.0))

    def test_sample_ode_euler_constant_velocity(self, device, dtype):
        # dx/dt = v(x,t) = 1.0. x(0)=0. x(1) should be 1.0
        # FlowSampler samples from noise (x(0)) to data (x(1)).
        # Actually standard flow matching goes from t=0 (source/noise) to t=1 (target/data).
        # FlowSampler.sample_ode checks interval. 
        # For LinearInterpolant, t0=0, t1=1-eps.
        
        model = MockModel(mode="constant", val=1.0).to(device)
        sampler = FlowSampler(model, interpolant="linear", prediction="velocity", sample_eps=0.0, device=device, dtype=dtype)
        
        # x_0 = 0
        z = torch.zeros(1, 1, device=device, dtype=dtype)
        
        # Euler with 10 steps
        # dt = 0.1
        # x_{k+1} = x_k + 1.0 * 0.1
        # x_{10} = 0 + 10 * 0.1 = 1.0
        samples = sampler.sample_ode(z, num_steps=10, method="euler")
        assert_close(samples, torch.full_like(samples, 1.0), atol=1e-5, rtol=1e-5)

    def test_sample_ode_euler_linear_velocity(self, device, dtype):
        # dx/dt = x. x(0)=1. 
        # Analytical solution: x(t) = x(0) * e^t.
        # x(1) = e.
        
        model = MockModel(mode="linear", val=1.0).to(device)
        sampler = FlowSampler(model, interpolant="linear", prediction="velocity", sample_eps=0.0, device=device, dtype=dtype)
        
        z = torch.ones(1, 1, device=device, dtype=dtype)
        
        # Euler with many steps to approximate analytical solution
        # x_{k+1} = x_k + x_k * dt = x_k(1+dt)
        # x_N = x_0 * (1 + 1/N)^N -> e as N -> inf
        N = 1000
        samples = sampler.sample_ode(z, num_steps=N, method="euler")
        
        expected = torch.ones_like(samples) * np.exp(1.0)
        # Euler convergence is O(dt), so error is ~ 1/N = 0.001. 
        # (1+0.001)^1000 approx 2.7169. e approx 2.7182. diff ~ 0.0013.
        assert_close(samples, expected, atol=0.01, rtol=0.01)

    def test_sample_ode_reverse(self, device, dtype):
        # Reverse sampling: from t=1 to t=0? or just swap t0, t1.
        # _check_interval: if reverse: t0, t1 = 1-t0, 1-t1. 
        # So it goes from 1 to 0. 
        # If dx/dt = 1, then integrating from 1 to 0 gives x(0) = x(1) + \int_1^0 1 dt = x(1) - 1.
        
        model = MockModel(mode="constant", val=1.0).to(device)
        sampler = FlowSampler(model, interpolant="linear", prediction="velocity", sample_eps=0.0, device=device, dtype=dtype)
        
        z = torch.zeros(1, 1, device=device, dtype=dtype) # Start at 0
        
        # We expect change of -1.0
        samples = sampler.sample_ode(z, num_steps=10, method="euler", reverse=True)
        assert_close(samples, torch.full_like(samples, -1.0))

    def test_prediction_score(self, device, dtype):
        # Test score prediction drift conversion
        # drift for score: -drift_mean + drift_var * model_output
        # For LinearInterpolant: x_t = (1-t)x_0 + t x_1
        # alpha_t = 1-t, beta_t = t (in some formulations) 
        # Let's check existing interpolant implementation for compute_drift.
        # Actually LineaInterpolant in torchebm: 
        # compute_drift returns (drift_mean, drift_var).
        # Need to know what LinearInterpolant.compute_drift returns.
        # Assuming simple rectified flow: drift_mean depends on x/t?
        # Let's trust logic but verify it runs.
        
        interpolant = LinearInterpolant()
        model = MockModel(mode="constant", val=1.0).to(device) # Score = 1
        sampler = FlowSampler(model, interpolant=interpolant, prediction="score", device=device, dtype=dtype)
        
        x = torch.randn(2, 2, device=device, dtype=dtype)
        t = torch.tensor([0.5], device=device, dtype=dtype) # shape (1,)
        
        # Should not crash
        drift_fn = sampler._get_drift()
        # Note: drift_fn expects t as (batch,) or (1,) depending on usage?
        # In sample_ode: t_batch = ones * t_val.
        
        t_batch = torch.ones(x.size(0), device=device, dtype=dtype) * 0.5
        out = drift_fn(x, t_batch)
        assert out.shape == x.shape

    def test_sample_sde_basic(self, device, dtype):
        # Just check it runs and produces output of correct shape
        model = MockModel(mode="constant", val=0.0).to(device)
        sampler = FlowSampler(model, interpolant="linear", prediction="velocity", device=device, dtype=dtype)
        
        z = torch.randn(4, 2, device=device, dtype=dtype)
        samples = sampler.sample_sde(z, num_steps=5, method="euler", diffusion_form="constant", diffusion_norm=0.0)
        
        # With 0 diffusion and 0 velocity, samples should equal z
        assert_close(samples, z)

    def test_apply_last_step(self, device, dtype):
        # Test Mean Step: x + drift * step_size
        model = MockModel(mode="constant", val=1.0).to(device)
        sampler = FlowSampler(model, prediction="velocity", device=device, dtype=dtype)
        
        x = torch.zeros(1, 1, device=device, dtype=dtype)
        t = torch.ones(1, device=device, dtype=dtype)
        
        # velocity=1. drift=1. step_size=0.1 => x + 1*0.1 = 0.1
        out = sampler._apply_last_step(x, t, sampler._get_drift(), "Mean", 0.1)
        assert_close(out, torch.full_like(x, 0.1))

    def test_check_interval(self):
        # Test interval logic
        model = MockModel()
        # Linear, velocity, no sde -> 0 to 1
        sampler = FlowSampler(model, interpolant="linear", prediction="velocity", sample_eps=0.01)
        t0, t1 = sampler._check_interval(sde=False)
        assert t0 == 0.0
        assert t1 == 1.0

        
        # SDE -> t0 gets eps if SBDM
        t0, t1 = sampler._check_interval(sde=True, diffusion_form="SBDM")
        assert t0 == 0.01
        assert t1 == 0.99
        
        # VP -> t0 always 0? No check logic:
        # if is_vp: t1 = 1-eps
        # else: if prediction!=velocity or sde: t0=eps ...
        
        sampler_vp = FlowSampler(model, interpolant="vp", prediction="score", sample_eps=0.01)
        t0, t1 = sampler_vp._check_interval(sde=False)
        # VP default t0 is 0.0 in _check_interval logic (it is initialized to 0.0 and not changed for VP)
        assert t0 == 0.0
        assert t1 == 0.99

    @pytest.mark.parametrize("diffusion_form", [
        "constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"
    ])
    def test_diffusion_forms(self, device, dtype, diffusion_form):
        """Test all supported diffusion forms run without errors."""
        model = MockModel(mode="constant", val=0.0).to(device)
        sampler = FlowSampler(
            model, 
            interpolant="linear", 
            prediction="velocity", 
            device=device, 
            dtype=dtype
        )
        
        z = torch.randn(4, 2, device=device, dtype=dtype)
        # Run with small number of steps, diffusion_norm=0 to keep results stable
        samples = sampler.sample_sde(
            z, 
            num_steps=5, 
            method="euler", 
            diffusion_form=diffusion_form,
            diffusion_norm=0.0,
        )
        
        assert samples.shape == z.shape
        assert torch.isfinite(samples).all()

    def test_invalid_diffusion_form(self, device, dtype):
        """Test that invalid diffusion form raises error."""
        model = MockModel(mode="constant", val=0.0).to(device)
        sampler = FlowSampler(model, interpolant="linear", device=device, dtype=dtype)
        
        z = torch.randn(4, 2, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Unknown diffusion form"):
            sampler.sample_sde(z, num_steps=5, diffusion_form="invalid_form")

