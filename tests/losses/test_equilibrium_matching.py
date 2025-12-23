
import pytest
import torch
import torch.nn as nn
import unittest.mock

from torchebm.losses import EquilibriumMatchingLoss
from torchebm.losses.loss_utils import compute_eqm_ct, get_interpolant
from torchebm.interpolants import expand_t_like_x

class DummyModel(nn.Module):
    def __init__(self, out_val=0.0):
        super().__init__()
        self.out_val = out_val
        self.last_x = None
        self.last_t = None

    def forward(self, x, t):
        self.last_x = x
        self.last_t = t
        return torch.full_like(x, self.out_val)

class LearnableModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
    
    def forward(self, x, t):
        return self.linear(x)

@pytest.mark.parametrize("prediction", ["velocity", "score", "noise"])
@pytest.mark.parametrize("interpolant", ["linear", "cosine", "vp"])
def test_eqm_loss_shapes_and_finite(prediction, interpolant):
    model = DummyModel()
    loss_fn = EquilibriumMatchingLoss(
        model=model,
        prediction=prediction,
        interpolant=interpolant,
    )
    
    x = torch.randn(16, 10)
    loss = loss_fn(x)
    
    assert loss.dim() == 0
    assert torch.isfinite(loss)

def test_eqm_manual_verification():
    # Verify the loss calculation step-by-step manually
    # We use mocking to ensure deterministic behavior for validation
    
    dim = 4
    batch_size = 2
    x1 = torch.randn(batch_size, dim)
    
    # Pre-determined x0 and t
    fixed_x0 = torch.randn(batch_size, dim)
    fixed_t_raw = torch.rand(batch_size) 
    
    with unittest.mock.patch("torchebm.losses.equilibrium_matching.torch.randn_like", return_value=fixed_x0) as mock_randn, \
         unittest.mock.patch("torchebm.losses.equilibrium_matching.torch.rand", return_value=fixed_t_raw) as mock_rand:
         
        model = DummyModel(out_val=0.5)
        # Force device='cpu' to avoid potential mismatch if dev env defaults to cuda but mocks are cpu
        loss_fn = EquilibriumMatchingLoss(
            model=model,
            prediction="velocity",
            interpolant="linear",
            train_eps=0.0,
            device="cpu",
        )
        
        # Ensure x1 is on CPU for simple tensor math
        x1_cpu = x1.cpu()
        loss_val = loss_fn(x1_cpu)
        
        # Manual Calculation using fixed_x0 and fixed_t
        t = fixed_t_raw
        
        interp = get_interpolant("linear")
        xt, ut = interp.interpolate(fixed_x0, x1_cpu, t)
        
        ct = compute_eqm_ct(t)
        ct_view = ct.view(batch_size, 1)
        target = ut * ct_view
        
        pred = torch.full_like(xt, 0.5)
        
        expected_sq_diff = (pred - target) ** 2
        expected_loss_per_sample = expected_sq_diff.mean(dim=1)
        expected_loss = expected_loss_per_sample.mean()
        
        assert torch.allclose(loss_val, expected_loss, atol=1e-5), \
            f"Loss mismatch: {loss_val} != {expected_loss}"

def test_eqm_gradient_flow():
    # Ensure gradients propagate to model
    model = LearnableModel(dim=5)
    loss_fn = EquilibriumMatchingLoss(model=model, prediction="velocity")
    
    x = torch.randn(8, 5)
    loss = loss_fn(x)
    loss.backward()
    
    assert model.linear.weight.grad is not None
    assert torch.any(model.linear.weight.grad != 0)

def test_dispersive_loss_integration():
    # Test if dispersive loss is added when requested
    
    class ModelWithAct(nn.Module):
        def forward(self, x, t, return_act=False):
            out = x
            if return_act:
                return out, [x] # Return x as activation
            return out
            
    model = ModelWithAct()
    loss_fn = EquilibriumMatchingLoss(
        model=model,
        prediction="velocity",
        apply_dispersion=True,
        dispersion_weight=1.0,
    )
    
    x = torch.randn(10, 5)
    loss = loss_fn(x, return_act=True)
    
    assert torch.isfinite(loss)
    
    # Also ensure the pure velocity loss (without dispersion) runs and returns a finite value.
    loss_fn_pure = EquilibriumMatchingLoss(
        model=model,
        prediction="velocity",
        apply_dispersion=False,   
    )
    loss_pure = loss_fn_pure(x, return_act=True)
    assert torch.isfinite(loss_pure)

def test_loss_weighting_types():
    model = DummyModel()
    
    # Test valid weights
    for w in ["velocity", "likelihood", None]:
        loss_fn = EquilibriumMatchingLoss(
            model=model,
            prediction="score", 
            loss_weight=w,
        )
        x = torch.randn(4, 4)
        loss = loss_fn(x)
        assert torch.isfinite(loss)

def test_device_movement(device):
    if not torch.cuda.is_available() and device.type == "cuda":
        pytest.skip("CUDA not available")
        
    model = LearnableModel(dim=4).to(device)
    loss_fn = EquilibriumMatchingLoss(model=model, device=device)
    
    # Input on CPU
    x = torch.randn(4, 4) 
    loss = loss_fn(x)
    
    assert loss.device.type == device.type


# Fixture for device to match other tests' pattern
@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu") 
    return torch.device(request.param)
