"""Tests for Equilibrium Matching (EqM) loss.

Tests the EqM implementation based on the paper:
"Equilibrium Matching: A Principled Approach to Flow Matching" (arXiv:2510.02300)

Key differences from Flow Matching that are tested:
1. Gradient direction: (ε - x) instead of (x - ε)
2. Time-invariant: model receives no time conditioning
3. Truncated decay: c(γ) = λ * min(1, (1-γ)/(1-a))
"""

import pytest
import torch
import torch.nn as nn
import unittest.mock

from torchebm.losses import EquilibriumMatchingLoss
from torchebm.losses.loss_utils import compute_eqm_ct, get_interpolant


class DummyModel(nn.Module):
    """Simple model that returns a constant value."""
    
    def __init__(self, out_val=0.0):
        super().__init__()
        self.out_val = out_val
        self.last_x = None
        self.last_t = None

    def forward(self, x, t=None, **kwargs):
        self.last_x = x
        self.last_t = t
        return torch.full_like(x, self.out_val)


class LearnableModel(nn.Module):
    """Simple learnable model."""
    
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
    
    def forward(self, x, t=None, **kwargs):
        return self.linear(x)


@pytest.mark.parametrize("interpolant", ["linear", "cosine", "vp"])
def test_eqm_loss_shapes_and_finite(interpolant):
    """Test that EqM loss returns finite scalar for different interpolants."""
    model = DummyModel()
    loss_fn = EquilibriumMatchingLoss(
        model=model,
        interpolant=interpolant,
    )
    
    x = torch.randn(16, 10)
    loss = loss_fn(x)
    
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_eqm_manual_verification():
    """Verify the EqM loss calculation step-by-step.
    
    EqM loss: L = ||f(x_γ) - (ε - x)c(γ)||²
    where:
        - ε = x0 (noise)
        - x = x1 (data)
        - target = (x0 - x1) * c(t)
    """
    dim = 4
    batch_size = 2
    x1 = torch.randn(batch_size, dim)
    
    # Pre-determined x0 and t for reproducibility
    fixed_x0 = torch.randn(batch_size, dim)
    fixed_t_raw = torch.rand(batch_size) 
    
    with unittest.mock.patch(
        "torchebm.losses.equilibrium_matching.torch.randn_like",
        return_value=fixed_x0
    ), unittest.mock.patch(
        "torchebm.losses.equilibrium_matching.torch.rand",
        return_value=fixed_t_raw
    ):
        model = DummyModel(out_val=0.5)
        loss_fn = EquilibriumMatchingLoss(
            model=model,
            interpolant="linear",
            train_eps=0.0,
            device="cpu",
        )
        
        x1_cpu = x1.cpu()
        loss_val = loss_fn(x1_cpu)
        
        # Manual calculation of EqM loss
        t = fixed_t_raw
        
        interp = get_interpolant("linear")
        xt, _ = interp.interpolate(fixed_x0, x1_cpu, t)
        
        # EqM target: (ε - x) * c(γ) = (x0 - x1) * c(t)
        ct = compute_eqm_ct(t)
        ct_view = ct.view(batch_size, 1)
        target = (fixed_x0 - x1_cpu) * ct_view  # Note: x0 - x1, not x1 - x0
        
        pred = torch.full_like(xt, 0.5)
        
        expected_sq_diff = (pred - target) ** 2
        expected_loss_per_sample = expected_sq_diff.mean(dim=1)
        expected_loss = expected_loss_per_sample.mean()
        
        assert torch.allclose(loss_val, expected_loss, atol=1e-5), \
            f"Loss mismatch: {loss_val} != {expected_loss}"


def test_eqm_gradient_direction():
    """Verify that EqM uses (ε - x) direction, not (x - ε).
    
    This is the key difference from Flow Matching.
    """
    dim = 4
    batch_size = 2
    
    # Create deterministic test case
    x1 = torch.ones(batch_size, dim) * 2.0  # data
    x0 = torch.zeros(batch_size, dim)  # noise
    t = torch.tensor([0.5, 0.5])  # t < 0.8, so c(t) = 4.0
    
    interp = get_interpolant("linear")
    xt, ut = interp.interpolate(x0, x1, t)
    
    # Flow Matching target: x1 - x0 = 2 - 0 = 2
    fm_target = (x1 - x0)  # (2, 2, 2, 2)
    
    # EqM target: x0 - x1 = 0 - 2 = -2 (before scaling)
    ct = compute_eqm_ct(t)
    ct_view = ct.view(batch_size, 1)
    eqm_target = (x0 - x1) * ct_view  # (-8, -8, -8, -8) with c(t)=4
    
    # Verify targets have opposite signs
    assert torch.all(fm_target * eqm_target[:, 0:1] < 0), \
        "EqM and FM targets should have opposite signs"
    
    # Verify c(t) = 4 for t < 0.8
    assert torch.allclose(ct, torch.tensor([4.0, 4.0]))


def test_eqm_time_invariance():
    """Verify model receives zeroed time when time_invariant=True.
    
    EqM learns a time-invariant energy landscape by zeroing out time.
    """
    class TimeTrackingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.received_time = None
            
        def forward(self, x, t=None, **kwargs):
            self.received_time = t
            return torch.zeros_like(x)
    
    model = TimeTrackingModel()
    loss_fn = EquilibriumMatchingLoss(model=model, time_invariant=True)
    
    x = torch.randn(4, 4)
    loss_fn(x)
    
    # Time should be all zeros for time-invariance
    assert model.received_time is not None, "Model should receive time argument"
    assert torch.all(model.received_time == 0), \
        f"EqM should zero out time (got {model.received_time})"


def test_eqm_time_variant_mode():
    """Verify model receives actual time when time_invariant=False."""
    class TimeTrackingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.received_time = None
            
        def forward(self, x, t=None, **kwargs):
            self.received_time = t
            return torch.zeros_like(x)
    
    model = TimeTrackingModel()
    loss_fn = EquilibriumMatchingLoss(model=model, time_invariant=False)
    
    x = torch.randn(4, 4)
    loss_fn(x)
    
    # Time should NOT be all zeros
    assert model.received_time is not None
    assert not torch.all(model.received_time == 0), \
        "With time_invariant=False, model should receive actual time values"


def test_eqm_gradient_flow():
    """Ensure gradients propagate to model parameters."""
    model = LearnableModel(dim=5)
    loss_fn = EquilibriumMatchingLoss(model=model)
    
    x = torch.randn(8, 5)
    loss = loss_fn(x)
    loss.backward()
    
    assert model.linear.weight.grad is not None
    assert torch.any(model.linear.weight.grad != 0)


def test_dispersive_loss_integration():
    """Test that dispersive loss is added when enabled."""
    
    class ModelWithAct(nn.Module):
        def forward(self, x, t=None, return_act=False, **kwargs):
            out = x
            if return_act:
                return out, [x]  # Return x as activation
            return out
            
    model = ModelWithAct()
    loss_fn = EquilibriumMatchingLoss(
        model=model,
        apply_dispersion=True,
        dispersion_weight=1.0,
    )
    
    x = torch.randn(10, 5)
    loss = loss_fn(x, return_act=True)
    
    assert torch.isfinite(loss)
    
    # Also test without dispersion
    loss_fn_pure = EquilibriumMatchingLoss(
        model=model,
        apply_dispersion=False,   
    )
    loss_pure = loss_fn_pure(x, return_act=True)
    assert torch.isfinite(loss_pure)


def test_ct_truncated_decay():
    """Verify c(t) implements correct truncated decay.
    
    Paper formula: c(t) = λ * min(1, (1-t)/(1-a))
    with a=0.8, λ=4
    """
    # Test values
    t = torch.tensor([0.0, 0.4, 0.8, 0.9, 1.0])
    ct = compute_eqm_ct(t)
    
    # Expected: t <= 0.8 -> c(t) = 4, t > 0.8 -> linear decay to 0
    # c(0.0) = 4.0
    # c(0.4) = 4.0
    # c(0.8) = 4.0
    # c(0.9) = 4 * (1-0.9)/(1-0.8) = 4 * 0.5 = 2.0
    # c(1.0) = 4 * 0 = 0.0
    expected = torch.tensor([4.0, 4.0, 4.0, 2.0, 0.0])
    
    assert torch.allclose(ct, expected, atol=1e-5), \
        f"c(t) mismatch: {ct} != {expected}"


def test_ct_custom_parameters():
    """Test c(t) with custom threshold and multiplier."""
    t = torch.tensor([0.0, 0.5, 0.9, 1.0])
    
    # Test with threshold=0.5, multiplier=2.0
    ct = compute_eqm_ct(t, threshold=0.5, multiplier=2.0)
    
    # Expected: t <= 0.5 -> c(t) = 2, t > 0.5 -> linear decay
    # c(0.0) = 2.0
    # c(0.5) = 2.0 * min(1, 0.5/0.5) = 2.0
    # c(0.9) = 2.0 * (1-0.9)/(1-0.5) = 2.0 * 0.2 = 0.4
    # c(1.0) = 0.0
    expected = torch.tensor([2.0, 2.0, 0.4, 0.0])
    
    assert torch.allclose(ct, expected, atol=1e-5), \
        f"c(t) with custom params mismatch: {ct} != {expected}"


def test_eqm_loss_with_custom_ct():
    """Test EqM loss uses custom ct parameters."""
    model = LearnableModel(dim=4)
    
    # Test with different ct parameters
    loss_fn = EquilibriumMatchingLoss(
        model=model,
        ct_threshold=0.5,
        ct_multiplier=2.0,
    )
    
    x = torch.randn(8, 4)
    loss = loss_fn(x)
    
    assert torch.isfinite(loss)
    assert loss_fn.ct_threshold == 0.5
    assert loss_fn.ct_multiplier == 2.0


def test_device_movement(device):
    """Test that loss works with different devices."""
    if not torch.cuda.is_available() and device.type == "cuda":
        pytest.skip("CUDA not available")
        
    model = LearnableModel(dim=4).to(device)
    loss_fn = EquilibriumMatchingLoss(model=model, device=device)
    
    # Input on CPU should be moved automatically
    x = torch.randn(4, 4) 
    loss = loss_fn(x)
    
    assert loss.device.type == device.type


def test_train_eps():
    """Test that train_eps restricts time interval."""
    model = DummyModel()
    
    # With train_eps=0.1, time should be sampled from [0.1, 0.9]
    loss_fn = EquilibriumMatchingLoss(
        model=model,
        train_eps=0.1,
    )
    
    t0, t1 = loss_fn._check_interval()
    assert t0 == 0.1
    assert t1 == 0.9


def test_repr():
    """Test string representation."""
    model = DummyModel()
    loss_fn = EquilibriumMatchingLoss(
        model=model,
        prediction="velocity",
        energy_type="none",
        interpolant="cosine",
    )
    
    repr_str = repr(loss_fn)
    assert "EquilibriumMatchingLoss" in repr_str
    assert "CosineInterpolant" in repr_str
    assert "prediction='velocity'" in repr_str
    assert "energy_type='none'" in repr_str


def test_multidimensional_input():
    """Test with image-like input (batch, channels, height, width)."""
    class ConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)
        
        def forward(self, x, t=None, **kwargs):
            return self.conv(x)
    
    model = ConvModel()
    loss_fn = EquilibriumMatchingLoss(model=model)
    
    x = torch.randn(4, 3, 8, 8)  # batch of 4 images
    loss = loss_fn(x)
    
    assert loss.dim() == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize("energy_type", ["none", "dot", "l2", "mean"])
def test_explicit_energy_formulations(energy_type):
    """Test all energy formulation types.
    
    - 'none': Implicit EqM, model predicts gradient directly
    - 'dot': g(x) = x · f(x)
    - 'l2': g(x) = -0.5 ||f(x)||²
    - 'mean': Same as dot
    """
    model = LearnableModel(dim=4)
    loss_fn = EquilibriumMatchingLoss(
        model=model,
        energy_type=energy_type,
    )
    
    x = torch.randn(8, 4)
    loss = loss_fn(x)
    
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    
    # Verify gradient flow
    loss.backward()
    assert model.linear.weight.grad is not None


def test_explicit_energy_returns_energy():
    """Verify explicit formulations return energy in loss dict."""
    model = LearnableModel(dim=4)
    loss_fn = EquilibriumMatchingLoss(
        model=model,
        energy_type="dot",
    )
    
    x = torch.randn(8, 4)
    terms = loss_fn.training_losses(x)
    
    assert "loss" in terms
    assert "energy" in terms
    assert terms["energy"].shape == (8,)  # One energy per sample


def test_implicit_no_energy_output():
    """Verify implicit formulation does not return energy."""
    model = LearnableModel(dim=4)
    loss_fn = EquilibriumMatchingLoss(
        model=model,
        energy_type="none",
    )
    
    x = torch.randn(8, 4)
    terms = loss_fn.training_losses(x)
    
    assert "loss" in terms
    assert "energy" not in terms


def test_dot_energy_gradient():
    """Verify dot product energy gradient is computed correctly.
    
    For g(x) = x · f(x), the gradient involves:
    ∇g(x) = f(x) + xᵀ∇f(x)
    """
    dim = 2
    model = LearnableModel(dim=dim)
    loss_fn = EquilibriumMatchingLoss(
        model=model,
        energy_type="dot",
    )
    
    x = torch.randn(4, dim)
    loss = loss_fn(x)
    
    # Should be finite and backpropagable
    assert torch.isfinite(loss)
    loss.backward()
    assert model.linear.weight.grad is not None


@pytest.mark.parametrize("prediction", ["velocity", "score", "noise"])
def test_prediction_types(prediction):
    """Test all prediction types work correctly."""
    model = LearnableModel(dim=4)
    loss_fn = EquilibriumMatchingLoss(
        model=model,
        prediction=prediction,
        interpolant="linear",
    )
    
    x = torch.randn(8, 4)
    loss = loss_fn(x)
    
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    
    loss.backward()
    assert model.linear.weight.grad is not None


@pytest.mark.parametrize("loss_weight", [None, "velocity", "likelihood"])
def test_loss_weight_schemes(loss_weight):
    """Test loss weighting schemes with score/noise prediction."""
    model = LearnableModel(dim=4)
    loss_fn = EquilibriumMatchingLoss(
        model=model,
        prediction="noise",  # Use noise to test weighting
        loss_weight=loss_weight,
        interpolant="linear",
    )
    
    x = torch.randn(8, 4)
    loss = loss_fn(x)
    
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_velocity_prediction_with_energy_type():
    """Test velocity prediction combined with explicit energy."""
    model = LearnableModel(dim=4)
    loss_fn = EquilibriumMatchingLoss(
        model=model,
        prediction="velocity",
        energy_type="dot",
    )
    
    x = torch.randn(8, 4)
    terms = loss_fn.training_losses(x)
    
    assert "loss" in terms
    assert "energy" in terms
    assert torch.isfinite(terms["loss"]).all()


# Fixture for device testing
@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu") 
    return torch.device(request.param)
