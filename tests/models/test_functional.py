r"""Tests for functional model operations."""

import pytest
import torch

from torchebm.core import HarmonicModel
from torchebm.models import energy_gradient


def test_energy_gradient_has_energy_sign_and_score_is_negative():
    x = torch.tensor([[1.0, -2.0], [3.0, 0.5]], requires_grad=True)
    energy = 0.5 * x.square().sum(dim=-1)

    gradient = energy_gradient(energy, x)

    assert torch.equal(gradient, x)
    assert torch.equal(-gradient, -x)
    assert not gradient.requires_grad


def test_energy_gradient_retains_training_parameter_graph():
    weight = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    x = torch.tensor([[1.0, -2.0], [0.5, 4.0]], requires_grad=True)
    energy = 0.5 * (weight * x.square()).sum(dim=-1)

    gradient = energy_gradient(energy, x, create_graph=True)
    gradient.square().sum().backward()

    expected = 2.0 * weight.detach() * x.detach().square().sum(dim=0)
    assert torch.equal(gradient, weight.detach() * x.detach())
    assert torch.equal(weight.grad, expected)


def test_energy_gradient_requires_connected_tensors():
    x = torch.randn(3, 2, requires_grad=True)
    other = torch.randn(3, requires_grad=True)

    with pytest.raises(RuntimeError):
        energy_gradient(other.square(), x)
    with pytest.raises(RuntimeError, match="differentiably connected"):
        energy_gradient(torch.zeros(3), x)


def test_base_model_gradient_preserves_sampling_behavior():
    model = HarmonicModel(k=2.0)
    x = torch.randn(4, 3, requires_grad=True)

    gradient = model.gradient(x)

    assert torch.allclose(gradient, 2.0 * x)
    assert not gradient.requires_grad
