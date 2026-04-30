import warnings

import pytest
import torch

from torchebm.core.base_model import (
    AckleyModel,
    BaseModel,
    DoubleWellModel,
    GaussianModel,
    HarmonicModel,
    RastriginModel,
    RosenbrockModel,
)
from torchebm.core.base_model import (
    AckleyEnergy,
    BaseEnergyFunction,
    DoubleWellEnergy,
    GaussianEnergy,
    HarmonicEnergy,
    RastriginEnergy,
    RosenbrockEnergy,
)


def test_base_model_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseModel()


@pytest.mark.parametrize("shape", [(3,), (4, 3)])
def test_harmonic_model_output_shape(shape):
    m = HarmonicModel(k=2.0)
    x = torch.randn(*shape)
    y = m(x)
    assert y.shape == (shape[0] if len(shape) == 2 else 1,)
    assert torch.isfinite(y).all()


def test_harmonic_energy_zero_at_origin():
    m = HarmonicModel(k=1.0)
    x = torch.zeros(2, 3)
    assert torch.allclose(m(x), torch.zeros(2))


def test_double_well_energy_zero_at_wells():
    m = DoubleWellModel(barrier_height=2.0, b=1.0)
    x = torch.tensor([[1.0], [-1.0]])
    assert torch.allclose(m(x), torch.zeros(2), atol=1e-6)


def test_double_well_gradient_finite():
    m = DoubleWellModel()
    x = torch.randn(4, 3)
    g = m.gradient(x)
    assert g.shape == x.shape
    assert torch.isfinite(g).all()


def test_gaussian_model_rejects_bad_mean():
    with pytest.raises(ValueError):
        GaussianModel(mean=torch.randn(2, 2), cov=torch.eye(2))


def test_gaussian_model_rejects_non_square_cov():
    with pytest.raises(ValueError):
        GaussianModel(mean=torch.zeros(2), cov=torch.randn(2, 3))


def test_gaussian_model_rejects_singular_cov():
    with pytest.raises(ValueError):
        GaussianModel(mean=torch.zeros(2), cov=torch.zeros(2, 2))


def test_gaussian_model_rejects_mismatched_dims():
    with pytest.raises(ValueError):
        GaussianModel(mean=torch.zeros(3), cov=torch.eye(2))


def test_gaussian_model_energy_zero_at_mean():
    mean = torch.zeros(3)
    cov = torch.eye(3)
    m = GaussianModel(mean=mean, cov=cov)
    assert torch.allclose(m(mean), torch.zeros(1), atol=1e-6)


def test_gaussian_model_batch_forward():
    m = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))
    x = torch.randn(5, 2)
    y = m(x)
    assert y.shape == (5,)


def test_gaussian_model_rejects_wrong_input_shape():
    m = GaussianModel(mean=torch.zeros(2), cov=torch.eye(2))
    with pytest.raises(ValueError):
        m(torch.randn(3, 4))


def test_rosenbrock_energy_zero_at_minimum():
    m = RosenbrockModel(a=1.0, b=100.0)
    x = torch.ones(1, 3)
    assert torch.allclose(m(x), torch.zeros(1), atol=1e-5)


def test_rosenbrock_requires_min_two_dims():
    m = RosenbrockModel()
    with pytest.raises(ValueError):
        m(torch.randn(2, 1))


def test_ackley_energy_zero_at_origin():
    m = AckleyModel()
    x = torch.zeros(1, 3)
    assert torch.allclose(m(x), torch.zeros(1), atol=1e-5)


def test_rastrigin_energy_zero_at_origin():
    m = RastriginModel()
    x = torch.zeros(1, 3)
    assert torch.allclose(m(x), torch.zeros(1), atol=1e-5)


@pytest.mark.parametrize(
    "deprecated_cls",
    [
        DoubleWellEnergy,
        GaussianEnergy,
        HarmonicEnergy,
        RosenbrockEnergy,
        AckleyEnergy,
        RastriginEnergy,
    ],
)
def test_deprecated_aliases_emit_warning(deprecated_cls):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if deprecated_cls is GaussianEnergy:
            deprecated_cls(mean=torch.zeros(2), cov=torch.eye(2))
        else:
            deprecated_cls()
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_base_energy_function_alias_still_abstract():
    with pytest.raises(TypeError):
        BaseEnergyFunction()


def test_harmonic_gradient_linear_in_x():
    m = HarmonicModel(k=2.0)
    x = torch.randn(4, 3)
    g = m.gradient(x)
    assert torch.allclose(g, 2.0 * x, atol=1e-5)
