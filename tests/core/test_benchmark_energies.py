import math

import pytest
import torch

from torchebm.core.base_model import (
    GaussianMixtureModel,
    ManyWellModel,
)


def _finite_difference_gradient(model, x, eps=1e-5):
    """Central finite-difference gradient of ``model``'s energy at ``x``."""
    grad = torch.zeros_like(x)
    for i in range(x.shape[-1]):
        xp = x.clone()
        xm = x.clone()
        xp[:, i] += eps
        xm[:, i] -= eps
        grad[:, i] = (model(xp) - model(xm)) / (2 * eps)
    return grad


# --------------------------------------------------------------------------- #
# ManyWell
# --------------------------------------------------------------------------- #
def test_manywell_default_is_32_dimensional():
    m = ManyWellModel()
    assert m.dim == 32


def test_manywell_energy_zero_at_origin():
    m = ManyWellModel(dim=32)
    x = torch.zeros(2, 32)
    assert torch.allclose(m(x), torch.zeros(2), atol=1e-6)


def test_manywell_matches_closed_form():
    m = ManyWellModel(dim=8)
    x = torch.randn(5, 8)
    d = x[:, 0::2]
    v = x[:, 1::2]
    expected = (d.pow(4) - 6.0 * d.pow(2) - 0.5 * d).sum(-1) + (0.5 * v.pow(2)).sum(-1)
    assert torch.allclose(m(x), expected, atol=1e-5)


@pytest.mark.parametrize("shape", [(32,), (4, 32)])
def test_manywell_output_shape(shape):
    m = ManyWellModel(dim=32)
    y = m(torch.randn(*shape))
    assert y.shape == (shape[0] if len(shape) == 2 else 1,)
    assert torch.isfinite(y).all()


def test_manywell_rejects_odd_dim():
    with pytest.raises(ValueError):
        ManyWellModel(dim=31)


def test_manywell_rejects_wrong_input_width():
    m = ManyWellModel(dim=32)
    with pytest.raises(ValueError):
        m(torch.randn(3, 16))


def test_manywell_gradient_matches_autograd():
    m = ManyWellModel(dim=32, dtype=torch.double)
    x = torch.randn(4, 32, dtype=torch.double)
    grad = m.gradient(x)
    assert grad.shape == x.shape
    assert torch.isfinite(grad).all()
    # Gradients validated against autograd (via finite differences).
    assert torch.allclose(grad, _finite_difference_gradient(m, x), atol=1e-4)


# --------------------------------------------------------------------------- #
# Gaussian mixture
# --------------------------------------------------------------------------- #
def test_gmm_single_component_matches_negative_log_gaussian():
    mean = torch.tensor([[1.0, -2.0]])
    cov = torch.tensor([[[2.0, 0.5], [0.5, 1.0]]])
    m = GaussianMixtureModel(means=mean, covariances=cov)

    x = torch.tensor([[0.3, 0.4], [1.0, -2.0]])
    delta = x - mean[0]
    precision = torch.linalg.inv(cov[0])
    maha = torch.einsum("bi,ij,bj->b", delta, precision, delta)
    expected = 0.5 * maha + 0.5 * (2 * math.log(2 * math.pi) + torch.logdet(cov[0]))
    assert torch.allclose(m(x), expected, atol=1e-5)


def test_gmm_batch_forward_shape():
    means = torch.tensor([[-1.0, 0.0], [2.0, 1.0]])
    m = GaussianMixtureModel(means=means)
    y = m(torch.randn(6, 2))
    assert y.shape == (6,)
    assert torch.isfinite(y).all()


def test_gmm_rejects_bad_means():
    with pytest.raises(ValueError):
        GaussianMixtureModel(means=torch.randn(2, 3, 4))


def test_gmm_rejects_mismatched_covariances():
    with pytest.raises(ValueError):
        GaussianMixtureModel(
            means=torch.zeros(3, 2), covariances=torch.eye(2).expand(2, 2, 2)
        )


def test_gmm_rejects_singular_covariance():
    with pytest.raises(ValueError):
        GaussianMixtureModel(means=torch.zeros(1, 2), covariances=torch.zeros(1, 2, 2))


def test_gmm_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        GaussianMixtureModel(means=torch.zeros(3, 2), weights=torch.ones(2))


def test_gmm_gradient_matches_autograd():
    means = torch.tensor([[-1.0, 0.0], [2.0, 1.0]], dtype=torch.double)
    m = GaussianMixtureModel(means=means, dtype=torch.double)
    x = torch.randn(4, 2, dtype=torch.double)
    grad = m.gradient(x)
    assert grad.shape == x.shape
    assert torch.isfinite(grad).all()
    # Gradients validated against autograd (via finite differences).
    assert torch.allclose(grad, _finite_difference_gradient(m, x), atol=1e-4)


# --------------------------------------------------------------------------- #
# GMM-40 benchmark factory
# --------------------------------------------------------------------------- #
def test_gmm40_has_40_components_in_2d():
    m = GaussianMixtureModel.gmm40()
    assert m.means.shape == (40, 2)
    assert m.dim == 2


def test_gmm40_is_reproducible():
    a = GaussianMixtureModel.gmm40(seed=123)
    b = GaussianMixtureModel.gmm40(seed=123)
    assert torch.allclose(a.means, b.means)


def test_gmm40_forward_and_gradient_finite():
    m = GaussianMixtureModel.gmm40()
    x = torch.randn(7, 2)
    energy = m(x)
    grad = m.gradient(x)
    assert energy.shape == (7,)
    assert grad.shape == x.shape
    assert torch.isfinite(energy).all()
    assert torch.isfinite(grad).all()
