r"""Tests for Noise-Contrastive Estimation (NCE) loss and noise distributions."""

import math
from typing import Optional

import pytest
import torch
import torch.nn as nn

from torchebm.core import BaseModel
from torchebm.losses import (
    GaussianMixtureNoise,
    GaussianNoise,
    NoiseContrastiveEstimation,
)


class QuadraticModel(BaseModel):
    r"""Toy quadratic energy model: E(x) = 0.5 * w * (x - mu)^2."""

    def __init__(self, init_mu: float = 0.0, init_w: float = 1.0):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor([init_mu]))
        self.log_w = nn.Parameter(torch.tensor([math.log(init_w)]))

    @property
    def w(self) -> torch.Tensor:
        return torch.exp(self.log_w)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        diff = x - self.mu
        return 0.5 * self.w * (diff**2).flatten(1).sum(dim=1)


class ConditionalQuadraticModel(BaseModel):
    r"""Conditional quadratic model consuming y."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        bias = y.float() if y is not None else 0.0
        return 0.5 * self.w * ((x - bias) ** 2).flatten(1).sum(dim=1)


# ---------------------------------------------------------------------------
# Noise Distribution Tests
# ---------------------------------------------------------------------------

def test_gaussian_noise_sampling_and_log_prob():
    loc = torch.tensor([1.0, 2.0])
    scale = torch.tensor([0.5, 1.5])
    dist = GaussianNoise(loc=loc, scale=scale)

    samples = dist.sample(100)
    assert samples.shape == (100, 2)

    # Check log_prob matches formula
    test_pt = torch.tensor([[1.0, 2.0]])
    expected_lp = -0.5 * (
        2.0 * math.log(2.0 * math.pi)
        + 2.0 * math.log(0.5)
        + 2.0 * math.log(1.5)
        + 0.0
    )
    lp = dist.log_prob(test_pt)
    assert lp.shape == (1,)
    assert torch.allclose(lp, torch.tensor([expected_lp]), atol=1e-5)


def test_gaussian_noise_invalid_scale():
    with pytest.raises(ValueError, match="scale must be strictly positive"):
        GaussianNoise(loc=0.0, scale=-1.0)

    with pytest.raises(ValueError, match="scale must be strictly positive"):
        GaussianNoise(loc=0.0, scale=0.0)


def test_gaussian_mixture_noise_sampling_and_log_prob():
    weights = torch.tensor([0.4, 0.6])
    locs = torch.tensor([[-2.0], [2.0]])
    scales = torch.tensor([[0.5], [0.8]])
    gmm = GaussianMixtureNoise(weights=weights, locs=locs, scales=scales)

    samples = gmm.sample(200)
    assert samples.shape == (200, 1)

    # Check log_prob at 2.0 is dominated by second component
    lp = gmm.log_prob(torch.tensor([[2.0]]))
    assert lp.ndim == 1
    assert torch.isfinite(lp)


def test_gaussian_mixture_noise_invalid_args():
    with pytest.raises(ValueError, match="weights must be 1-D"):
        GaussianMixtureNoise(
            weights=torch.ones(2, 2),
            locs=torch.zeros(2, 1),
            scales=torch.ones(2, 1),
        )

    with pytest.raises(ValueError, match="scales must be strictly positive"):
        GaussianMixtureNoise(
            weights=torch.tensor([0.5, 0.5]),
            locs=torch.zeros(2, 1),
            scales=torch.tensor([[-1.0], [1.0]]),
        )


# ---------------------------------------------------------------------------
# NCE Loss Tests
# ---------------------------------------------------------------------------

def test_nce_init_validation():
    model = QuadraticModel()

    with pytest.raises(ValueError, match="model cannot be None"):
        NoiseContrastiveEstimation(model=None)

    with pytest.raises(ValueError, match="noise_ratio must be positive"):
        NoiseContrastiveEstimation(model=model, noise_ratio=-1.0)

    with pytest.raises(ValueError, match="noise_ratio must be positive"):
        NoiseContrastiveEstimation(model=model, noise_ratio=0.0)


def test_nce_learnable_vs_fixed_c():
    model = QuadraticModel()
    nce_learnable = NoiseContrastiveEstimation(model=model, learnable_c=True, initial_c=1.5)
    assert isinstance(nce_learnable.c, nn.Parameter)
    assert torch.allclose(nce_learnable.c, torch.tensor([1.5]))

    nce_fixed = NoiseContrastiveEstimation(model=model, learnable_c=False, initial_c=2.0)
    assert not isinstance(nce_fixed.c, nn.Parameter)
    assert torch.allclose(nce_fixed.c, torch.tensor(2.0))


def test_nce_forward_basic_and_diagnostics():
    model = QuadraticModel()
    noise = GaussianNoise(loc=0.0, scale=1.0)
    nce = NoiseContrastiveEstimation(model=model, noise_distribution=noise, noise_ratio=2.0)

    x = torch.randn(32, 1)
    loss = nce(x)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0

    # With pre-sampled noise
    noise_samples = torch.randn(64, 1)
    loss_presampled = nce(x, noise_samples=noise_samples)
    assert torch.isfinite(loss_presampled)

    # With diagnostics
    loss_diag, diag = nce(x, return_diagnostics=True)
    assert "loss_data" in diag
    assert "loss_noise" in diag
    assert "c" in diag
    assert torch.allclose(loss_diag, diag["loss_data"] + diag["loss_noise"])


def test_nce_recovers_toy_gaussian_energy():
    r"""Test that NCE recovers the true parameters of a toy Gaussian energy.

    Data: x ~ N(mu*=2.0, sigma*=1.0^2).
    Model: E(x) = 0.5 * w * (x - mu)^2, log p_m(x) = -E(x) + c.
    Noise: y ~ N(0.0, 3.0^2).
    Optimal parameters: mu* = 2.0, w* = 1.0, c* = -log(sqrt(2*pi)) ~ -0.9189.
    """
    torch.manual_seed(42)

    true_mu = 2.0
    true_scale = 1.0
    n_samples = 4000
    data = true_mu + true_scale * torch.randn(n_samples, 1)

    model = QuadraticModel(init_mu=0.0, init_w=0.4)
    noise = GaussianNoise(loc=0.0, scale=3.0)
    nce = NoiseContrastiveEstimation(
        model=model,
        noise_distribution=noise,
        noise_ratio=2.0,
        learnable_c=True,
        initial_c=0.0,
    )

    optimizer = torch.optim.Adam(list(model.parameters()) + [nce.c], lr=0.05)

    batch_size = 256
    initial_loss = None
    final_loss = None

    for epoch in range(120):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = n_samples // batch_size
        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            batch_x = data[idx]

            optimizer.zero_grad()
            loss = nce(batch_x)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= n_batches
        if initial_loss is None:
            initial_loss = epoch_loss
        final_loss = epoch_loss

    # Assert loss decreased significantly
    assert final_loss < initial_loss

    # Assert recovered parameters are close to true values
    recovered_mu = model.mu.item()
    recovered_w = model.w.item()
    recovered_c = nce.c.item()

    assert abs(recovered_mu - true_mu) < 0.25, f"Expected mu ~ {true_mu}, got {recovered_mu}"
    assert abs(recovered_w - 1.0) < 0.35, f"Expected w ~ 1.0, got {recovered_w}"
    expected_c = -0.5 * math.log(2.0 * math.pi)
    assert abs(recovered_c - expected_c) < 0.45, f"Expected c ~ {expected_c}, got {recovered_c}"


def test_nce_with_torch_multivariate_normal():
    r"""Verify NCE correctly handles PyTorch MultivariateNormal distributions without shape inflation."""
    class DimensionCheckingModel(nn.Module):
        def forward(self, x):
            assert x.ndim == 2, f"Expected 2D input (B, D), got shape {x.shape}"
            assert x.shape[-1] == 2, f"Expected last dim 2, got {x.shape[-1]}"
            return (x ** 2).sum(dim=-1)

    dist = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
    model = DimensionCheckingModel()
    loss_fn = NoiseContrastiveEstimation(model=model, noise_distribution=dist)

    x = torch.randn(10, 2)
    loss = loss_fn(x)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_nce_with_harmonic_model_docstring_example():
    r"""Verify the public docstring example works as written."""
    from torchebm.core import HarmonicModel
    model = HarmonicModel()
    noise = GaussianNoise(loc=0.0, scale=1.0)
    loss_fn = NoiseContrastiveEstimation(model=model, noise_distribution=noise)

    x = torch.randn(32, 2)
    loss = loss_fn(x)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_nce_non_finite_log_prob_raises():
    r"""Verify that non-finite log-probabilities from noise distributions raise ValueError."""
    class BadNoise:
        def sample(self, shape):
            return torch.randn(shape)

        def log_prob(self, x):
            lp = torch.full((x.shape[0],), float("nan"))
            return lp

    model = QuadraticModel()
    loss_fn = NoiseContrastiveEstimation(model=model, noise_distribution=BadNoise())
    x = torch.randn(10, 2)

    with pytest.raises(ValueError, match="noise distribution log_prob returned non-finite values"):
        loss_fn(x)


def test_nce_scalar_energy_single_sample():
    r"""Verify that single-sample inputs (batch_size=1) with 0-d scalar model outputs work."""
    class ScalarModel(nn.Module):
        def forward(self, x):
            # Returns a 0-d scalar if input has 1 sample, or 1-d otherwise
            return (x ** 2).sum()

    model = ScalarModel()
    loss_fn = NoiseContrastiveEstimation(model=model, noise_ratio=1.0)
    x = torch.randn(1, 2)
    loss = loss_fn(x)
    assert torch.isfinite(loss)
    assert loss.ndim == 0

