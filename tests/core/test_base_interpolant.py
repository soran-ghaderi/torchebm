import pytest
import torch

from torchebm.core.base_interpolant import BaseInterpolant, expand_t_like_x
from torchebm.interpolants import LinearInterpolant


def test_expand_t_like_x_1d_to_2d():
    x = torch.randn(4, 3)
    t = torch.rand(4)
    te = expand_t_like_x(t, x)
    assert te.shape == (4, 1)


def test_expand_t_like_x_image():
    x = torch.randn(2, 3, 8, 8)
    t = torch.rand(2)
    te = expand_t_like_x(t, x)
    assert te.shape == (2, 1, 1, 1)


def test_linear_interpolant_interpolate_endpoints():
    interp = LinearInterpolant()
    x0 = torch.randn(4, 3)
    x1 = torch.randn(4, 3)
    # t=0 in alpha-sigma convention (alpha(0)=0, sigma(0)=1) should give x0
    xt, ut = interp.interpolate(x0, x1, torch.zeros(4))
    assert torch.allclose(xt, x0, atol=1e-5)
    xt, ut = interp.interpolate(x0, x1, torch.ones(4))
    assert torch.allclose(xt, x1, atol=1e-5)


@pytest.mark.parametrize(
    "form", ["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"]
)
def test_compute_diffusion_forms_return_finite(form):
    interp = LinearInterpolant()
    x = torch.randn(4, 3)
    t = torch.rand(4) * 0.9 + 0.05
    out = interp.compute_diffusion(x, t, form=form)
    assert torch.isfinite(out).all()


def test_compute_diffusion_rejects_unknown_form():
    interp = LinearInterpolant()
    x = torch.randn(2, 3)
    t = torch.rand(2) * 0.9 + 0.05
    with pytest.raises(ValueError):
        interp.compute_diffusion(x, t, form="bogus")


def test_compute_d_alpha_alpha_ratio_t_finite_at_small_alpha():
    interp = LinearInterpolant()
    # At t=0 alpha=0; clamped denominator keeps result finite.
    t = torch.tensor([[0.0], [0.5]])
    out = interp.compute_d_alpha_alpha_ratio_t(t)
    assert torch.isfinite(out).all()


def test_velocity_score_round_trip_interior_t():
    interp = LinearInterpolant()
    x = torch.randn(4, 3)
    t = torch.full((4,), 0.5)
    score = torch.randn(4, 3)
    v = interp.score_to_velocity(score, x, t)
    score_back = interp.velocity_to_score(v, x, t)
    assert torch.allclose(score, score_back, atol=1e-4, rtol=1e-4)


def test_velocity_to_noise_returns_finite():
    interp = LinearInterpolant()
    x = torch.randn(4, 3)
    t = torch.full((4,), 0.5)
    v = torch.randn(4, 3)
    noise = interp.velocity_to_noise(v, x, t)
    assert torch.isfinite(noise).all()


def test_base_interpolant_cannot_instantiate_directly():
    with pytest.raises(TypeError):
        BaseInterpolant()
