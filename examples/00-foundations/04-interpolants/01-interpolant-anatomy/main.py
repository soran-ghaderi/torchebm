"""Interpolants: the time path x_t = alpha(t) x1 + sigma(t) x0 between noise and data.

    x0 ~ N(0, I) (source), x1 ~ data (target); the conditional velocity
    u_t = d_alpha(t) x1 + d_sigma(t) x0 is the regression target of flow matching.

alpha rises 0 -> 1 (data weight) and sigma falls 1 -> 0 (noise weight), so the
endpoints are exact for every schedule: t=0 recovers x0, t=1 recovers x1. The
call draws the path point and its conditional velocity.
"""

import torch

from torchebm.interpolants import (
    CosineInterpolant,
    LinearInterpolant,
    VariancePreservingInterpolant,
)

interpolants = {
    "linear": LinearInterpolant(),
    "cosine": CosineInterpolant(),
    "vp": VariancePreservingInterpolant(),
}

t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
for name, interpolant in interpolants.items():
    alpha, d_alpha = interpolant.compute_alpha_t(t)
    sigma, d_sigma = interpolant.compute_sigma_t(t)
    print(f"{name:7s} alpha(t) = {alpha.round(decimals=3).tolist()}"
          f"   sigma(t) = {sigma.round(decimals=3).tolist()}")

torch.manual_seed(0)
x0 = torch.randn(4096, 2)
x1 = torch.randn(4096, 2) * 0.1 + 2.0
for name, interpolant in interpolants.items():
    t_mid = torch.full((4096,), 0.5)
    x_t, u_t = interpolant(x0, x1, t_mid)
    print(f"{name:7s} at t=0.5: E|x_t| = {x_t.norm(dim=1).mean():.2f}"
          f"   E|u_t| = {u_t.norm(dim=1).mean():.2f}")

interpolant = LinearInterpolant()
x_t0, _ = interpolant(x0, x1, torch.zeros(4096))
x_t1, _ = interpolant(x0, x1, torch.ones(4096))
print("endpoint error:", (x_t0 - x0).abs().max().item(), (x_t1 - x1).abs().max().item())
