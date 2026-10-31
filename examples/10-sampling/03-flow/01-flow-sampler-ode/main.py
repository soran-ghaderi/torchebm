"""FlowSampler: integrate a velocity field from noise to data, no training needed.

For a Gaussian target N(mu, s^2 I) under the linear path x_t = t x1 + (1-t) x0,
the marginal velocity is closed-form:

    u(x, t) = mu + (t s^2 - (1 - t)) / (t^2 s^2 + (1 - t)^2) * (x - t mu)

so the sampler's mechanics can be studied in isolation from learning. More
steps mean less discretization error; integrator="heun" or the adaptive
"dopri5" collapse the error at few steps, and trained velocity models (see
20-training) plug in identically.
"""

import torch
from torch import nn

from torchebm.samplers import FlowSampler

MU = torch.tensor([2.0, -1.0])
S = 0.5


class AnalyticVelocity(nn.Module):
    def forward(self, x, t, **kwargs):
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device).view(-1, 1)
        var = t**2 * S**2 + (1 - t) ** 2
        return MU + (t * S**2 - (1 - t)) / var * (x - t * MU)


model = AnalyticVelocity()
sampler = FlowSampler(interpolant="linear", integrator="euler")

torch.manual_seed(0)
x0 = torch.randn(8192, 2)
for n_steps in (2, 8, 64):
    samples = sampler(model, x0, n_steps=n_steps)
    mean_err = (samples.mean(0) - MU).norm()
    std_err = (samples.std(0) - S).abs().max()
    print(f"n_steps={n_steps:3d}  mean error {mean_err:.3f}   std error {std_err:.3f}")
