"""FlowSampler: integrate a velocity field from noise to data, no training needed.

For a Gaussian target N(mu, s^2 I) under the linear path x_t = t x1 + (1-t) x0,
the marginal velocity is closed-form:

    u(x, t) = mu + (t s^2 - (1 - t)) / (t^2 s^2 + (1 - t)^2) * (x - t mu)

so the sampler's mechanics can be studied in isolation from learning.
"""

import torch
from torch import nn

from torchebm.samplers import FlowSampler

MU = torch.tensor([2.0, -1.0])
S = 0.5


class AnalyticVelocity(nn.Module):  # the exact marginal velocity field u(x, t)
    def forward(self, x, t, **kwargs):
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
        if t.ndim == 0:
            t = t.expand(x.size(0))
        t = t.view(-1, 1)
        var = t**2 * S**2 + (1 - t) ** 2
        return MU + (t * S**2 - (1 - t)) / var * (x - t * MU)


flow = FlowSampler(
    model=AnalyticVelocity(),
    interpolant="linear",
    prediction="velocity",
    integrator="euler",
)

# More steps, smaller discretization error: the samples converge to N(mu, s^2 I).
torch.manual_seed(0)
z = torch.randn(8192, 2)
for num_steps in (2, 8, 64):
    samples = flow.sample(x=z, n_steps=num_steps)
    mean_err = (samples.mean(0) - MU).norm()
    std_err = (samples.std(0) - S).abs().max()
    print(f"num_steps={num_steps:3d}  mean error {mean_err:.3f}   std error {std_err:.3f}")

# Swap integrator="euler" for "heun" or "dopri5" (adaptive) and the error at few
# steps collapses; trained velocity models (see 20-training) plug in identically.
