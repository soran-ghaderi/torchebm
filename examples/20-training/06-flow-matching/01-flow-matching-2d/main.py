"""Flow Matching in 2D: regress a velocity field, integrate it to sample.

Conditional flow matching draws (x0, x1) pairs from noise and data, places
x_t on the interpolant between them, and regresses the model onto the
interpolant velocity u_t:

    L = || v(x_t, t) - u_t ||^2

Sampling integrates the learned velocity forward from noise with FlowSampler
(no negation; the field already points noise -> data).

The model is TimeConditionedMLP, which carries DiT's adaLN-Zero conditioning
on a vector stream by default; conditioning="concat" is the hand-rolled
[x, t_emb] baseline seen in most tutorials. Both train here for comparison.
"""

import os

import torch

from torchebm.datasets import TwoMoonsDataset
from torchebm.losses import FlowMatchingLoss
from torchebm.models import TimeConditionedMLP
from torchebm.samplers import FlowSampler

SMOKE = os.getenv("TORCHEBM_SMOKE") == "1"
N_STEPS = 20 if SMOKE else 3000
N_GEN = 128 if SMOKE else 2000


def train(conditioning):
    """Train a velocity field on two-moons with the given conditioning mode."""
    torch.manual_seed(0)
    data = TwoMoonsDataset(n_samples=4000, noise=0.05, seed=0).get_data()
    field = TimeConditionedMLP(in_dim=2, conditioning=conditioning)
    loss_fn = FlowMatchingLoss(model=field, interpolant="linear")
    opt = torch.optim.Adam(field.parameters(), lr=1e-3)
    for _ in range(N_STEPS):
        batch = data[torch.randint(len(data), (256,))]
        loss = loss_fn(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return field, data


def median_dist(samples, data):
    """Median distance from each sample to the nearest data point."""
    return torch.cdist(samples, data).min(dim=1).values.median().item()


torch.manual_seed(1)
x0 = torch.randn(N_GEN, 2)

for conditioning in ("adaln_zero", "concat"):
    field, data = train(conditioning)
    samples = FlowSampler(field, interpolant="linear", integrator="euler").sample(
        x=x0.clone(), n_steps=100
    )
    print(f"{conditioning:11s} median dist to data: {median_dist(samples, data):.3f}")
