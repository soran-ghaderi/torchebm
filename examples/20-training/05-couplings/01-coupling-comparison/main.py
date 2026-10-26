"""Couplings: how noise is paired with data before interpolation.

The coupling decides which x0 trains against which x1; straighter pairings mean
straighter probability paths and fewer sampling steps downstream.

    cost = E ||x1 - x0||^2 under the pairing

A CouplingResult unpacks as the pair (x0, x1); extras such as the per-pair
weights of unbalanced OT, which relaxes the marginal constraints and reweights
the pairs, ride on it as attributes and go to the loss's weight= argument. In
a training loop the coupling runs right before the interpolant:

    x0, x1 = coupling(x0, x1)
    x_t, u_t = path(x0, x1, t)
"""

import torch

from torchebm.couplings import (
    ExactOTCoupling,
    GreedyCoupling,
    IndependentCoupling,
    SinkhornCoupling,
    UnbalancedSinkhornCoupling,
)
from torchebm.datasets import TwoMoonsDataset

torch.manual_seed(0)
x0 = torch.randn(512, 2)
x1 = TwoMoonsDataset(n_samples=512, noise=0.05, seed=0).get_data()


def cost(a, b):
    return ((b - a) ** 2).sum(-1).mean()


couplings = {
    "independent": IndependentCoupling(),
    "greedy": GreedyCoupling(),
    "sinkhorn (reg=0.05)": SinkhornCoupling(reg=0.05),
    "exact OT": ExactOTCoupling(),
}

print(f"{'coupling':22s} E||x1 - x0||^2")
for name, coupling in couplings.items():
    x0_paired, x1_paired = coupling(x0, x1)
    print(f"{name:22s} {cost(x0_paired, x1_paired):.3f}")

result = UnbalancedSinkhornCoupling(reg=0.05, reg_marginal=1.0)(x0, x1)
weights = result.weights
print(f"unbalanced sinkhorn    {cost(result.x0, result.x1):.3f}"
      f"   weights: min {weights.min():.2f} max {weights.max():.2f} (uniform would be 1.0)")
