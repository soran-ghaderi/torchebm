"""Couplings: how noise is paired with data before interpolation.

The coupling decides which x0 trains against which x1; straighter pairings mean
straighter probability paths and fewer sampling steps downstream.

    cost = E ||x1 - x0||^2 under the pairing
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
x0 = torch.randn(512, 2)                                        # source: noise
x1 = TwoMoonsDataset(n_samples=512, noise=0.05, seed=0).get_data()  # target: data


def cost(a, b):
    return ((b - a) ** 2).sum(-1).mean()


couplings = {
    "independent": IndependentCoupling(),      # keep the incoming pairing
    "greedy": GreedyCoupling(),                # nearest-unmatched heuristic
    "sinkhorn (reg=0.05)": SinkhornCoupling(reg=0.05),  # entropic OT
    "exact OT": ExactOTCoupling(),             # the assignment optimum
}

print(f"{'coupling':22s} E||x1 - x0||^2")
for name, coupling in couplings.items():
    a0, a1 = coupling(x0, x1)                  # CouplingResult unpacks as the pair
    print(f"{name:22s} {cost(a0, a1):.3f}")

# Extras ride on the result object without breaking unpacking: unbalanced OT
# reweights pairs instead of enforcing exact marginals.
res = UnbalancedSinkhornCoupling(reg=0.05, reg_marginal=1.0)(x0, x1)
w = res.weights
print(f"unbalanced sinkhorn    {cost(res.x0, res.x1):.3f}"
      f"   weights: min {w.min():.2f} max {w.max():.2f} (uniform would be 1.0)")

# Every training loss that consumes (x0, x1) pairs accepts any of these via its
# coupling= argument; EnergyMatchingLoss uses the weights automatically.
