"""Synthetic 2D benchmarks: the standard toy targets used across flow and EBM papers.

Each dataset is a torch Dataset generating (n_samples, 2) points on construction;
get_data() returns the full tensor for direct batching.
"""

import torch

from torchebm.datasets import (
    CheckerboardDataset,
    CircleDataset,
    EightGaussiansDataset,
    GaussianMixtureDataset,
    GridDataset,
    PinwheelDataset,
    SwissRollDataset,
    TwoMoonsDataset,
)

N, SEED = 2048, 0

datasets = {
    "gaussian_mixture": GaussianMixtureDataset(n_samples=N, seed=SEED),
    "eight_gaussians": EightGaussiansDataset(n_samples=N, seed=SEED),
    "two_moons": TwoMoonsDataset(n_samples=N, noise=0.05, seed=SEED),
    "swiss_roll": SwissRollDataset(n_samples=N, seed=SEED),
    "circle": CircleDataset(n_samples=N, seed=SEED),
    "checkerboard": CheckerboardDataset(n_samples=N, seed=SEED),
    "pinwheel": PinwheelDataset(n_samples=N, seed=SEED),
    "grid": GridDataset(n_samples_per_dim=45, seed=SEED),  # 45^2 = 2025 points
}

for name, ds in datasets.items():
    x = ds.get_data()                        # (N, 2)
    extent = x.abs().max().item()            # how far the support reaches
    print(f"{name:17s} shape {tuple(x.shape)}   mean {x.mean(0).round(decimals=2).tolist()}"
          f"   std {x.std(0).round(decimals=2).tolist()}   extent {extent:.2f}")

# Shared constructor contract: seed, device, dtype, plus per-shape knobs
# (n_samples, noise, std, radius, ...; the grid counts per dimension).
# len(ds) and ds[i] also work: every generator is a torch Dataset.
print("len(two_moons) =", len(datasets["two_moons"]))
