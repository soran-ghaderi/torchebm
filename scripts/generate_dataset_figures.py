"""Regenerate the dataset gallery PNGs in docs/assets/images/datasets/.

One square, axis-free scatter per synthetic dataset, colored by local density
along the brand blue ramp on the off-white brand background. Filenames are
stable; README.md and the docs embed them by name.

Run from the repo root:  python scripts/generate_dataset_figures.py
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

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

SEED = 42
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "assets" / "images" / "datasets"

# Brand palette (shared with the examples' plot.py convention).
BACKGROUND = "#fcfcfb"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "torchebm_blues",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def local_density(xy: np.ndarray, bins: int = 60) -> np.ndarray:
    """Per-point density from a 2D histogram lookup (no scipy dependency)."""
    hist, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins)
    ix = np.clip(np.searchsorted(xe, xy[:, 0]) - 1, 0, bins - 1)
    iy = np.clip(np.searchsorted(ye, xy[:, 1]) - 1, 0, bins - 1)
    d = hist[ix, iy]
    return d / d.max()


def save_scatter(data: torch.Tensor, filename: str, point_size: float = 5.0) -> None:
    xy = data.detach().cpu().numpy()
    order = np.argsort(local_density(xy))  # dense points drawn on top
    xy, dens = xy[order], local_density(xy)[order]

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.scatter(
        xy[:, 0], xy[:, 1],
        c=dens, cmap=BLUE_RAMP, vmin=0.0, vmax=1.0,
        s=point_size, linewidths=0, rasterized=True,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.margins(0.06)

    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"wrote {path}")


DATASETS = {
    "gaussian_mixture.png": GaussianMixtureDataset(
        n_samples=3000, n_components=4, std=0.1, seed=SEED
    ),
    "eight_gaussians.png": EightGaussiansDataset(n_samples=3000, std=0.05, seed=SEED),
    "two_moons.png": TwoMoonsDataset(n_samples=3000, noise=0.05, seed=SEED),
    "swiss_roll.png": SwissRollDataset(
        n_samples=3000, noise=0.05, arclength=3.0, seed=SEED
    ),
    "circle.png": CircleDataset(n_samples=1000, noise=0.05, radius=1.0, seed=SEED),
    "checkerboard.png": CheckerboardDataset(
        n_samples=10000, range_limit=3.0, noise=0.01, seed=SEED
    ),
    "pinwheel.png": PinwheelDataset(
        n_samples=3000, n_classes=5, noise=0.05,
        radial_scale=1.0, angular_scale=0.1, spiral_scale=1.2, seed=SEED,
    ),
    "grid.png": GridDataset(n_samples_per_dim=10, range_limit=1.0, noise=0.01, seed=SEED),
}

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, dataset in DATASETS.items():
        size = 9.0 if filename in ("grid.png", "circle.png") else 5.0
        save_scatter(dataset.get_data(), filename, point_size=size)
    print(f"done: {len(DATASETS)} figures in {OUTPUT_DIR}")
