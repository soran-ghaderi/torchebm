r"""
Density transport animation for TorchEBM.

An image is treated as an unnormalised target density: pixel intensity defines
a Boltzmann density p(x) \propto rho(x), so the energy is E(x) = -log rho(x).
A cloud of particles initialised as Gaussian noise is transported into the
target by Langevin dynamics, sampled with TorchEBM's own `LangevinDynamics`
through a `BaseModel` energy. Sampling is annealed coarse-to-fine: the density
starts heavily blurred (broad basins of attraction) and is sharpened in stages,
which mirrors annealed Langevin / score-based sampling and yields a clean
assembly of fine detail.

This is the same score-based-generative intuition researchers know, with a real
image as the target rather than a toy mixture. Swap the `--target` to render any
of the bundled targets.

Run:
    python examples/90-showcase/01-density-transport/main.py --target logo
    python examples/90-showcase/01-density-transport/main.py --target galaxy
    python examples/90-showcase/01-density-transport/main.py --target neuron
    python examples/90-showcase/01-density-transport/main.py --target starry_night
    python examples/90-showcase/01-density-transport/main.py --target all
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image, ImageFilter

from torchebm.core import BaseModel
from torchebm.samplers.langevin_dynamics import LangevinDynamics

# ----------------------------------------------------------------------------- #
# Paths and remote sources
# ----------------------------------------------------------------------------- #

# Outputs land next to this script (assets/ is gitignored for examples).
OUT = Path(__file__).resolve().parent / "assets"
CACHE = OUT / "_targets"
CACHE.mkdir(parents=True, exist_ok=True)

_UA = "Mozilla/5.0 (X11; Linux x86_64) torchebm-density-transport"

# Public-domain / freely licensed source images, resolved via Wikimedia's
# Special:FilePath endpoint (redirects to the real, hash-pathed file URL) so we
# do not hard-code fragile thumbnail paths.
_FP = "https://commons.wikimedia.org/wiki/Special:FilePath/"
REMOTE = {
    "galaxy": _FP + "NGC_4414_%28NASA-med%29.jpg?width=720",
    "neuron": _FP + "GFPneuron.png?width=720",
    "starry_night": _FP + "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg?width=900",
}

GRID = 256  # density resolution


# ----------------------------------------------------------------------------- #
# Target preparation: image -> density grid in [0, 1], shape (GRID, GRID)
# ----------------------------------------------------------------------------- #

def _download(url: str, dst: Path) -> Path:
    if dst.exists():
        return dst
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        dst.write_bytes(r.read())
    return dst


def _render_wordmark() -> Image.Image:
    """Render the ' ∇ TorchEBM ' wordmark to a grayscale image via matplotlib."""
    fig = plt.figure(figsize=(6, 6), dpi=GRID // 6)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black")
    ax.axis("off")
    ax.text(
        0.5, 0.60, r"$\nabla$", color="white", fontsize=104,
        ha="center", va="center", fontweight="bold",
    )
    ax.text(
        0.5, 0.30, "TorchEBM", color="white", fontsize=58,
        ha="center", va="center", fontweight="bold", family="DejaVu Sans",
    )
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    return Image.fromarray(buf[..., :3]).convert("L")


def load_density(target: str) -> np.ndarray:
    """Return a (GRID, GRID) float array in [0, 1]; bright = high density."""
    if target == "logo":
        img = _render_wordmark()
    else:
        if target not in REMOTE:
            raise ValueError(f"unknown target {target!r}")
        path = _download(REMOTE[target], CACHE / f"{target}.img")
        img = Image.open(path).convert("L")

    img = img.resize((GRID, GRID), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255.0

    # Starry Night and galaxies read best as-is (bright = subject). A microscopy
    # neuron is bright-on-dark already. The logo is white-on-black already.
    # Normalise contrast and lift a small floor so background still pulls weakly.
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    a = np.power(a, 2.2)  # gamma to deepen background and sharpen the subject
    a = 0.0015 + 0.9985 * a  # tiny floor: background keeps almost no probability mass
    # Flip so image-row-0 (top) maps to +y (math convention, origin lower).
    return np.flipud(a).copy()


def blurred(density: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0:
        return density
    img = Image.fromarray((density * 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=sigma_px))
    out = np.asarray(img, dtype=np.float32) / 255.0
    return (0.0015 + 0.9985 * (out - out.min()) / (out.max() - out.min() + 1e-8))


# ----------------------------------------------------------------------------- #
# TorchEBM energy: E(x) = -log rho(x) with a differentiable bilinear lookup
# ----------------------------------------------------------------------------- #

class ImageDensity(BaseModel):
    r"""Energy from an image density via differentiable bilinear sampling.

    The density grid is held as a (1, 1, H, W) buffer. For a batch of points in
    the box [-1, 1]^2, `grid_sample` performs a differentiable bilinear lookup,
    so autograd supplies the score \nabla_x log rho(x) that the sampler needs.
    """

    def __init__(self, density: np.ndarray, temperature: float = 1.0):
        super().__init__()
        grid = torch.from_numpy(density).float()[None, None]  # (1,1,H,W)
        self.register_buffer("grid", grid)
        self.temperature = float(temperature)

    def set_density(self, density: np.ndarray) -> None:
        self.grid.copy_(torch.from_numpy(density).float()[None, None].to(self.grid))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 2) in [-1, 1]; grid_sample wants (N-as-batch) coords (x, y).
        n = x.shape[0]
        samp = x.view(1, n, 1, 2)
        dens = F.grid_sample(
            self.grid, samp, mode="bilinear",
            padding_mode="border", align_corners=True,
        ).view(n)
        return -torch.log(dens.clamp_min(1e-6)) / self.temperature


# ----------------------------------------------------------------------------- #
# Sampling: annealed coarse-to-fine Langevin transport
# ----------------------------------------------------------------------------- #

def transport(density: np.ndarray, device: str, n_particles: int = 11000):
    """Return a (frames, n_particles, 2) trajectory of the assembling cloud."""
    torch.manual_seed(0)
    model = ImageDensity(blurred(density, 36.0)).to(device)

    # Coarse-to-fine schedule: (blur in px, step_size, noise_scale, steps, thin).
    # The first, heavily blurred stage gives a global basin so particles anywhere
    # in the box are pulled toward the subject; later stages sharpen onto detail
    # while the noise is annealed toward near-zero for a crisp settle.
    stages = [
        (36.0, 9e-3, 0.85, 200, 4),
        (16.0, 5e-3, 0.6, 200, 4),
        (7.0, 2.5e-3, 0.4, 220, 4),
        (3.0, 1.2e-3, 0.24, 240, 4),
        (1.2, 6e-4, 0.12, 260, 4),
        (0.5, 3e-4, 0.05, 300, 4),
    ]

    # Start uniform across the box so the cloud visibly assembles from noise.
    x = (2.0 * torch.rand(n_particles, 2, device=device) - 1.0)
    frames = [x.detach().cpu().numpy().copy()]

    for sigma, step, noise, n_steps, thin in stages:
        model.set_density(blurred(density, sigma))
        sampler = LangevinDynamics(
            model=model, step_size=step, noise_scale=noise, device=device,
        )
        traj = sampler.sample(
            x=x, dim=2, n_steps=n_steps, thin=thin, return_trajectory=True,
        )
        # traj: (n_particles, n_kept, 2) -> per-frame (n_particles, 2)
        traj = traj.clamp_(-1, 1)
        for k in range(traj.shape[1]):
            frames.append(traj[:, k, :].detach().cpu().numpy().copy())
        x = traj[:, -1, :].contiguous()

    return np.stack(frames, axis=0)


# ----------------------------------------------------------------------------- #
# Rendering: dark theme, glowing particles, faint target contour
# ----------------------------------------------------------------------------- #

def render(frames: np.ndarray, density: np.ndarray, out_path: Path, title: str):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "#0b0b0f",
        "savefig.facecolor": "#0b0b0f",
    })
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=170)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_facecolor("#0b0b0f")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # Faint target as a guide rail behind the particles (only the bright subject
    # shows, since the background density is now near zero).
    ax.imshow(
        density, extent=(-1, 1, -1, 1), origin="lower",
        cmap="magma", alpha=0.22, zorder=0, interpolation="bilinear",
    )

    scat = ax.scatter(
        frames[0][:, 0], frames[0][:, 1], s=1.6, c="#ffd27f",
        alpha=0.5, linewidths=0, zorder=2,
    )
    tag = ax.text(
        0.5, 0.045, title, transform=ax.transAxes, color="#f5f5f7",
        ha="center", va="center", fontsize=15, fontweight="bold", alpha=0.92,
    )
    sub = ax.text(
        0.5, 0.012, "Langevin transport into an image density · TorchEBM",
        transform=ax.transAxes, color="#9aa0aa", ha="center", va="center",
        fontsize=8.5,
    )

    n = len(frames)

    def update(i):
        pts = frames[i]
        scat.set_offsets(pts)
        # Particles warm up (brighten) as the cloud settles.
        prog = i / (n - 1)
        scat.set_alpha(0.4 + 0.35 * prog)
        return scat, tag, sub

    anim = FuncAnimation(fig, update, frames=n, interval=40, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=25))
    plt.close(fig)
    print(f"wrote {out_path}  ({n} frames)")


TITLES = {
    "logo": "∇ TorchEBM",
    "galaxy": "NGC 4414",
    "neuron": "Cortical neuron",
    "starry_night": "The Starry Night",
}


def run(target: str, device: str):
    density = load_density(target)
    frames = transport(density, device=device)
    render(frames, density, OUT / f"density_transport_{target}.gif", TITLES[target])


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--target",
        default="logo",
        choices=["logo", "galaxy", "neuron", "starry_night", "all"],
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    targets = ["logo", "galaxy", "neuron", "starry_night"] if args.target == "all" else [args.target]
    for t in targets:
        run(t, args.device)


if __name__ == "__main__":
    main()
