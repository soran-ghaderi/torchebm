r"""
Comprehensive validation tests for EqM training using TorchEBM API.

Tests:
1. 2D Gaussian mixture model (quick validation, ~1min)
2. 2D Swiss roll (quick validation, ~1min)
3. CIFAR-10 (full validation, 10+ epochs recommended)

Usage:
    # Quick 2D validation (default)
    python tests/test_eqm_validation.py

    # All interpolants comparison on 2D
    python tests/test_eqm_validation.py --compare-interpolants

    # CIFAR-10 training
    python tests/test_eqm_validation.py --dataset cifar10 --epochs 50
"""

import argparse
import os
from copy import deepcopy
from time import time
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from torchebm.losses import EquilibriumMatchingLoss
from torchebm.samplers import FlowSampler


# ─────────────────────────────────────────────────────────────────────────────
# 2D Data Distributions
# ─────────────────────────────────────────────────────────────────────────────


def sample_gaussian_mixture(
    n: int,
    device: torch.device,
    n_modes: int = 8,
    radius: float = 3.0,
    std: float = 0.3,
) -> torch.Tensor:
    r"""Sample from Gaussian mixture arranged in a circle."""
    mode_idx = torch.randint(0, n_modes, (n,), device=device)
    angles = 2 * np.pi * mode_idx.float() / n_modes
    centers = torch.stack(
        [radius * torch.cos(angles), radius * torch.sin(angles)], dim=1
    )
    return centers + std * torch.randn(n, 2, device=device)


def sample_two_moons(n: int, device: torch.device, noise: float = 0.1) -> torch.Tensor:
    r"""Sample from two moons distribution."""
    n_per_moon = n // 2
    # First moon
    t1 = torch.linspace(0, np.pi, n_per_moon, device=device)
    x1 = torch.stack([torch.cos(t1), torch.sin(t1)], dim=1)
    # Second moon (shifted)
    t2 = torch.linspace(0, np.pi, n - n_per_moon, device=device)
    x2 = torch.stack([1 - torch.cos(t2), 1 - torch.sin(t2) - 0.5], dim=1)
    data = torch.cat([x1, x2], dim=0)
    return data + noise * torch.randn_like(data)


def sample_swiss_roll(
    n: int, device: torch.device, noise: float = 0.05
) -> torch.Tensor:
    r"""Sample 2D Swiss roll (projection)."""
    t = 1.5 * np.pi * (1 + 2 * torch.rand(n, device=device))
    x = t * torch.cos(t)
    y = t * torch.sin(t)
    data = torch.stack([x, y], dim=1) / 10  # Normalize
    return data + noise * torch.randn_like(data)


def sample_checkerboard(
    n: int, device: torch.device, grid_size: int = 4
) -> torch.Tensor:
    r"""Sample from checkerboard pattern."""
    data = torch.rand(n * 10, 2, device=device) * grid_size
    mask = (data[:, 0].long() + data[:, 1].long()) % 2 == 0
    data = data[mask][:n]
    return data / grid_size * 4 - 2  # Normalize to [-2, 2]


# ─────────────────────────────────────────────────────────────────────────────
# Simple 2D Networks
# ─────────────────────────────────────────────────────────────────────────────


class MLP2D(nn.Module):
    r"""Simple MLP for 2D flow matching with time conditioning."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        layers = [nn.Linear(2, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

        # Skip connection weight
        self.skip = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        t_emb = self.time_embed(t.unsqueeze(-1))
        h = self.net[0](x)  # Input projection
        h = h + t_emb  # Add time embedding
        for layer in self.net[1:-1]:
            if isinstance(layer, nn.Linear):
                h = layer(h) + t_emb
            else:
                h = layer(h)
        return self.net[-1](h)


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10 UNet (lightweight)
# ─────────────────────────────────────────────────────────────────────────────


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    r"""Lightweight UNet for CIFAR-10."""

    def __init__(self, channels: int = 3, base_dim: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.enc1 = ResBlock(channels, base_dim, time_dim)
        self.enc2 = ResBlock(base_dim, base_dim * 2, time_dim)
        self.enc3 = ResBlock(base_dim * 2, base_dim * 4, time_dim)

        self.down1 = nn.Conv2d(base_dim, base_dim, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 2, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(base_dim * 4, base_dim * 4, 3, stride=2, padding=1)

        # Middle
        self.mid = ResBlock(base_dim * 4, base_dim * 4, time_dim)

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            base_dim * 4, base_dim * 4, 4, stride=2, padding=1
        )
        self.dec3 = ResBlock(base_dim * 8, base_dim * 4, time_dim)
        self.up2 = nn.ConvTranspose2d(
            base_dim * 4, base_dim * 2, 4, stride=2, padding=1
        )
        self.dec2 = ResBlock(base_dim * 4, base_dim * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, 4, stride=2, padding=1)
        self.dec1 = ResBlock(base_dim * 2, base_dim, time_dim)

        self.out = nn.Conv2d(base_dim, channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        t = self.time_embed(t)

        e1 = self.enc1(x, t)
        e2 = self.enc2(self.down1(e1), t)
        e3 = self.enc3(self.down2(e2), t)

        m = self.mid(self.down3(e3), t)

        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1), t)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t)

        return self.out(d1)


# ─────────────────────────────────────────────────────────────────────────────
# Training Utilities
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float = 0.999):
    for p_ema, p in zip(ema.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 2D Experiment
# ─────────────────────────────────────────────────────────────────────────────


def run_2d_experiment(
    data_fn,
    data_name: str,
    interpolant: str = "linear",
    prediction: str = "velocity",
    device: torch.device = torch.device("cpu"),
    n_steps: int = 3000,
    batch_size: int = 512,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    save_dir: str = "./eqm_2d_output",
) -> Tuple[nn.Module, list]:
    r"""Run 2D experiment and return trained model + losses."""

    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"2D Experiment: {data_name}")
    print(f"Interpolant: {interpolant}, Prediction: {prediction}")
    print(f"{'='*60}")

    # Model
    model = MLP2D(hidden_dim=hidden_dim, num_layers=4).to(device)
    print(f"Model parameters: {count_params(model):,}")

    # EqM Loss
    train_eps = 1e-5
    loss_fn = EquilibriumMatchingLoss(
        model,
        prediction=prediction,
        interpolant=interpolant,
        train_eps=train_eps,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training
    losses = []
    start = time()

    pbar = tqdm(range(n_steps), desc="Training")
    for step in pbar:
        x = data_fn(batch_size, device)
        loss = loss_fn(x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    elapsed = time() - start
    print(f"Training completed in {elapsed:.1f}s")

    # Generate samples
    model.eval()
    sampler = FlowSampler(
        model=model,
        interpolant=interpolant,
        prediction=prediction,
        train_eps=train_eps,
        sample_eps=train_eps,
        device=device,
    )

    z = torch.randn(2000, 2, device=device)
    with torch.no_grad():
        samples = sampler.sample_ode(z, num_steps=100, method="heun")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Real data
    real_data = data_fn(2000, device).cpu().numpy()
    axes[0].scatter(real_data[:, 0], real_data[:, 1], alpha=0.5, s=5)
    axes[0].set_title(f"Real Data: {data_name}")
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-5, 5)

    # Generated samples
    samples_np = samples.cpu().numpy()
    axes[1].scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5, s=5, c="orange")
    axes[1].set_title(f"Generated ({interpolant}, {prediction})")
    axes[1].set_xlim(-5, 5)
    axes[1].set_ylim(-5, 5)

    # Training loss
    axes[2].plot(losses)
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Training Loss")
    axes[2].set_yscale("log")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{data_name}_{interpolant}_{prediction}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")

    return model, losses


def compare_interpolants(device: torch.device, save_dir: str = "./eqm_2d_output"):
    r"""Compare different interpolant types on 2D data."""

    print("\n" + "=" * 60)
    print("INTERPOLANT COMPARISON")
    print("=" * 60)

    data_fn = lambda n, dev: sample_gaussian_mixture(n, dev, n_modes=8)
    interpolants = ["linear", "cosine", "vp"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Real data
    real_data = data_fn(2000, device).cpu().numpy()
    axes[0, 0].scatter(real_data[:, 0], real_data[:, 1], alpha=0.5, s=3)
    axes[0, 0].set_title("Real Data (8-mode GMM)")
    axes[0, 0].set_xlim(-5, 5)
    axes[0, 0].set_ylim(-5, 5)
    axes[1, 0].axis("off")

    all_losses = {}

    for i, interp in enumerate(interpolants, start=1):
        print(f"\nTraining with {interp} interpolant...")

        model = MLP2D(hidden_dim=256, num_layers=4).to(device)
        loss_fn = EquilibriumMatchingLoss(
            model,
            prediction="velocity",
            interpolant=interp,
            train_eps=1e-5,
            device=device,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses = []
        for step in tqdm(range(3000), desc=f"{interp}"):
            x = data_fn(512, device)
            loss = loss_fn(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        all_losses[interp] = losses

        # Generate samples
        model.eval()
        sampler = FlowSampler(
            model=model,
            interpolant=interp,
            prediction="velocity",
            train_eps=1e-5,
            sample_eps=1e-5,
            device=device,
        )
        z = torch.randn(2000, 2, device=device)
        with torch.no_grad():
            samples = sampler.sample_ode(z, num_steps=100, method="heun")

        samples_np = samples.cpu().numpy()
        axes[0, i].scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5, s=3)
        axes[0, i].set_title(f"{interp.upper()} Interpolant")
        axes[0, i].set_xlim(-5, 5)
        axes[0, i].set_ylim(-5, 5)

    # Plot all losses
    for interp, losses in all_losses.items():
        axes[1, 1].plot(losses, label=interp, alpha=0.8)
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("Training Loss Comparison")
    axes[1, 1].legend()
    axes[1, 1].set_yscale("log")

    # Smoothed losses
    window = 100
    for interp, losses in all_losses.items():
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        axes[1, 2].plot(smoothed, label=interp, alpha=0.8)
    axes[1, 2].set_xlabel("Step")
    axes[1, 2].set_ylabel("Smoothed Loss")
    axes[1, 2].set_title(f"Smoothed Loss (window={window})")
    axes[1, 2].legend()

    axes[1, 3].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "interpolant_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved comparison to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10 Experiment
# ─────────────────────────────────────────────────────────────────────────────


def run_cifar10_experiment(
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 128,
    base_dim: int = 64,
    lr: float = 2e-4,
    interpolant: str = "linear",
    prediction: str = "velocity",
    save_dir: str = "./eqm_cifar_output",
    sample_every: int = 10,
):
    r"""Run CIFAR-10 training experiment."""
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid, save_image

    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("CIFAR-10 EXPERIMENT")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"Interpolant: {interpolant}, Prediction: {prediction}")
    print("=" * 60)

    # Data
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} images")

    # Model
    model = SimpleUNet(channels=3, base_dim=base_dim).to(device)
    ema = deepcopy(model).to(device)
    ema.eval()
    print(f"Model parameters: {count_params(model):,}")

    # Loss & Optimizer
    train_eps = 1e-3
    loss_fn = EquilibriumMatchingLoss(
        model,
        prediction=prediction,
        interpolant=interpolant,
        train_eps=train_eps,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loader)
    )

    # Training
    losses = []
    start = time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, _ in pbar:
            x = x.to(device)
            loss = loss_fn(x)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            update_ema(ema, model, decay=0.9999)

            epoch_loss += loss.item()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(
            f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Generate samples
        if (epoch + 1) % sample_every == 0 or epoch == epochs - 1:
            ema.eval()
            sampler = FlowSampler(
                model=ema,
                interpolant=interpolant,
                prediction=prediction,
                train_eps=train_eps,
                sample_eps=train_eps,
                device=device,
            )

            z = torch.randn(64, 3, 32, 32, device=device)
            with torch.no_grad():
                samples = sampler.sample_ode(z, num_steps=50, method="euler")

            samples = (samples.clamp(-1, 1) + 1) / 2
            grid = make_grid(samples, nrow=8, padding=2)
            save_path = os.path.join(save_dir, f"samples_epoch_{epoch+1:03d}.png")
            save_image(grid, save_path)
            print(f"  Saved samples to {save_path}")

    elapsed = time() - start
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")

    # Save checkpoint
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "interpolant": interpolant,
            "prediction": prediction,
            "train_eps": train_eps,
            "losses": losses,
        },
        os.path.join(save_dir, "checkpoint.pt"),
    )

    # Plot losses
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    window = min(500, len(losses) // 10)
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        plt.plot(smoothed)
        plt.xlabel("Step")
        plt.ylabel("Smoothed Loss")
        plt.title(f"Smoothed Loss (window={window})")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_loss.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="EqM Validation Tests")
    parser.add_argument("--dataset", type=str, default="2d", choices=["2d", "cifar10"])
    parser.add_argument(
        "--data-2d",
        type=str,
        default="gmm",
        choices=["gmm", "moons", "swiss", "checker"],
    )
    parser.add_argument(
        "--interpolant", type=str, default="linear", choices=["linear", "cosine", "vp"]
    )
    parser.add_argument(
        "--prediction",
        type=str,
        default="velocity",
        choices=["velocity", "score", "noise"],
    )
    parser.add_argument("--compare-interpolants", action="store_true")

    # 2D args
    parser.add_argument("--steps-2d", type=int, default=3000)
    parser.add_argument("--hidden-dim", type=int, default=256)

    # CIFAR args
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--base-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--sample-every", type=int, default=10)

    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.dataset == "2d":
        save_dir = args.output_dir or "./eqm_2d_output"
        os.makedirs(save_dir, exist_ok=True)

        if args.compare_interpolants:
            compare_interpolants(device, save_dir)
        else:
            data_fns = {
                "gmm": lambda n, d: sample_gaussian_mixture(n, d),
                "moons": lambda n, d: sample_two_moons(n, d),
                "swiss": lambda n, d: sample_swiss_roll(n, d),
                "checker": lambda n, d: sample_checkerboard(n, d),
            }

            run_2d_experiment(
                data_fn=data_fns[args.data_2d],
                data_name=args.data_2d,
                interpolant=args.interpolant,
                prediction=args.prediction,
                device=device,
                n_steps=args.steps_2d,
                hidden_dim=args.hidden_dim,
                save_dir=save_dir,
            )

    elif args.dataset == "cifar10":
        save_dir = args.output_dir or "./eqm_cifar_output"
        run_cifar10_experiment(
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            base_dim=args.base_dim,
            lr=args.lr,
            interpolant=args.interpolant,
            prediction=args.prediction,
            save_dir=save_dir,
            sample_every=args.sample_every,
        )


if __name__ == "__main__":
    main()
