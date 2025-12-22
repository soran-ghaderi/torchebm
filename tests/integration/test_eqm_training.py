r"""
Validation test for EqM training using the TorchEBM API.

This script trains an EqM model on CIFAR-10 and generates samples to validate
the implementation matches the original paper's approach.

Usage:
    python tests/test_eqm_training.py [--epochs 10] [--batch-size 64]
"""

import argparse
import os
from copy import deepcopy
from time import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from torchebm.losses import EquilibriumMatchingLoss
from torchebm.samplers import FlowSampler


# ─────────────────────────────────────────────────────────────────────────────
# Simple UNet-style model for image generation (smaller than full EqM for testing)
# ─────────────────────────────────────────────────────────────────────────────


class SinusoidalPositionEmbeddings(nn.Module):
    r"""Sinusoidal time embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    r"""Residual block with time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(h + self.shortcut(x))


class SimpleUNet(nn.Module):
    r"""Simple UNet for CIFAR-10 velocity prediction."""

    def __init__(
        self, in_channels: int = 3, base_channels: int = 64, time_emb_dim: int = 128
    ):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder
        self.enc1 = ResBlock(in_channels, base_channels, time_emb_dim)
        self.enc2 = ResBlock(base_channels, base_channels * 2, time_emb_dim)
        self.enc3 = ResBlock(base_channels * 2, base_channels * 4, time_emb_dim)

        self.down1 = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(
            base_channels * 2, base_channels * 2, 3, stride=2, padding=1
        )
        self.down3 = nn.Conv2d(
            base_channels * 4, base_channels * 4, 3, stride=2, padding=1
        )

        # Middle
        self.mid = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 4, 4, stride=2, padding=1
        )
        self.dec3 = ResBlock(base_channels * 8, base_channels * 4, time_emb_dim)

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, 4, stride=2, padding=1
        )
        self.dec2 = ResBlock(base_channels * 4, base_channels * 2, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, 4, stride=2, padding=1
        )
        self.dec1 = ResBlock(base_channels * 2, base_channels, time_emb_dim)

        self.out_conv = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        t_emb = self.time_emb(t)

        # Encoder
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        e3 = self.enc3(self.down2(e2), t_emb)

        # Middle
        m = self.mid(self.down3(e3), t_emb)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)

        return self.out_conv(d1)


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.999):
    r"""Update EMA model weights."""
    for ema_params, params in zip(ema_model.parameters(), model.parameters()):
        ema_params.data.mul_(decay).add_(params.data, alpha=1 - decay)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    r"""Denormalize images from [-1, 1] to [0, 1]."""
    return (x + 1) / 2


def count_parameters(model: nn.Module) -> int:
    r"""Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────


def train(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
        ]
    )

    dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Dataset: CIFAR-10, {len(dataset)} images")

    # Model
    model = SimpleUNet(
        in_channels=3,
        base_channels=args.base_channels,
        time_emb_dim=args.time_emb_dim,
    ).to(device)

    ema_model = deepcopy(model).to(device)
    ema_model.eval()

    print(f"Model parameters: {count_parameters(model):,}")

    # Loss function using TorchEBM API
    loss_fn = EquilibriumMatchingLoss(
        model,
        prediction=args.prediction,
        interpolant=args.interpolant,
        loss_weight=args.loss_weight,
        train_eps=args.train_eps,
        apply_dispersion=args.dispersion,
        dispersion_weight=0.5,
        device=device,
    )
    print(f"EqM Loss: {loss_fn}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    # Training loop
    losses = []
    global_step = 0
    start_time = time()

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(device)

            # Compute loss
            loss = loss_fn(x)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update EMA
            update_ema(ema_model, model, decay=args.ema_decay)

            # Logging
            epoch_loss += loss.item()
            losses.append(loss.item())
            global_step += 1

            if batch_idx % args.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        elapsed = time() - start_time
        print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}, elapsed={elapsed:.1f}s")

        # Generate samples every N epochs
        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            generate_samples(
                ema_model,
                args,
                device,
                epoch + 1,
                os.path.join(args.output_dir, f"samples_epoch_{epoch + 1:03d}.png"),
            )

    # Save final model
    checkpoint = {
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "args": args,
    }
    torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint.pt"))
    print(f"Saved checkpoint to {args.output_dir}/checkpoint.pt")

    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    # Smoothed loss
    window = min(100, len(losses) // 10)
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        plt.plot(smoothed)
        plt.xlabel("Step")
        plt.ylabel("Smoothed Loss")
        plt.title(f"Smoothed Loss (window={window})")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_loss.png"), dpi=150)
    print(f"Saved training plot to {args.output_dir}/training_loss.png")


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    args,
    device: torch.device,
    epoch: int,
    save_path: str,
):
    r"""Generate and save samples."""
    model.eval()

    sampler = FlowSampler(
        model=model,
        interpolant=args.interpolant,
        prediction=args.prediction,
        train_eps=args.train_eps,
        sample_eps=args.sample_eps,
        device=device,
    )

    # Generate samples
    n_samples = 64
    z = torch.randn(n_samples, 3, 32, 32, device=device)

    print(f"  Generating {n_samples} samples...")

    # ODE sampling
    samples_ode = sampler.sample_ode(
        z, num_steps=args.sample_steps, method=args.ode_method
    )
    samples_ode = denormalize(samples_ode.clamp(-1, 1))

    # SDE sampling (optional, can be slower)
    if args.sde_samples:
        samples_sde = sampler.sample_sde(
            z,
            num_steps=args.sample_steps * 2,
            method="euler",
            diffusion_form="SBDM",
            last_step="Mean",
        )
        samples_sde = denormalize(samples_sde.clamp(-1, 1))

    # Create grid and save
    grid = make_grid(samples_ode, nrow=8, padding=2)
    save_image(grid, save_path)
    print(f"  Saved samples to {save_path}")

    # Also save SDE samples if enabled
    if args.sde_samples:
        sde_path = save_path.replace(".png", "_sde.png")
        grid_sde = make_grid(samples_sde, nrow=8, padding=2)
        save_image(grid_sde, sde_path)
        print(f"  Saved SDE samples to {sde_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="EqM Training Validation Test")

    # Data
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument(
        "--output-dir", type=str, default="./eqm_test_output", help="Output directory"
    )

    # Model
    parser.add_argument(
        "--base-channels", type=int, default=64, help="Base channels for UNet"
    )
    parser.add_argument(
        "--time-emb-dim", type=int, default=128, help="Time embedding dimension"
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="EMA decay")

    # EqM config
    parser.add_argument(
        "--prediction",
        type=str,
        default="velocity",
        choices=["velocity", "score", "noise"],
    )
    parser.add_argument(
        "--interpolant", type=str, default="linear", choices=["linear", "cosine", "vp"]
    )
    parser.add_argument(
        "--loss-weight",
        type=str,
        default=None,
        choices=[None, "velocity", "likelihood"],
    )
    parser.add_argument("--train-eps", type=float, default=0.001)
    parser.add_argument("--sample-eps", type=float, default=0.001)
    parser.add_argument(
        "--dispersion", action="store_true", help="Enable dispersive loss"
    )

    # Sampling
    parser.add_argument("--sample-steps", type=int, default=50, help="Sampling steps")
    parser.add_argument(
        "--ode-method", type=str, default="euler", choices=["euler", "heun", "dopri5"]
    )
    parser.add_argument(
        "--sde-samples", action="store_true", help="Also generate SDE samples"
    )

    # Logging
    parser.add_argument("--log-every", type=int, default=50, help="Log every N batches")
    parser.add_argument(
        "--sample-every", type=int, default=5, help="Sample every N epochs"
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
