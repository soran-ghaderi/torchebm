r"""Utilities for EqM training and sampling."""

from collections import OrderedDict
from typing import Optional, Dict, Any, Union
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999) -> None:
    r"""Update EMA model parameters.

    Args:
        ema_model: Exponential moving average model
        model: Current model
        decay: EMA decay rate
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name in ema_params:
            ema_param = ema_params[name]
            if ema_param.device != param.device:
                ema_param.data = ema_param.data.to(param.device)
            ema_param.mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model: nn.Module, flag: bool = True) -> None:
    r"""Set requires_grad flag for all model parameters.

    Args:
        model: Model to modify
        flag: Whether parameters require gradients
    """
    for p in model.parameters():
        p.requires_grad = flag


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    r"""Center crop and resize image to target size.

    Implementation from ADM for preprocessing.

    Args:
        pil_image: PIL image to crop
        image_size: Target size for square crop

    Returns:
        Center-cropped PIL image
    """
    # Downsample if image is too large
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    # Scale to have minimum dimension equal to image_size
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    # Center crop to square
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def create_npz_from_sample_folder(sample_dir: str, num: int = 50000) -> str:
    r"""Build .npz file from folder of PNG samples.

    Args:
        sample_dir: Directory containing numbered PNG files
        num: Number of samples to include

    Returns:
        Path to created .npz file
    """
    samples = []
    for i in range(num):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)

    samples = np.stack(samples)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}]")
    return npz_path


def save_checkpoint(
    model: nn.Module,
    ema_model: Optional[nn.Module],
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: str,
    args: Optional[Dict[str, Any]] = None,
) -> str:
    r"""Save training checkpoint.

    Args:
        model: Model to save
        ema_model: EMA model (optional)
        optimizer: Optimizer state
        step: Current training step
        checkpoint_dir: Directory for checkpoints
        args: Additional arguments to save

    Returns:
        Path to saved checkpoint
    """
    checkpoint = {
        "model": (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        ),
        "opt": optimizer.state_dict(),
        "step": step,
    }

    if ema_model is not None:
        checkpoint["ema"] = ema_model.state_dict()

    if args is not None:
        checkpoint["args"] = args

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/{step:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    ema_model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    r"""Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        ema_model: EMA model to load (optional)
        optimizer: Optimizer to load state (optional)
        device: Device to map tensors to

    Returns:
        Dictionary with checkpoint contents
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model
    if "model" in checkpoint:
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])

    # Load EMA if provided
    if ema_model is not None and "ema" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema"])

    # Load optimizer if provided
    if optimizer is not None and "opt" in checkpoint:
        optimizer.load_state_dict(checkpoint["opt"])

    return checkpoint


def get_vae_encode_decode():
    r"""Get VAE encoder/decoder functions for latent space operations.

    Returns:
        Tuple of (encode_fn, decode_fn)
    """
    try:
        from diffusers.models import AutoencoderKL
    except ImportError:
        raise ImportError(
            "diffusers required for VAE. Install with: pip install diffusers"
        )

    def get_vae(vae_type: str = "ema", device: torch.device = torch.device("cuda")):
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_type}")
        vae = vae.to(device)
        vae.eval()
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        return vae

    @torch.no_grad()
    def encode_fn(
        x: torch.Tensor, vae: nn.Module, chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        r"""Encode images to latent space.

        Args:
            x: Images in [-1, 1] range, shape (B, C, H, W)
            vae: VAE model
            chunk_size: Process in chunks to avoid OOM. If None, processes entire batch.

        Returns:
            Encoded latents scaled by 0.18215
        """
        if chunk_size is None or x.size(0) <= chunk_size:
            return vae.encode(x).latent_dist.sample().mul_(0.18215)

        latents = []
        for i in range(0, x.size(0), chunk_size):
            chunk = x[i : i + chunk_size]
            latent_chunk = vae.encode(chunk).latent_dist.sample().mul_(0.18215)
            latents.append(latent_chunk)
            torch.cuda.empty_cache()
        return torch.cat(latents, dim=0)

    @torch.no_grad()
    def decode_fn(
        z: torch.Tensor, vae: nn.Module, chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        r"""Decode latents to images.

        Args:
            z: Scaled latents, shape (B, C, H, W)
            vae: VAE model
            chunk_size: Process in chunks to avoid OOM. If None, processes entire batch.

        Returns:
            Decoded images
        """
        z_scaled = z / 0.18215
        if chunk_size is None or z.size(0) <= chunk_size:
            return vae.decode(z_scaled).sample

        images = []
        for i in range(0, z.size(0), chunk_size):
            chunk = z_scaled[i : i + chunk_size]
            image_chunk = vae.decode(chunk).sample
            images.append(image_chunk)
            torch.cuda.empty_cache()
        return torch.cat(images, dim=0)

    return get_vae, encode_fn, decode_fn


def parse_transport_args(parser):
    r"""Add transport-related arguments to parser.

    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        "--path-type",
        type=str,
        default="Linear",
        choices=["Linear", "GVP", "VP"],
        help="Type of interpolation path",
    )
    parser.add_argument(
        "--prediction",
        type=str,
        default="velocity",
        choices=["velocity", "score", "noise"],
        help="Model prediction type",
    )
    parser.add_argument(
        "--loss-weight",
        type=str,
        default=None,
        choices=[None, "velocity", "likelihood"],
        help="Loss weighting scheme",
    )
    parser.add_argument(
        "--train-eps", type=float, default=None, help="Training epsilon for stability"
    )
    parser.add_argument(
        "--sample-eps", type=float, default=None, help="Sampling epsilon for stability"
    )


class WandbLogger:
    r"""Simple wandb logging wrapper."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        if enabled:
            try:
                import wandb

                self.wandb = wandb
            except ImportError:
                print("wandb not installed, disabling logging")
                self.enabled = False

    def initialize(self, config: Dict[str, Any], entity: str, project: str, name: str):
        r"""Initialize wandb run."""
        if self.enabled:
            self.wandb.init(config=config, entity=entity, project=project, name=name)

    def log(self, metrics: Dict[str, Any], step: int):
        r"""Log metrics to wandb."""
        if self.enabled:
            self.wandb.log(metrics, step=step)

    def finish(self):
        r"""Finish wandb run."""
        if self.enabled:
            self.wandb.finish()


__all__ = [
    "update_ema",
    "requires_grad",
    "center_crop_arr",
    "create_npz_from_sample_folder",
    "save_checkpoint",
    "load_checkpoint",
    "get_vae_encode_decode",
    "parse_transport_args",
    "WandbLogger",
]
