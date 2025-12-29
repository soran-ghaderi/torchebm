r"""Image processing utilities for TorchEBM."""

from typing import Optional

import numpy as np
from PIL import Image


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    r"""Center crop and resize image to target size.

    Args:
        pil_image: PIL image to crop.
        image_size: Target size for square crop.

    Returns:
        Center-cropped PIL image.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def create_npz_from_sample_folder(sample_dir: str, num: int = 50000) -> str:
    r"""Build .npz file from folder of PNG samples.

    Args:
        sample_dir: Directory containing numbered PNG files.
        num: Number of samples to include.

    Returns:
        Path to created .npz file.
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


__all__ = [
    "center_crop_arr",
    "create_npz_from_sample_folder",
]
