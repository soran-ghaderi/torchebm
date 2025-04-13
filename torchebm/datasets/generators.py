import torch
import numpy as np
from typing import Optional, Union, Tuple


def _to_tensor(
    data: np.ndarray,
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    tensor = torch.from_numpy(data).to(dtype)
    if device:
        tensor = tensor.to(torch.device(device))
    return tensor


def make_gaussian_mixture(
    n_samples: int = 2000,
    n_components: int = 8,
    std: float = 0.05,
    radius: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Generates samples from a 2D Gaussian mixture arranged uniformly in a circle.

    Args:
        n_samples (int): Total number of samples to generate.
        n_components (int): Number of Gaussian components (modes).
        std (float): Standard deviation of each Gaussian component.
        radius (float): Radius of the circle on which the centers lie.
        device (Optional[Union[str, torch.device]]): Device to place the tensor on.

    Returns:
        torch.Tensor: Tensor of shape (n_samples, 2).
    """
    if n_components <= 0:
        raise ValueError("n_components must be positive")

    thetas = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
    centers = np.array(
        [(radius * np.cos(t), radius * np.sin(t)) for t in thetas], dtype=np.float32
    )
    centers_torch = torch.from_numpy(centers)  # Keep centers on CPU for indexing

    data = torch.empty(n_samples, 2)  # Pre-allocate tensor
    samples_per_component = n_samples // n_components
    remainder = n_samples % n_components

    start_idx = 0
    for i in range(n_components):
        num = samples_per_component + (1 if i < remainder else 0)
        if num == 0:
            continue
        end_idx = start_idx + num
        # Sample noise directly on the target device if possible
        noise = torch.randn(num, 2, device=device) * std
        data[start_idx:end_idx] = (
            centers_torch[i].to(device) + noise
        )  # Move center to device here
        start_idx = end_idx

    # Shuffle the data to mix components
    data = data[torch.randperm(n_samples)]

    return data


def make_two_moons(
    n_samples: int = 1000,
    noise: float = 0.1,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Generates the 'two moons' dataset."""
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
    ).T
    X = X + np.random.normal(scale=noise, size=X.shape)

    tensor_data = torch.from_numpy(X).float()
    if device:
        tensor_data = tensor_data.to(torch.device(device))
    return tensor_data


import matplotlib.pyplot as plt

mixture_data = make_gaussian_mixture(n_samples=500, n_components=4, std=0.1)
moons_data = make_two_moons(n_samples=500, noise=0.05)

# Quick plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(mixture_data[:, 0], mixture_data[:, 1], s=5)
plt.title("Gaussian Mixture")
plt.subplot(1, 2, 2)
plt.scatter(moons_data[:, 0], moons_data[:, 1], s=5)
plt.title("Two Moons")
plt.show()
