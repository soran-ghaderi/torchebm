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


def make_8gaussians(
    n_samples: int = 2000,
    std: float = 0.02,
    scale: float = 2.0,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Generates samples from the specific '8 Gaussians' mixture.

    Args:
        n_samples (int): Total number of samples to generate.
        std (float): Standard deviation of each Gaussian component.
        scale (float): Scaling factor for the centers (often 2).
        device (Optional[Union[str, torch.device]]): Device to place the tensor on.

    Returns:
        torch.Tensor: Tensor of shape (n_samples, 2).
    """
    centers = (
        np.array(
            [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
                (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            ],
            dtype=np.float32,
        )
        * scale
    )

    return make_gaussian_mixture(
        n_samples=n_samples,
        n_components=8,  # Fixed at 8
        std=std,
        radius=scale,  # Use scale directly (approximation, centers are fixed)
        device=device,
    )


def make_two_moons(
    n_samples: int = 2000,
    noise: float = 0.05,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Generates the 'two moons' dataset.

    Args:
        n_samples (int): Total number of samples.
        noise (float): Standard deviation of Gaussian noise added.
        device (Optional[Union[str, torch.device]]): Device to place the tensor on.

    Returns:
        torch.Tensor: Tensor of shape (n_samples, 2).
    """
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Outer moon
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    # Inner moon
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
    ).T.astype(np.float32)

    # Add noise using torch for potential device efficiency
    tensor_data = _to_tensor(X, device=device)
    tensor_data += torch.randn_like(tensor_data) * noise

    return tensor_data


def make_swiss_roll(
    n_samples: int = 2000,
    noise: float = 0.05,
    arclength: float = 3.0,  # Controls how many rolls (pi*arclength)
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Generates a 2D Swiss roll dataset.

    Args:
        n_samples (int): Number of samples.
        noise (float): Standard deviation of Gaussian noise added.
        arclength (float): Controls the length/tightness of the roll.
        device (Optional[Union[str, torch.device]]): Device to place the tensor on.

    Returns:
        torch.Tensor: Tensor of shape (n_samples, 2).
    """
    t = arclength * np.pi * (1 + 2 * np.random.rand(n_samples))

    x = t * np.cos(t)
    y = t * np.sin(t)

    X = np.vstack((x, y)).T.astype(np.float32)

    # Add noise using torch
    tensor_data = _to_tensor(X, device=device)
    tensor_data += torch.randn_like(tensor_data) * noise

    # Center and scale slightly
    tensor_data = (tensor_data - tensor_data.mean(dim=0)) / (
        tensor_data.std(dim=0).mean() * 2.0
    )

    return tensor_data


def make_circle(
    n_samples: int = 2000,
    noise: float = 0.05,
    radius: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Generates points sampled uniformly on a circle with noise.

    Args:
        n_samples (int): Number of samples.
        noise (float): Standard deviation of Gaussian noise added.
        radius (float): Radius of the circle.
        device (Optional[Union[str, torch.device]]): Device to place the tensor on.

    Returns:
        torch.Tensor: Tensor of shape (n_samples, 2).
    """
    angles = 2 * np.pi * np.random.rand(n_samples)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    X = np.vstack((x, y)).T.astype(np.float32)

    # Add noise using torch
    tensor_data = _to_tensor(X, device=device)
    tensor_data += torch.randn_like(tensor_data) * noise

    return tensor_data


def make_checkerboard(
    n_samples: int = 2000,
    range_limit: float = 4.0,
    noise: float = 0.01,  # Small noise to avoid perfect grid
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Generates points in a 2D checkerboard pattern.

    Args:
        n_samples (int): Target number of samples.
        range_limit (float): Defines the square region [-lim, lim] x [-lim, lim].
        noise (float): Small Gaussian noise added to points.
        device (Optional[Union[str, torch.device]]): Device to place the tensor on.

    Returns:
        torch.Tensor: Tensor of shape (n_samples, 2).
    """
    collected_samples = []
    target = n_samples
    batch_size = max(1000, n_samples // 10)  # Generate in batches for efficiency

    while len(collected_samples) < target:
        # Generate points uniformly in the range
        x = np.random.uniform(-range_limit, range_limit, size=batch_size)
        y = np.random.uniform(-range_limit, range_limit, size=batch_size)

        # Checkerboard condition: floor(x) + floor(y) is odd
        # Using ceil or floor changes the exact pattern but maintains the checkerboard
        keep = (np.floor(x) + np.floor(y)) % 2 != 0
        valid_points = np.vstack((x[keep], y[keep])).T.astype(np.float32)

        needed = target - len(collected_samples)
        collected_samples.extend(valid_points[:needed])

    X = np.array(collected_samples, dtype=np.float32)
    tensor_data = _to_tensor(X, device=device)
    tensor_data += torch.randn_like(tensor_data) * noise  # Add slight noise

    return tensor_data


def make_pinwheel(
    n_samples: int = 2000,
    n_classes: int = 5,
    noise: float = 0.05,
    scale_radius: float = 0.3,
    scale_angle: float = 0.4,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Generates the pinwheel dataset.

    Args:
        n_samples (int): Number of samples.
        n_classes (int): Number of 'blades' in the pinwheel.
        noise (float): Standard deviation of Gaussian noise added.
        scale_radius (float): Scaling factor for the radius.
        scale_angle (float): Scaling factor for the angle spread within a class.
        device (Optional[Union[str, torch.device]]): Device to place the tensor on.

    Returns:
        torch.Tensor: Tensor of shape (n_samples, 2).
    """
    data = np.zeros((n_samples, 2), dtype=np.float32)
    for i in range(n_samples):
        class_idx = np.random.randint(n_classes)
        # Use np.sqrt(rand) for radial distribution, rand for angular offsets
        radius = np.sqrt(np.random.rand()) * scale_radius
        base_angle = np.pi / n_classes * (class_idx + np.random.rand() * scale_angle)
        angle_offset = np.random.rand() * (2 * np.pi / n_classes) + class_idx * (
            2 * np.pi / n_classes
        )

        total_angle = base_angle + angle_offset
        data[i, 0] = radius * np.sin(total_angle)
        data[i, 1] = radius * np.cos(total_angle)

    # Add noise using torch
    tensor_data = _to_tensor(data, device=device)
    tensor_data += torch.randn_like(tensor_data) * noise

    # Center and scale
    tensor_data = (tensor_data - tensor_data.mean(dim=0)) / (tensor_data.std() * 1.5)

    return tensor_data


def make_2d_grid(
    n_samples_per_dim: int = 10,
    range_limit: float = 1.0,
    noise: float = 0.01,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Generates points on a 2D grid within [-range_limit, range_limit].

    Args:
        n_samples_per_dim (int): Number of points along each dimension. Total samples = n^2.
        range_limit (float): Defines the square region [-lim, lim] x [-lim, lim].
        noise (float): Standard deviation of Gaussian noise added to grid points.
        device (Optional[Union[str, torch.device]]): Device to place the tensor on.

    Returns:
        torch.Tensor: Tensor of shape (n_samples_per_dim**2, 2).
    """
    x_coords = np.linspace(-range_limit, range_limit, n_samples_per_dim)
    y_coords = np.linspace(-range_limit, range_limit, n_samples_per_dim)
    xv, yv = np.meshgrid(x_coords, y_coords)
    X = np.vstack([xv.flatten(), yv.flatten()]).T.astype(np.float32)

    tensor_data = _to_tensor(X, device=device)
    tensor_data += torch.randn_like(tensor_data) * noise

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

data_moons = make_two_moons(n_samples=1000, noise=0.05)
data_checker = make_checkerboard(n_samples=10000, range_limit=3)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(data_moons[:, 0], data_moons[:, 1], s=2)
plt.title("Two Moons")
plt.subplot(1, 2, 2)
plt.scatter(data_checker[:, 0], data_checker[:, 1], s=2)
plt.title("Checkerboard")
plt.show()
