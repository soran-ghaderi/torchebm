import torch
import numpy as np
from typing import Optional, Union


def make_gaussian_mixture(
    n_samples: int = 1000,
    n_components: int = 8,
    std: float = 0.05,
    radius: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Generates samples from a 2D Gaussian mixture arranged in a circle."""
    thetas = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
    centers = np.array(
        [(radius * np.cos(t), radius * np.sin(t)) for t in thetas], dtype=np.float32
    )
    centers_torch = torch.from_numpy(centers)

    data = []
    for _ in range(n_samples):
        comp_idx = np.random.randint(0, n_components)
        # Sample from N(centers[comp_idx], std*I)
        point = torch.randn(2) * std + centers_torch[comp_idx]
        data.append(point)

    tensor_data = torch.stack(data)
    if device:
        tensor_data = tensor_data.to(torch.device(device))
    return tensor_data


import matplotlib.pyplot as plt

mixture_data = make_gaussian_mixture(n_samples=500, n_components=4, std=0.1)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(mixture_data[:, 0], mixture_data[:, 1], s=5)
plt.title("Gaussian Mixture")
plt.show()
