r"""Dataset Generators Module."""

import warnings

import torch
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple

# --- Utility Function --- (Keep as is or move if desired)


def _to_tensor(
    data: np.ndarray,
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Converts a NumPy array to a PyTorch tensor."""
    tensor = torch.from_numpy(data).to(dtype)
    if device:
        tensor = tensor.to(torch.device(device))
    return tensor


# --- Base Class ---


class BaseSyntheticDataset(Dataset, ABC):
    """
    Abstract base class for generating 2D synthetic datasets.

    Args:
        n_samples (int): The total number of samples to generate.
        device (Optional[Union[str, torch.device]]): The device to place the tensor on.
        dtype (torch.dtype): The data type for the output tensor.
        seed (Optional[int]): A random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ):
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        self.n_samples = n_samples
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.data: Optional[torch.Tensor] = None  # Data will be stored here
        self._generate()  # Generate data upon initialization

    def _seed_generators(self):
        """Sets the random seeds for numpy and torch if a seed is provided."""
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            # If using CUDA, also seed the CUDA generator
            if torch.cuda.is_available() and (
                isinstance(self.device, torch.device)
                and self.device.type == "cuda"
                or self.device == "cuda"
            ):
                torch.cuda.manual_seed_all(self.seed)  # Seed all GPUs

    @abstractmethod
    def _generate_data(self) -> torch.Tensor:
        """
        Core data generation logic to be implemented by subclasses.
        """
        pass

    def _generate(self):
        """Internal method to handle seeding and call the generation logic."""
        self._seed_generators()
        # Generate data using the subclass implementation
        generated_output = self._generate_data()

        # Ensure it's a tensor and on the correct device/dtype
        if isinstance(generated_output, np.ndarray):
            self.data = _to_tensor(
                generated_output, dtype=self.dtype, device=self.device
            )
        elif isinstance(generated_output, torch.Tensor):
            self.data = generated_output.to(dtype=self.dtype, device=self.device)
        else:
            raise TypeError(
                f"_generate_data must return a NumPy array or PyTorch Tensor, got {type(generated_output)}"
            )

        # Verify batch_shape
        if self.data.shape[0] != self.n_samples:
            warnings.warn(
                f"Generated data has {self.data.shape[0]} samples, but {self.n_samples} were requested. Check generation logic.",
                RuntimeWarning,
            )
            # Optional: adjust self.n_samples or raise error depending on desired strictness
            # self.n_samples = self.data.batch_shape[0]

    def regenerate(self, seed: Optional[int] = None):
        """
        Re-generates the dataset, optionally with a new seed.

        Args:
            seed (Optional[int]): A new random seed. If `None`, the original seed is used.
        """
        if seed is not None:
            self.seed = seed  # Update the seed if a new one is provided
        self._generate()

    def get_data(self) -> torch.Tensor:
        """
        Returns the entire generated dataset as a single tensor.
        """
        if self.data is None:
            # Should not happen if _generate() is called in __init__
            self._generate()
        return self.data

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns the sample at the specified index.
        """
        if self.data is None:
            self._generate()  # Ensure data exists

        if not 0 <= idx < self.n_samples:
            raise IndexError(
                f"Index {idx} out of bounds for dataset with size {self.n_samples}"
            )
        return self.data[idx]

    def __repr__(self) -> str:
        """String representation of the dataset object."""
        params = [f"n_samples={self.n_samples}"]
        # Add specific params from subclasses if desired, e.g. by inspecting self.__dict__
        # Or define __repr__ in subclasses
        return f"{self.__class__.__name__}({', '.join(params)}, device={self.device}, dtype={self.dtype})"


# --- Concrete Dataset Classes ---


class GaussianMixtureDataset(BaseSyntheticDataset):
    """
    Generates a 2D Gaussian mixture dataset with components arranged in a circle.

    Args:
        n_samples (int): The total number of samples.
        n_components (int): The number of Gaussian components (modes).
        std (float): The standard deviation of each Gaussian component.
        radius (float): The radius of the circle on which the centers lie.
        device (Optional[Union[str, torch.device]]): The device for the tensor.
        dtype (torch.dtype): The data type for the tensor.
        seed (Optional[int]): A random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        n_components: int = 8,
        std: float = 0.05,
        radius: float = 1.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ):
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if std < 0:
            raise ValueError("std must be non-negative")
        self.n_components = n_components
        self.std = std
        self.radius = radius
        super().__init__(n_samples=n_samples, device=device, dtype=dtype, seed=seed)

    def _generate_data(self) -> torch.Tensor:
        # Logic from make_gaussian_mixture
        thetas = np.linspace(0, 2 * np.pi, self.n_components, endpoint=False)
        centers = np.array(
            [(self.radius * np.cos(t), self.radius * np.sin(t)) for t in thetas],
            dtype=np.float32,
        )
        # Use torch directly for efficiency and device handling
        centers_torch = torch.from_numpy(centers)  # Keep on CPU for indexing efficiency

        data = torch.empty(self.n_samples, 2, device=self.device, dtype=self.dtype)
        samples_per_component = self.n_samples // self.n_components
        remainder = self.n_samples % self.n_components

        current_idx = 0
        for i in range(self.n_components):
            num = samples_per_component + (1 if i < remainder else 0)
            if num == 0:
                continue
            end_idx = current_idx + num
            # Generate noise directly on target device if possible
            noise = torch.randn(num, 2, device=self.device, dtype=self.dtype) * self.std
            component_center = centers_torch[i].to(device=self.device, dtype=self.dtype)
            data[current_idx:end_idx] = component_center + noise
            current_idx = end_idx

        # Shuffle the data to mix components
        data = data[
            torch.randperm(self.n_samples, device=self.device)
        ]  # Use device-aware permutation
        return data  # Return tensor directly


class EightGaussiansDataset(BaseSyntheticDataset):
    """
    Generates samples from the '8 Gaussians' mixture distribution.

    Args:
        n_samples (int): The total number of samples.
        std (float): The standard deviation of each component.
        scale (float): A scaling factor for the centers of the Gaussians.
        device (Optional[Union[str, torch.device]]): The device for the tensor.
        dtype (torch.dtype): The data type for the tensor.
        seed (Optional[int]): A random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        std: float = 0.02,
        scale: float = 2.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ):
        self.std = std
        self.scale = scale
        # Define the specific 8 centers
        centers_np = (
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
            * self.scale
        )
        self.centers_torch = torch.from_numpy(centers_np)
        self.n_components = 8
        super().__init__(n_samples=n_samples, device=device, dtype=dtype, seed=seed)

    def _generate_data(self) -> torch.Tensor:
        # Similar logic to GaussianMixtureDataset but with fixed centers
        centers_dev = self.centers_torch.to(device=self.device, dtype=self.dtype)

        data = torch.empty(self.n_samples, 2, device=self.device, dtype=self.dtype)
        samples_per_component = self.n_samples // self.n_components
        remainder = self.n_samples % self.n_components

        current_idx = 0
        for i in range(self.n_components):
            num = samples_per_component + (1 if i < remainder else 0)
            if num == 0:
                continue
            end_idx = current_idx + num
            noise = torch.randn(num, 2, device=self.device, dtype=self.dtype) * self.std
            data[current_idx:end_idx] = centers_dev[i] + noise
            current_idx = end_idx

        data = data[torch.randperm(self.n_samples, device=self.device)]
        return data


class TwoMoonsDataset(BaseSyntheticDataset):
    """
    Generates the 'two moons' dataset.

    Args:
        n_samples (int): The total number of samples.
        noise (float): The standard deviation of the Gaussian noise to add.
        device (Optional[Union[str, torch.device]]): The device for the tensor.
        dtype (torch.dtype): The data type for the tensor.
        seed (Optional[int]): A random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        noise: float = 0.05,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ):
        self.noise = noise
        super().__init__(n_samples=n_samples, device=device, dtype=dtype, seed=seed)

    def _generate_data(self) -> np.ndarray:
        # Logic from make_two_moons (using numpy initially is fine here)
        n_samples_out = self.n_samples // 2
        n_samples_in = self.n_samples - n_samples_out

        outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
        outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
        inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
        inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

        X = np.vstack(
            [
                np.append(outer_circ_x, inner_circ_x),
                np.append(outer_circ_y, inner_circ_y),
            ]
        ).T.astype(np.float32)

        # Add noise using torch AFTER converting base batch_shape to tensor
        tensor_data = torch.from_numpy(X)  # Keep on CPU initially for noise addition
        noise_val = torch.randn_like(tensor_data) * self.noise
        tensor_data += noise_val

        # Base class __init__ will handle final _to_tensor conversion for device/dtype
        # Alternatively, add noise directly on the target device:
        # tensor_data = torch.from_numpy(X).to(device=self.device, dtype=self.dtype)
        # tensor_data += torch.randn_like(tensor_data) * self.noise
        # return tensor_data # Return tensor directly if handled here

        return tensor_data  # Return tensor, base class handles device/dtype


class SwissRollDataset(BaseSyntheticDataset):
    """
    Generates a 2D Swiss roll dataset.

    Args:
        n_samples (int): The number of samples.
        noise (float): The standard deviation of the Gaussian noise to add.
        arclength (float): A factor controlling how many rolls the spiral has.
        device (Optional[Union[str, torch.device]]): The device for the tensor.
        dtype (torch.dtype): The data type for the tensor.
        seed (Optional[int]): A random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        noise: float = 0.05,
        arclength: float = 3.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ):
        self.noise = noise
        self.arclength = arclength
        super().__init__(n_samples=n_samples, device=device, dtype=dtype, seed=seed)

    def _generate_data(self) -> torch.Tensor:
        # Logic from make_swiss_roll
        t = self.arclength * np.pi * (1 + 2 * np.random.rand(self.n_samples))
        x = t * np.cos(t)
        y = t * np.sin(t)
        X = np.vstack((x, y)).T.astype(np.float32)

        tensor_data = torch.from_numpy(X)  # CPU tensor initially
        tensor_data += torch.randn_like(tensor_data) * self.noise

        # Center and scale slightly (optional, can be done outside)
        tensor_data = (tensor_data - tensor_data.mean(dim=0)) / (
            tensor_data.std(dim=0).mean()
            * 2.0  # Be careful with division by zero if std is ~0
        )

        return tensor_data  # Return tensor, base class handles device/dtype


class CircleDataset(BaseSyntheticDataset):
    """
    Generates points sampled uniformly on a circle with noise.

    Args:
        n_samples (int): The number of samples.
        noise (float): The standard deviation of the Gaussian noise to add.
        radius (float): The radius of the circle.
        device (Optional[Union[str, torch.device]]): The device for the tensor.
        dtype (torch.dtype): The data type for the tensor.
        seed (Optional[int]): A random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        noise: float = 0.05,
        radius: float = 1.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ):
        self.noise = noise
        self.radius = radius
        super().__init__(n_samples=n_samples, device=device, dtype=dtype, seed=seed)

    def _generate_data(self) -> torch.Tensor:
        # Logic from make_circle
        angles = 2 * np.pi * np.random.rand(self.n_samples)
        x = self.radius * np.cos(angles)
        y = self.radius * np.sin(angles)
        X = np.vstack((x, y)).T.astype(np.float32)

        tensor_data = torch.from_numpy(X)
        tensor_data += torch.randn_like(tensor_data) * self.noise

        return tensor_data


class CheckerboardDataset(BaseSyntheticDataset):
    """
    Generates points in a 2D checkerboard pattern using rejection sampling.

    Args:
        n_samples (int): The target number of samples.
        range_limit (float): Defines the square region `[-lim, lim] x [-lim, lim]`.
        noise (float): Small Gaussian noise added to the points.
        device (Optional[Union[str, torch.device]]): The device for the tensor.
        dtype (torch.dtype): The data type for the tensor.
        seed (Optional[int]): A random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        range_limit: float = 4.0,
        noise: float = 0.01,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ):
        self.range_limit = range_limit
        self.noise = noise
        super().__init__(n_samples=n_samples, device=device, dtype=dtype, seed=seed)

    def _generate_data(self) -> torch.Tensor:
        # Logic from make_checkerboard
        collected_samples = []
        target = self.n_samples
        # Estimate batch size needed (density is ~0.5)
        batch_size = max(1000, int(target * 2.5))  # Generate more than needed per batch

        while len(collected_samples) < target:
            x = np.random.uniform(-self.range_limit, self.range_limit, size=batch_size)
            y = np.random.uniform(-self.range_limit, self.range_limit, size=batch_size)

            keep = (np.floor(x) + np.floor(y)) % 2 != 0
            valid_points = np.vstack((x[keep], y[keep])).T.astype(np.float32)

            needed = target - len(collected_samples)
            collected_samples.extend(valid_points[:needed])  # Add only needed points

        X = np.array(
            collected_samples[:target], dtype=np.float32
        )  # Ensure exact n_samples
        tensor_data = torch.from_numpy(X)
        tensor_data += torch.randn_like(tensor_data) * self.noise

        return tensor_data


class PinwheelDataset(BaseSyntheticDataset):
    """
    Generates the pinwheel dataset with curved blades.

    Args:
        n_samples (int): The total number of samples.
        n_classes (int): The number of 'blades' in the pinwheel.
        noise (float): The standard deviation of the final additive Cartesian noise.
        radial_scale (float): Controls the maximum radius/length of the blades.
        angular_scale (float): Controls the standard deviation of the angle noise (thickness).
        spiral_scale (float): Controls the tightness of the spiral.
        device (Optional[Union[str, torch.device]]): The device for the tensor.
        dtype (torch.dtype): The data type for the tensor.
        seed (Optional[int]): A random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        n_classes: int = 5,
        noise: float = 0.05,
        radial_scale: float = 2.0,
        angular_scale: float = 0.1,
        spiral_scale: float = 5.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ):
        if n_classes <= 0:
            raise ValueError("n_classes must be positive")
        self.n_classes = n_classes
        self.noise = noise
        self.radial_scale = radial_scale
        self.angular_scale = angular_scale
        self.spiral_scale = spiral_scale
        super().__init__(n_samples=n_samples, device=device, dtype=dtype, seed=seed)

    def _generate_data(self) -> torch.Tensor:
        # Logic from make_pinwheel
        all_points_np = []
        samples_per_class = self.n_samples // self.n_classes
        remainder = self.n_samples % self.n_classes

        for class_idx in range(self.n_classes):
            n_class_samples = samples_per_class + (1 if class_idx < remainder else 0)
            if n_class_samples == 0:
                continue

            t = np.sqrt(np.random.rand(n_class_samples))  # Radial density control
            radii = t * self.radial_scale
            base_angle = class_idx * (2 * np.pi / self.n_classes)
            spiral_angle = self.spiral_scale * t
            angle_noise = np.random.randn(n_class_samples) * self.angular_scale
            thetas = base_angle + spiral_angle + angle_noise

            x = radii * np.cos(thetas)
            y = radii * np.sin(thetas)
            all_points_np.append(np.stack([x, y], axis=1))

        data_np = np.concatenate(all_points_np, axis=0).astype(np.float32)
        np.random.shuffle(data_np)  # Shuffle before converting to tensor

        tensor_data = torch.from_numpy(data_np)

        if self.noise > 0:
            tensor_data += torch.randn_like(tensor_data) * self.noise

        return tensor_data


# class GridDataset(BaseSyntheticDataset):
#     """
#     Generates points on a 2D grid.
#
#     Note: The total number of samples will be `n_samples_per_dim` ** 2.
#
#     Args:
#         n_samples_per_dim (int): The number of points along each dimension.
#         range_limit (float): Defines the square region `[-lim, lim] x [-lim, lim]`.
#         noise (float): The standard deviation of the Gaussian noise to add.
#         device (Optional[Union[str, torch.device]]): The device for the tensor.
#         dtype (torch.dtype): The data type for the tensor.
#         seed (Optional[int]): A random seed for reproducibility (primarily affects noise).
#     """
#
#     def __init__(
#         self,
#         n_samples_per_dim: int = 10,
#         range_limit: float = 1.0,
#         noise: float = 0.01,
#         device: Optional[Union[str, torch.device]] = None,
#         dtype: torch.dtype = torch.float32,
#         seed: Optional[int] = None,  # Seed mainly affects noise here
#     ):
#         if n_samples_per_dim <= 0:
#             raise ValueError("n_samples_per_dim must be positive")
#         self.n_samples_per_dim = n_samples_per_dim
#         self.range_limit = range_limit
#         self.noise = noise
#         # Override n_samples for the base class
#         total_samples = n_samples_per_dim * n_samples_per_dim
#         super().__init__(n_samples=total_samples, device=device, dtype=dtype, seed=seed)
#
#     def _generate_data(self) -> torch.Tensor:
#         # Logic from make_2d_grid
#         x_coords = np.linspace(
#             -self.range_limit, self.range_limit, self.n_samples_per_dim
#         )
#         y_coords = np.linspace(
#             -self.range_limit, self.range_limit, self.n_samples_per_dim
#         )
#         xv, yv = np.meshgrid(x_coords, y_coords)
#         X = np.vstack([xv.flatten(), yv.flatten()]).T.astype(np.float32)
#
#         tensor_data = torch.from_numpy(X)
#         # Apply noise using torch random functions (affected by seed set in base class)
#         tensor_data += torch.randn_like(tensor_data) * self.noise
#
#         return tensor_data


class GridDataset(BaseSyntheticDataset):
    """
    Generates points on a 2D grid.

    Note: The total number of samples will be `n_samples_per_dim` ** 2.

    Args:
        n_samples_per_dim (int): The number of points along each dimension.
        range_limit (float): Defines the square region `[-lim, lim] x [-lim, lim]`.
        noise (float): The standard deviation of the Gaussian noise to add.
        device (Optional[Union[str, torch.device]]): The device for the tensor.
        dtype (torch.dtype): The data type for the tensor.
        seed (Optional[int]): A random seed for reproducibility (primarily affects noise).
    """

    def __init__(
        self,
        n_samples_per_dim: int = 10,
        range_limit: float = 1.0,
        noise: float = 0.01,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,  # Seed mainly affects noise here
    ):
        if n_samples_per_dim <= 0:
            raise ValueError("n_samples_per_dim must be positive")
        self.n_samples_per_dim = n_samples_per_dim
        self.range_limit = range_limit
        self.noise = noise
        # Override n_samples for the base class
        total_samples = n_samples_per_dim * n_samples_per_dim
        super().__init__(n_samples=total_samples, device=device, dtype=dtype, seed=seed)

    def _generate_data(self) -> torch.Tensor:
        # Create a more uniform grid spacing
        x_coords = torch.linspace(
            -self.range_limit, self.range_limit, self.n_samples_per_dim
        )
        y_coords = torch.linspace(
            -self.range_limit, self.range_limit, self.n_samples_per_dim
        )

        # Create the grid points
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")

        # Stack the coordinates to form the 2D points
        points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

        # Apply noise if specified
        if self.noise > 0:
            # Set the random seed if provided
            if hasattr(self, "rng"):
                # Use the RNG from the base class
                noise = torch.tensor(
                    self.rng.normal(0, self.noise, size=points.shape),
                    dtype=points.dtype,
                    device=points.device,
                )
            else:
                # Fall back to torch's random generator
                noise = torch.randn_like(points) * self.noise

            points = points + noise

        return points


# r"""
# Dataset Generators Module.
#
# This module provides a collection of functions for generating synthetic datasets commonly used
# in testing and evaluating energy-based models. These generators create various 2D distributions
# with different characteristics, making them ideal for visualization and demonstration purposes.
#
# !!! success "Key Features"
#     - Diverse collection of 2D synthetic distributions.
#     - Configurable sample sizes, noise levels, and distribution parameters.
#     - Device and dtype support for tensor outputs.
#     - Visualization support for generated datasets.
#
# ---
#
# ## Module Components
#
# Functions:
#     make_gaussian_mixture: Generates samples from a 2D Gaussian mixture arranged in a circle.
#     make_8gaussians: Generates samples from a specific 8-component Gaussian mixture.
#     make_two_moons: Generates the classic "two moons" dataset.
#     make_swiss_roll: Generates a 2D Swiss roll dataset.
#     make_circle: Generates points sampled uniformly on a circle with noise.
#     make_checkerboard: Generates points in a 2D checkerboard pattern.
#     make_pinwheel: Generates the pinwheel dataset with specified number of "blades".
#     make_2d_grid: Generates points on a regular 2D grid with optional noise.
#
# ---
#
# ## Usage Example
#
# !!! example "Generating and Visualizing Datasets"
#     ```python
#     from torchebm.datasets.generators import make_two_moons, make_gaussian_mixture
#     import matplotlib.pyplot as plt
#     import torch
#
#     # Generate 1000 samples from the Two Moons distribution
#     moons_data = make_two_moons(n_samples=1000, noise=0.05)
#
#     # Generate 500 samples from a 4-component Gaussian mixture
#     mixture_data = make_gaussian_mixture(n_samples=500, n_components=4, std=0.1)
#
#     # Visualize the datasets
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.scatter(moons_data[:, 0], moons_data[:, 1], s=5)
#     plt.title("Two Moons")
#     plt.subplot(1, 2, 2)
#     plt.scatter(mixture_data[:, 0], mixture_data[:, 1], s=5)
#     plt.title("Gaussian Mixture")
#     plt.show()
#     ```
#
# ---
#
# ## Mathematical Background
#
# !!! info "Distribution Characteristics"
#     Each dataset generator creates points from a different probability distribution:
#
#     - **Gaussian Mixtures**: Weighted combinations of Gaussian distributions, often arranged
#       in specific patterns like circles.
#     - **Two Moons**: Two interlocking half-circles with added noise, creating a challenging
#       bimodal distribution that's not linearly separable.
#     - **Checkerboard**: Alternating high and low density regions in a grid pattern, testing
#       an EBM's ability to capture multiple modes in a regular structure.
#     - **Swiss Roll**: A 2D manifold with spiral structure, testing the model's ability to
#       learn curved manifolds.
#
# !!! tip "Choosing a Dataset"
#     - For testing basic density estimation: use `make_gaussian_mixture`
#     - For evaluating mode-seeking behavior: use `make_8gaussians` or `make_checkerboard`
#     - For testing separation of entangled distributions: use `make_two_moons`
#     - For manifold learning: use `make_swiss_roll` or `make_circle`
#
# ---
#
# ## Implementation Details
#
# !!! note "Device Handling"
#     All generators create tensors that can be placed directly on a specified device,
#     making them integration-friendly with CUDA-based models.
#
# !!! warning "Random Number Generation"
#     The generators use PyTorch and NumPy random functions. For reproducible results,
#     set random seeds before calling these functions:
#     ```python
#     import torch
#     import numpy as np
#
#     torch.manual_seed(42)
#     np.random.seed(42)
#     ```
# """
#
# import torch
# import numpy as np
# from typing import Optional, Union, Tuple
#
#
# def _to_tensor(
#     data: np.ndarray,
#     dtype: torch.dtype = torch.float32,
#     device: Optional[Union[str, torch.device]] = None,
# ) -> torch.Tensor:
#     """
#     Convert NumPy array to PyTorch tensor with specified dtype and device.
#
#     Args:
#         data: NumPy array to convert.
#         dtype: PyTorch data type for the output tensor.
#         device: Device to place the tensor on.
#
#     Returns:
#         PyTorch tensor with the specified properties.
#     """
#     tensor = torch.from_numpy(data).to(dtype)
#     if device:
#         tensor = tensor.to(torch.device(device))
#     return tensor
#
#
# def make_gaussian_mixture(
#     n_samples: int = 2000,
#     n_components: int = 8,
#     std: float = 0.05,
#     radius: float = 1.0,
#     device: Optional[Union[str, torch.device]] = None,
# ) -> torch.Tensor:
#     """
#     Generates samples from a 2D Gaussian mixture arranged uniformly in a circle.
#
#     Creates a mixture of Gaussian distributions with centers equally spaced on a circle.
#     This distribution is useful for testing mode-seeking behavior in energy-based models.
#
#     Args:
#         n_samples: Total number of samples to generate.
#         n_components: Number of Gaussian components (modes).
#         std: Standard deviation of each Gaussian component.
#         radius: Radius of the circle on which the centers lie.
#         device: Device to place the tensor on.
#
#     Returns:
#         torch.Tensor: Tensor of batch_shape (n_samples, 2) containing the generated points.
#
#     Example:
#         >>> points = make_gaussian_mixture(n_samples=1000, n_components=6, std=0.1)
#         >>> print(points.batch_shape)
#         torch.Size([1000, 2])
#     """
#     if n_components <= 0:
#         raise ValueError("n_components must be positive")
#
#     thetas = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
#     centers = np.array(
#         [(radius * np.cos(t), radius * np.sin(t)) for t in thetas], dtype=np.float32
#     )
#     centers_torch = torch.from_numpy(centers)  # Keep centers on CPU for indexing
#
#     data = torch.empty(n_samples, 2)  # Pre-allocate tensor
#     samples_per_component = n_samples // n_components
#     remainder = n_samples % n_components
#
#     start_idx = 0
#     for i in range(n_components):
#         num = samples_per_component + (1 if i < remainder else 0)
#         if num == 0:
#             continue
#         end_idx = start_idx + num
#         # Sample noise directly on the target device if possible
#         noise = torch.randn(num, 2, device=device) * std
#         data[start_idx:end_idx] = (
#             centers_torch[i].to(device) + noise
#         )  # Move center to device here
#         start_idx = end_idx
#
#     # Shuffle the data to mix components
#     data = data[torch.randperm(n_samples)]
#
#     return data
#
#
# def make_8gaussians(
#     n_samples: int = 2000,
#     std: float = 0.02,
#     scale: float = 2.0,
#     device: Optional[Union[str, torch.device]] = None,
# ) -> torch.Tensor:
#     """
#     Generates samples from the specific '8 Gaussians' mixture.
#
#     This creates a specific arrangement of 8 Gaussian modes commonly used in the
#     energy-based modeling literature, with centers at the compass points and diagonals.
#
#     Args:
#         n_samples: Total number of samples to generate.
#         std: Standard deviation of each Gaussian component.
#         scale: Scaling factor for the centers (often 2).
#         device: Device to place the tensor on.
#
#     Returns:
#         torch.Tensor: Tensor of batch_shape (n_samples, 2) containing the generated points.
#
#     Example:
#         >>> points = make_8gaussians(n_samples=1000, std=0.05)
#         >>> print(points.batch_shape)
#         torch.Size([1000, 2])
#     """
#     centers = (
#         np.array(
#             [
#                 (1, 0),
#                 (-1, 0),
#                 (0, 1),
#                 (0, -1),
#                 (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
#                 (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
#                 (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
#                 (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
#             ],
#             dtype=np.float32,
#         )
#         * scale
#     )
#
#     return make_gaussian_mixture(
#         n_samples=n_samples,
#         n_components=8,  # Fixed at 8
#         std=std,
#         radius=scale,  # Use scale directly (approximation, centers are fixed)
#         device=device,
#     )
#
#
# def make_two_moons(
#     n_samples: int = 2000,
#     noise: float = 0.05,
#     device: Optional[Union[str, torch.device]] = None,
# ) -> torch.Tensor:
#     """
#     Generates the 'two moons' dataset.
#
#     Creates two interleaving half-circles with added Gaussian noise.
#     This is a classic dataset for testing classification, clustering,
#     and density estimation algorithms.
#
#     Args:
#         n_samples: Total number of samples.
#         noise: Standard deviation of Gaussian noise added.
#         device: Device to place the tensor on.
#
#     Returns:
#         torch.Tensor: Tensor of batch_shape (n_samples, 2) containing the generated points.
#
#     Example:
#         >>> points = make_two_moons(n_samples=1000, noise=0.1)
#         >>> print(points.batch_shape)
#         torch.Size([1000, 2])
#     """
#     n_samples_out = n_samples // 2
#     n_samples_in = n_samples - n_samples_out
#
#     # Outer moon
#     outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
#     outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
#     # Inner moon
#     inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
#     inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5
#
#     X = np.vstack(
#         [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
#     ).T.astype(np.float32)
#
#     # Add noise using torch for potential device efficiency
#     tensor_data = _to_tensor(X, device=device)
#     tensor_data += torch.randn_like(tensor_data) * noise
#
#     return tensor_data
#
#
# def make_swiss_roll(
#     n_samples: int = 2000,
#     noise: float = 0.05,
#     arclength: float = 3.0,  # Controls how many rolls (pi*arclength)
#     device: Optional[Union[str, torch.device]] = None,
# ) -> torch.Tensor:
#     """
#     Generates a 2D Swiss roll dataset.
#
#     The Swiss roll is a classic example of a nonlinear manifold embedded in a higher-dimensional
#     space. This generator creates a 2D version with a spiral structure.
#
#     Args:
#         n_samples: Number of samples.
#         noise: Standard deviation of Gaussian noise added.
#         arclength: Controls the length/tightness of the roll.
#         device: Device to place the tensor on.
#
#     Returns:
#         torch.Tensor: Tensor of batch_shape (n_samples, 2) containing the generated points.
#
#     Example:
#         >>> points = make_swiss_roll(n_samples=1000, noise=0.05, arclength=4.0)
#         >>> print(points.batch_shape)
#         torch.Size([1000, 2])
#     """
#     t = arclength * np.pi * (1 + 2 * np.random.rand(n_samples))
#
#     x = t * np.cos(t)
#     y = t * np.sin(t)
#
#     X = np.vstack((x, y)).T.astype(np.float32)
#
#     # Add noise using torch
#     tensor_data = _to_tensor(X, device=device)
#     tensor_data += torch.randn_like(tensor_data) * noise
#
#     # Center and scale slightly
#     tensor_data = (tensor_data - tensor_data.mean(dim=0)) / (
#         tensor_data.std(dim=0).mean() * 2.0
#     )
#
#     return tensor_data
#
#
# def make_circle(
#     n_samples: int = 2000,
#     noise: float = 0.05,
#     radius: float = 1.0,
#     device: Optional[Union[str, torch.device]] = None,
# ) -> torch.Tensor:
#     """
#     Generates points sampled uniformly on a circle with noise.
#
#     Creates a set of points uniformly distributed around a circle
#     of specified radius, with optional Gaussian noise added.
#
#     Args:
#         n_samples: Number of samples.
#         noise: Standard deviation of Gaussian noise added.
#         radius: Radius of the circle.
#         device: Device to place the tensor on.
#
#     Returns:
#         torch.Tensor: Tensor of batch_shape (n_samples, 2) containing the generated points.
#
#     Example:
#         >>> points = make_circle(n_samples=1000, noise=0.02, radius=1.5)
#         >>> print(points.batch_shape)
#         torch.Size([1000, 2])
#     """
#     angles = 2 * np.pi * np.random.rand(n_samples)
#     x = radius * np.cos(angles)
#     y = radius * np.sin(angles)
#     X = np.vstack((x, y)).T.astype(np.float32)
#
#     # Add noise using torch
#     tensor_data = _to_tensor(X, device=device)
#     tensor_data += torch.randn_like(tensor_data) * noise
#
#     return tensor_data
#
#
# def make_checkerboard(
#     n_samples: int = 2000,
#     range_limit: float = 4.0,
#     noise: float = 0.01,  # Small noise to avoid perfect grid
#     device: Optional[Union[str, torch.device]] = None,
# ) -> torch.Tensor:
#     """
#     Generates points in a 2D checkerboard pattern.
#
#     Creates a dataset with points distributed in a checkerboard pattern
#     of alternating high and low density regions, similar to a 2D chess board.
#
#     Args:
#         n_samples: Target number of samples.
#         range_limit: Defines the square region [-lim, lim] x [-lim, lim].
#         noise: Small Gaussian noise added to points.
#         device: Device to place the tensor on.
#
#     Returns:
#         torch.Tensor: Tensor of batch_shape (n_samples, 2) containing the generated points.
#
#     Example:
#         >>> points = make_checkerboard(n_samples=1000, range_limit=3.0)
#         >>> print(points.batch_shape)
#         torch.Size([1000, 2])
#     """
#     collected_samples = []
#     target = n_samples
#     batch_size = max(1000, n_samples // 10)  # Generate in batches for efficiency
#
#     while len(collected_samples) < target:
#         # Generate points uniformly in the range
#         x = np.random.uniform(-range_limit, range_limit, size=batch_size)
#         y = np.random.uniform(-range_limit, range_limit, size=batch_size)
#
#         # Checkerboard condition: floor(x) + floor(y) is odd
#         # Using ceil or floor changes the exact pattern but maintains the checkerboard
#         keep = (np.floor(x) + np.floor(y)) % 2 != 0
#         valid_points = np.vstack((x[keep], y[keep])).T.astype(np.float32)
#
#         needed = target - len(collected_samples)
#         collected_samples.extend(valid_points[:needed])
#
#     X = np.array(collected_samples, dtype=np.float32)
#     tensor_data = _to_tensor(X, device=device)
#     tensor_data += torch.randn_like(tensor_data) * noise  # Add slight noise
#
#     return tensor_data
#
#
# def make_pinwheel(
#     n_samples: int = 2000,
#     n_classes: int = 5,
#     noise: float = 0.05,  # Final additive Cartesian noise std dev
#     radial_scale: float = 2.0,  # Controls max radius/length of blades
#     angular_scale: float = 0.1,  # Controls std dev of angle noise (blade thickness)
#     spiral_scale: float = 5.0,  # Controls how tightly blades spiral (rate of angle change with radius)
#     seed: Optional[int] = None,  # For reproducibility
#     device: Optional[Union[str, torch.device]] = None,
# ) -> torch.Tensor:
#     """
#     Generates the pinwheel dataset with curved blades, resembling the target image.
#
#     Points are generated in polar coordinates (radius, angle) and then converted
#     to Cartesian coordinates. The angle depends on both the class (blade) and
#     the radius, creating the spiral effect. Density decreases radially due to
#     the sqrt(uniform) distribution used for the base radial variable.
#
#     Args:
#         n_samples: Total number of samples.
#         n_classes: Number of 'blades' in the pinwheel.
#         noise: Standard deviation of Gaussian noise added to final Cartesian coordinates.
#         radial_scale: Scales the maximum radius of the points, controlling blade length.
#         angular_scale: Standard deviation of the angular noise around the spiral centerline
#                        (controls blade thickness).
#         spiral_scale: Controls the rate at which blades spiral outwards. Higher values
#                       mean tighter spirals (angle changes more rapidly with radius).
#         seed: Optional random seed for reproducibility of numpy and torch operations.
#         device: Device to place the output tensor on (e.g., 'cpu', 'cuda').
#
#     Returns:
#         torch.Tensor: Tensor of batch_shape (n_samples, 2) containing the generated points.
#
#     Example:
#         >>> points = make_pinwheel_corrected(n_samples=1000, n_classes=6, seed=42)
#         >>> print(points.batch_shape)
#         torch.Size([1000, 2])
#         >>> # Can be plotted using matplotlib:
#         >>> # import matplotlib.pyplot as plt
#         >>> # plt.figure(figsize=(8, 8))
#         >>> # plt.scatter(points[:, 0], points[:, 1], s=5, alpha=0.6)
#         >>> # plt.title("Corrected Pinwheel")
#         >>> # plt.grid(True, alpha=0.3)
#         >>> # plt.axis('equal')
#         >>> # plt.show()
#     """
#     if seed is not None:
#         np.random.seed(seed)
#         # Seed torch as well if noise > 0 for reproducible noise addition
#         if noise > 0:
#             torch.manual_seed(seed)
#
#     all_points = []
#     samples_per_class = n_samples // n_classes
#     remainder = n_samples % n_classes
#
#     for class_idx in range(n_classes):
#         n_class_samples = samples_per_class + (1 if class_idx < remainder else 0)
#         if n_class_samples == 0:
#             continue
#
#         # --- Generate points for this class/blade ---
#
#         # 1. Base variable 't': Controls position along the spiral.
#         # Using sqrt(uniform) makes points denser near the center (radius=0).
#         # Range [0, 1]
#         t = np.sqrt(np.random.rand(n_class_samples))
#
#         # 2. Radius: Increases proportionally to 't', scaled by radial_scale.
#         radii = t * radial_scale
#
#         # 3. Angle: Composed of three parts:
#         #    a) Base angle for the class/blade
#         base_angle = class_idx * (2 * np.pi / n_classes)
#         #    b) Spiral term: Angle increases linearly with 't' (and thus radius).
#         #       Controlled by spiral_scale.
#         spiral_angle = spiral_scale * t
#         #    c) Angular noise: Adds thickness to the blade around the spiral centerline.
#         #       Controlled by angular_scale.
#         angle_noise = np.random.randn(n_class_samples) * angular_scale
#
#         # Total angle
#         thetas = base_angle + spiral_angle + angle_noise
#
#         # 4. Convert polar (radii, thetas) to Cartesian (x, y)
#         x = radii * np.cos(thetas)
#         y = radii * np.sin(thetas)
#
#         class_points = np.stack([x, y], axis=1)
#         all_points.append(class_points)
#
#     # Combine points from all classes into a single array
#     data = np.concatenate(all_points, axis=0).astype(np.float32)
#
#     # Shuffle points so samples from different classes are mixed
#     np.random.shuffle(data)
#
#     # Convert to torch tensor
#     tensor_data = torch.from_numpy(data).to(device)
#
#     # Add final additive Gaussian noise in Cartesian coordinates
#     if noise > 0:
#         # Ensure noise tensor is on the same device
#         noise_values = torch.randn_like(tensor_data) * noise
#         tensor_data += noise_values.to(tensor_data.device)  # Ensure device match
#
#     # Note: Removed the final centering and scaling from the original code.
#     # This version generates points centered around (0,0) by construction.
#     # If specific mean/std is required, apply it *after* calling this function.
#
#     return tensor_data
#
#
# def make_2d_grid(
#     n_samples_per_dim: int = 10,
#     range_limit: float = 1.0,
#     noise: float = 0.01,
#     device: Optional[Union[str, torch.device]] = None,
# ) -> torch.Tensor:
#     """Generates points on a 2D grid within [-range_limit, range_limit].
#
#     Creates a regular grid of points in a square region, with optional
#     Gaussian noise added to each point.
#
#     Args:
#         n_samples_per_dim: Number of points along each dimension. Total samples = n^2.
#         range_limit: Defines the square region [-lim, lim] x [-lim, lim].
#         noise: Standard deviation of Gaussian noise added to grid points.
#         device: Device to place the tensor on.
#
#     Returns:
#         torch.Tensor: Tensor of batch_shape (n_samples_per_dim**2, 2) containing the generated points.
#
#     Example:
#         >>> points = make_2d_grid(n_samples_per_dim=20, noise=0.02)
#         >>> print(points.batch_shape)
#         torch.Size([400, 2])
#     """
#     x_coords = np.linspace(-range_limit, range_limit, n_samples_per_dim)
#     y_coords = np.linspace(-range_limit, range_limit, n_samples_per_dim)
#     xv, yv = np.meshgrid(x_coords, y_coords)
#     X = np.vstack([xv.flatten(), yv.flatten()]).T.astype(np.float32)
#
#     tensor_data = _to_tensor(X, device=device)
#     tensor_data += torch.randn_like(tensor_data) * noise
#
#     return tensor_data
