# `torchebm.datasets`

## `CheckerboardDataset`

Bases: `BaseSyntheticDataset`

Generates points in a 2D checkerboard pattern using rejection sampling.

Parameters:

| Name          | Type                           | Description                                          | Default   |
| ------------- | ------------------------------ | ---------------------------------------------------- | --------- |
| `n_samples`   | `int`                          | The target number of samples.                        | `2000`    |
| `range_limit` | `float`                        | Defines the square region [-lim, lim] x [-lim, lim]. | `4.0`     |
| `noise`       | `float`                        | Small Gaussian noise added to the points.            | `0.01`    |
| `device`      | `Optional[Union[str, device]]` | The device for the tensor.                           | `None`    |
| `dtype`       | `dtype`                        | The data type for the tensor.                        | `float32` |
| `seed`        | `Optional[int]`                | A random seed for reproducibility.                   | `None`    |

Source code in `torchebm/datasets/generators.py`

```python
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
```

## `CircleDataset`

Bases: `BaseSyntheticDataset`

Generates points sampled uniformly on a circle with noise.

Parameters:

| Name        | Type                           | Description                                          | Default   |
| ----------- | ------------------------------ | ---------------------------------------------------- | --------- |
| `n_samples` | `int`                          | The number of samples.                               | `2000`    |
| `noise`     | `float`                        | The standard deviation of the Gaussian noise to add. | `0.05`    |
| `radius`    | `float`                        | The radius of the circle.                            | `1.0`     |
| `device`    | `Optional[Union[str, device]]` | The device for the tensor.                           | `None`    |
| `dtype`     | `dtype`                        | The data type for the tensor.                        | `float32` |
| `seed`      | `Optional[int]`                | A random seed for reproducibility.                   | `None`    |

Source code in `torchebm/datasets/generators.py`

```python
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
```

## `EightGaussiansDataset`

Bases: `BaseSyntheticDataset`

Generates samples from the '8 Gaussians' mixture distribution.

Parameters:

| Name        | Type                           | Description                                        | Default   |
| ----------- | ------------------------------ | -------------------------------------------------- | --------- |
| `n_samples` | `int`                          | The total number of samples.                       | `2000`    |
| `std`       | `float`                        | The standard deviation of each component.          | `0.02`    |
| `scale`     | `float`                        | A scaling factor for the centers of the Gaussians. | `2.0`     |
| `device`    | `Optional[Union[str, device]]` | The device for the tensor.                         | `None`    |
| `dtype`     | `dtype`                        | The data type for the tensor.                      | `float32` |
| `seed`      | `Optional[int]`                | A random seed for reproducibility.                 | `None`    |

Source code in `torchebm/datasets/generators.py`

```python
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
```

## `GaussianMixtureDataset`

Bases: `BaseSyntheticDataset`

Generates a 2D Gaussian mixture dataset with components arranged in a circle.

Parameters:

| Name           | Type                           | Description                                        | Default   |
| -------------- | ------------------------------ | -------------------------------------------------- | --------- |
| `n_samples`    | `int`                          | The total number of samples.                       | `2000`    |
| `n_components` | `int`                          | The number of Gaussian components (modes).         | `8`       |
| `std`          | `float`                        | The standard deviation of each Gaussian component. | `0.05`    |
| `radius`       | `float`                        | The radius of the circle on which the centers lie. | `1.0`     |
| `device`       | `Optional[Union[str, device]]` | The device for the tensor.                         | `None`    |
| `dtype`        | `dtype`                        | The data type for the tensor.                      | `float32` |
| `seed`         | `Optional[int]`                | A random seed for reproducibility.                 | `None`    |

Source code in `torchebm/datasets/generators.py`

```python
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
```

## `GridDataset`

Bases: `BaseSyntheticDataset`

Generates points on a 2D grid.

Note: The total number of samples will be `n_samples_per_dim` \*\* 2.

Parameters:

| Name                | Type                           | Description                                                  | Default   |
| ------------------- | ------------------------------ | ------------------------------------------------------------ | --------- |
| `n_samples_per_dim` | `int`                          | The number of points along each dimension.                   | `10`      |
| `range_limit`       | `float`                        | Defines the square region [-lim, lim] x [-lim, lim].         | `1.0`     |
| `noise`             | `float`                        | The standard deviation of the Gaussian noise to add.         | `0.01`    |
| `device`            | `Optional[Union[str, device]]` | The device for the tensor.                                   | `None`    |
| `dtype`             | `dtype`                        | The data type for the tensor.                                | `float32` |
| `seed`              | `Optional[int]`                | A random seed for reproducibility (primarily affects noise). | `None`    |

Source code in `torchebm/datasets/generators.py`

```python
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
```

## `PinwheelDataset`

Bases: `BaseSyntheticDataset`

Generates the pinwheel dataset with curved blades.

Parameters:

| Name            | Type                           | Description                                                     | Default   |
| --------------- | ------------------------------ | --------------------------------------------------------------- | --------- |
| `n_samples`     | `int`                          | The total number of samples.                                    | `2000`    |
| `n_classes`     | `int`                          | The number of 'blades' in the pinwheel.                         | `5`       |
| `noise`         | `float`                        | The standard deviation of the final additive Cartesian noise.   | `0.05`    |
| `radial_scale`  | `float`                        | Controls the maximum radius/length of the blades.               | `2.0`     |
| `angular_scale` | `float`                        | Controls the standard deviation of the angle noise (thickness). | `0.1`     |
| `spiral_scale`  | `float`                        | Controls the tightness of the spiral.                           | `5.0`     |
| `device`        | `Optional[Union[str, device]]` | The device for the tensor.                                      | `None`    |
| `dtype`         | `dtype`                        | The data type for the tensor.                                   | `float32` |
| `seed`          | `Optional[int]`                | A random seed for reproducibility.                              | `None`    |

Source code in `torchebm/datasets/generators.py`

```python
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
```

## `SwissRollDataset`

Bases: `BaseSyntheticDataset`

Generates a 2D Swiss roll dataset.

Parameters:

| Name        | Type                           | Description                                          | Default   |
| ----------- | ------------------------------ | ---------------------------------------------------- | --------- |
| `n_samples` | `int`                          | The number of samples.                               | `2000`    |
| `noise`     | `float`                        | The standard deviation of the Gaussian noise to add. | `0.05`    |
| `arclength` | `float`                        | A factor controlling how many rolls the spiral has.  | `3.0`     |
| `device`    | `Optional[Union[str, device]]` | The device for the tensor.                           | `None`    |
| `dtype`     | `dtype`                        | The data type for the tensor.                        | `float32` |
| `seed`      | `Optional[int]`                | A random seed for reproducibility.                   | `None`    |

Source code in `torchebm/datasets/generators.py`

```python
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
```

## `TwoMoonsDataset`

Bases: `BaseSyntheticDataset`

Generates the 'two moons' dataset.

Parameters:

| Name        | Type                           | Description                                          | Default   |
| ----------- | ------------------------------ | ---------------------------------------------------- | --------- |
| `n_samples` | `int`                          | The total number of samples.                         | `2000`    |
| `noise`     | `float`                        | The standard deviation of the Gaussian noise to add. | `0.05`    |
| `device`    | `Optional[Union[str, device]]` | The device for the tensor.                           | `None`    |
| `dtype`     | `dtype`                        | The data type for the tensor.                        | `float32` |
| `seed`      | `Optional[int]`                | A random seed for reproducibility.                   | `None`    |

Source code in `torchebm/datasets/generators.py`

```python
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
```
