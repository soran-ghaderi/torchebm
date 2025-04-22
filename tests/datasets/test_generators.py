"""Tests for the torchebm.datasets.generators module."""

import pytest
import torch
import numpy as np
from typing import Type

from torchebm.datasets.generators import (
    BaseSyntheticDataset,
    GaussianMixtureDataset,
    EightGaussiansDataset,
    TwoMoonsDataset,
    SwissRollDataset,
    CircleDataset,
    CheckerboardDataset,
    PinwheelDataset,
    GridDataset,
)


class TestBaseSyntheticDataset:
    """Tests for the BaseSyntheticDataset abstract base class behaviors."""

    def test_base_class_is_abstract(self):
        """Test that BaseSyntheticDataset cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSyntheticDataset(n_samples=100)

    def test_child_must_implement_generate_data(self):
        """Test that child classes must implement _generate_data method."""

        class BrokenDataset(BaseSyntheticDataset):
            # Intentionally does not implement _generate_data
            pass

        with pytest.raises(TypeError):
            BrokenDataset(n_samples=100)


@pytest.mark.parametrize(
    "dataset_class,required_params",
    [
        (
            GaussianMixtureDataset,
            {"n_samples": 100, "n_components": 4, "std": 0.1, "radius": 1.0},
        ),
        (EightGaussiansDataset, {"n_samples": 100, "std": 0.1, "scale": 2.0}),
        (TwoMoonsDataset, {"n_samples": 100, "noise": 0.1}),
        (SwissRollDataset, {"n_samples": 100, "noise": 0.1, "arclength": 2.0}),
        (CircleDataset, {"n_samples": 100, "noise": 0.1, "radius": 1.0}),
        (CheckerboardDataset, {"n_samples": 100, "range_limit": 3.0, "noise": 0.01}),
        (
            PinwheelDataset,
            {
                "n_samples": 100,
                "n_classes": 5,
                "noise": 0.1,
                "radial_scale": 1.0,
                "angular_scale": 0.1,
                "spiral_scale": 1.0,
            },
        ),
        (GridDataset, {"n_samples_per_dim": 10, "range_limit": 1.0, "noise": 0.01}),
    ],
)
class TestAllDatasets:
    """Tests for all dataset classes."""

    def test_dataset_initialization(self, dataset_class, required_params):
        """Test that dataset can be initialized with minimum required parameters."""
        dataset = dataset_class(**required_params)
        assert isinstance(dataset, dataset_class)

        # For grid dataset, n_samples is derived from n_samples_per_dim
        if dataset_class == GridDataset:
            expected_samples = required_params["n_samples_per_dim"] ** 2
        else:
            expected_samples = required_params["n_samples"]

        assert len(dataset) == expected_samples

    def test_dataset_get_data(self, dataset_class, required_params):
        """Test that get_data() returns a tensor with expected batch_shape."""
        dataset = dataset_class(**required_params)
        data = dataset.get_data()

        assert isinstance(data, torch.Tensor)
        assert data.dim() == 2
        assert data.shape[1] == 2  # All datasets should be 2D

        # For grid dataset, n_samples is derived from n_samples_per_dim
        if dataset_class == GridDataset:
            expected_samples = required_params["n_samples_per_dim"] ** 2
        else:
            expected_samples = required_params["n_samples"]

        assert data.shape[0] == expected_samples

    def test_dataset_getitem(self, dataset_class, required_params):
        """Test that __getitem__ returns individual points with expected batch_shape."""
        dataset = dataset_class(**required_params)

        # Test first item
        first_item = dataset[0]
        assert isinstance(first_item, torch.Tensor)
        assert first_item.dim() == 1
        assert first_item.shape[0] == 2

        # Test last item
        last_idx = len(dataset) - 1
        last_item = dataset[last_idx]
        assert isinstance(last_item, torch.Tensor)
        assert last_item.shape[0] == 2

        # Test out of bounds
        with pytest.raises(IndexError):
            dataset[len(dataset)]

    def test_dataset_len(self, dataset_class, required_params):
        """Test that __len__ returns expected number of samples."""
        dataset = dataset_class(**required_params)

        # For grid dataset, n_samples is derived from n_samples_per_dim
        if dataset_class == GridDataset:
            expected_samples = required_params["n_samples_per_dim"] ** 2
        else:
            expected_samples = required_params["n_samples"]

        assert len(dataset) == expected_samples

    def test_dataset_repr(self, dataset_class, required_params):
        """Test that __repr__ returns a string with class name."""
        dataset = dataset_class(**required_params)
        repr_str = repr(dataset)
        assert dataset_class.__name__ in repr_str

    def test_dataset_regenerate(self, dataset_class, required_params):
        """Test that regenerate() creates new data."""
        dataset = dataset_class(**required_params, seed=42)
        data1 = dataset.get_data().clone()
        dataset.regenerate(seed=43)
        data2 = dataset.get_data()

        # Data should be different with different seeds
        assert not torch.allclose(data1, data2)

        # Regenerating with same seed should produce same data
        dataset.regenerate(seed=42)
        data3 = dataset.get_data()
        assert torch.allclose(data1, data3)

    def test_dataset_device(self, dataset_class, required_params):
        """Test that dataset respects device parameter."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Generate on CPU
        dataset_cpu = dataset_class(**required_params, device="cpu")
        data_cpu = dataset_cpu.get_data()
        assert data_cpu.device.type == "cpu"

        # Generate on CUDA
        dataset_cuda = dataset_class(**required_params, device="cuda")
        data_cuda = dataset_cuda.get_data()
        assert data_cuda.device.type == "cuda"

    def test_dataset_dtype(self, dataset_class, required_params):
        """Test that dataset respects dtype parameter."""
        # Default should be float32
        dataset_default = dataset_class(**required_params)
        data_default = dataset_default.get_data()
        assert data_default.dtype == torch.float32

        # Explicit float64
        dataset_float64 = dataset_class(**required_params, dtype=torch.float64)
        data_float64 = dataset_float64.get_data()
        assert data_float64.dtype == torch.float64

    def test_dataset_seed(self, dataset_class, required_params):
        """Test that seed parameter guarantees reproducibility."""
        dataset1 = dataset_class(**required_params, seed=42)
        data1 = dataset1.get_data()

        dataset2 = dataset_class(**required_params, seed=42)
        data2 = dataset2.get_data()

        # Same seed should produce identical data
        assert torch.allclose(data1, data2)

        # Different seed should produce different data
        dataset3 = dataset_class(**required_params, seed=43)
        data3 = dataset3.get_data()
        assert not torch.allclose(data1, data3)


def test_gaussian_mixture_parameter_validation():
    """Test parameter validation for GaussianMixtureDataset."""
    # Test that n_components must be positive
    with pytest.raises(ValueError):
        GaussianMixtureDataset(n_samples=100, n_components=0)

    # Test that std must be positive
    with pytest.raises(ValueError):
        GaussianMixtureDataset(n_samples=100, n_components=4, std=-0.1)


def test_pinwheel_parameter_validation():
    """Test parameter validation for PinwheelDataset."""
    # Test that n_classes must be positive
    with pytest.raises(ValueError):
        PinwheelDataset(n_samples=100, n_classes=0)


def test_grid_parameter_validation():
    """Test parameter validation for GridDataset."""
    # Test that n_samples_per_dim must be positive
    with pytest.raises(ValueError):
        GridDataset(n_samples_per_dim=0)


def test_n_samples_validation():
    """Test that n_samples must be positive for all datasets."""
    with pytest.raises(ValueError):
        GaussianMixtureDataset(n_samples=0)


class TestDataLoaderCompatibility:
    """Test compatibility with PyTorch DataLoader."""

    def test_dataloader_iteration(self):
        """Test that datasets work with DataLoader."""
        from torch.utils.data import DataLoader

        dataset = TwoMoonsDataset(n_samples=100, noise=0.05)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

        # Iterate through all batches
        total_samples = 0
        for batch in dataloader:
            assert isinstance(batch, torch.Tensor)
            assert batch.dim() == 2
            assert batch.shape[1] == 2
            total_samples += batch.shape[0]

        assert total_samples == 100


class TestDatasetSpecifics:
    """Test specific characteristics of each dataset type."""

    def test_two_moons_shape(self):
        """Test that TwoMoonsDataset has expected moon batch_shape."""
        dataset = TwoMoonsDataset(n_samples=2000, noise=0.01, seed=42)
        data = dataset.get_data().cpu().numpy()

        # Simple test: check if points are distributed in upper and lower regions
        upper_region = data[data[:, 1] > 0.5]
        lower_region = data[data[:, 1] < 0]

        # Should have points in both regions
        assert len(upper_region) > 0
        assert len(lower_region) > 0

        # Upper region should be mostly on the left side
        assert np.mean(upper_region[:, 0]) < 0.5

        # Lower region should be mostly on the right side
        assert np.mean(lower_region[:, 0]) > 0.5

    def test_checkerboard_pattern(self):
        """Test that CheckerboardDataset has alternating pattern."""
        dataset = CheckerboardDataset(
            n_samples=10000, range_limit=3.0, noise=0.01, seed=42
        )
        data = dataset.get_data().cpu().numpy()

        # Scale down to integer grid and count points in cells
        grid_size = 6  # 3x3 grid on each side
        cell_counts = np.zeros((grid_size, grid_size))

        # Map points to grid cells
        for x, y in data:
            # Map from [-range_limit, range_limit] to [0, grid_size]
            grid_x = int((x + 3.0) * grid_size / 6.0)
            grid_y = int((y + 3.0) * grid_size / 6.0)

            # Ensure within bounds
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                cell_counts[grid_y, grid_x] += 1

        # Check alternating pattern by comparing neighboring cells
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                # If cell has points, adjacent cells should have fewer
                if cell_counts[i, j] > 50:  # Arbitrary threshold for "has points"
                    assert cell_counts[i + 1, j] < 50 or cell_counts[i, j + 1] < 50

    def test_grid_dimensions(self):
        """Test that GridDataset creates a grid with expected dimensions."""
        n_dim = 15
        dataset = GridDataset(n_samples_per_dim=n_dim, range_limit=1.0, noise=0)
        data = dataset.get_data()

        # Without noise, points should be exactly on a grid
        # Extract unique x and y coordinates
        unique_x = torch.unique(data[:, 0])
        unique_y = torch.unique(data[:, 1])

        assert len(unique_x) == n_dim
        assert len(unique_y) == n_dim


if __name__ == "__main__":
    pytest.main()
