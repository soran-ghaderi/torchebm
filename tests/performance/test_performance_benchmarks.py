"""
Performance benchmark tests for optimizations.
"""
import time
import pytest
import torch
from torchebm.core.base_model import GaussianModel, DoubleWellModel
from torchebm.samplers import LangevinDynamics, HamiltonianMonteCarlo


class TestPerformanceBenchmarks:
    """Performance benchmark tests to verify optimization improvements."""

    def test_gaussian_model_forward_performance(self):
        """Test GaussianModel.forward() performance with einsum optimization."""
        device = "cpu"
        dim = 50
        batch_size = 1000
        
        mean = torch.zeros(dim, device=device)
        cov = torch.eye(dim, device=device)
        model = GaussianModel(mean=mean, cov=cov).to(device)
        
        x = torch.randn(batch_size, dim, device=device)
        
        # Warmup
        for _ in range(10):
            _ = model(x)
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            energy = model(x)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 2 seconds on CPU)
        assert elapsed < 2.0, f"GaussianModel.forward() too slow: {elapsed:.3f}s"
        assert energy.shape == (batch_size,)
        
    def test_gradient_computation_performance(self):
        """Test BaseModel.gradient() performance without unnecessary conversions."""
        device = "cpu"
        dim = 20
        batch_size = 100
        
        model = DoubleWellModel(barrier_height=2.0).to(device)
        x = torch.randn(batch_size, dim, device=device, dtype=torch.float32)
        
        # Warmup
        for _ in range(5):
            _ = model.gradient(x)
        
        # Benchmark
        start = time.time()
        for _ in range(50):
            grad = model.gradient(x)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 3 seconds on CPU)
        assert elapsed < 3.0, f"BaseModel.gradient() too slow: {elapsed:.3f}s"
        assert grad.shape == x.shape
        assert grad.dtype == x.dtype  # Should preserve dtype now
        
    def test_langevin_dynamics_sampling_performance(self):
        """Test LangevinDynamics sampling performance."""
        device = "cpu"
        dim = 10
        n_samples = 50
        n_steps = 100
        
        mean = torch.zeros(dim, device=device)
        cov = torch.eye(dim, device=device)
        model = GaussianModel(mean=mean, cov=cov).to(device)
        
        sampler = LangevinDynamics(
            model=model,
            step_size=0.01,
            device=device
        )
        
        # Warmup
        _ = sampler.sample(dim=dim, n_steps=10, n_samples=10)
        
        # Benchmark
        start = time.time()
        samples = sampler.sample(dim=dim, n_steps=n_steps, n_samples=n_samples)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 5 seconds on CPU)
        assert elapsed < 5.0, f"LangevinDynamics.sample() too slow: {elapsed:.3f}s"
        assert samples.shape == (n_samples, dim)
        
    def test_hmc_sampling_performance(self):
        """Test HamiltonianMonteCarlo sampling performance."""
        device = "cpu"
        dim = 10
        n_samples = 20
        n_steps = 50
        
        mean = torch.zeros(dim, device=device)
        cov = torch.eye(dim, device=device)
        model = GaussianModel(mean=mean, cov=cov).to(device)
        
        sampler = HamiltonianMonteCarlo(
            model=model,
            step_size=0.1,
            n_leapfrog_steps=5,
            device=device
        )
        
        # Warmup
        _ = sampler.sample(dim=dim, n_steps=5, n_samples=5)
        
        # Benchmark
        start = time.time()
        samples = sampler.sample(dim=dim, n_steps=n_steps, n_samples=n_samples)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 8 seconds on CPU)
        assert elapsed < 8.0, f"HamiltonianMonteCarlo.sample() too slow: {elapsed:.3f}s"
        assert samples.shape == (n_samples, dim)
        
    def test_hmc_with_diagnostics_performance(self):
        """Test HMC sampling with diagnostics doesn't use inefficient expand operations."""
        device = "cpu"
        dim = 10
        n_samples = 20
        n_steps = 30
        
        mean = torch.zeros(dim, device=device)
        cov = torch.eye(dim, device=device)
        model = GaussianModel(mean=mean, cov=cov).to(device)
        
        sampler = HamiltonianMonteCarlo(
            model=model,
            step_size=0.1,
            n_leapfrog_steps=5,
            device=device
        )
        
        # Warmup
        _ = sampler.sample(dim=dim, n_steps=5, n_samples=5, return_diagnostics=True)
        
        # Benchmark with diagnostics
        start = time.time()
        samples, diagnostics = sampler.sample(
            dim=dim, 
            n_steps=n_steps, 
            n_samples=n_samples,
            return_diagnostics=True
        )
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 8 seconds on CPU)
        assert elapsed < 8.0, f"HMC with diagnostics too slow: {elapsed:.3f}s"
        assert samples.shape == (n_samples, dim)
        assert diagnostics.shape == (n_steps, 4, n_samples, dim)
        
    def test_langevin_with_diagnostics_performance(self):
        """Test Langevin sampling with diagnostics doesn't use inefficient expand operations."""
        device = "cpu"
        dim = 10
        n_samples = 50
        n_steps = 100
        
        mean = torch.zeros(dim, device=device)
        cov = torch.eye(dim, device=device)
        model = GaussianModel(mean=mean, cov=cov).to(device)
        
        sampler = LangevinDynamics(
            model=model,
            step_size=0.01,
            device=device
        )
        
        # Warmup
        _ = sampler.sample(dim=dim, n_steps=10, n_samples=10, return_diagnostics=True)
        
        # Benchmark with diagnostics
        start = time.time()
        samples, diagnostics = sampler.sample(
            dim=dim, 
            n_steps=n_steps, 
            n_samples=n_samples,
            return_diagnostics=True
        )
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 6 seconds on CPU)
        assert elapsed < 6.0, f"Langevin with diagnostics too slow: {elapsed:.3f}s"
        assert samples.shape == (n_samples, dim)
        assert diagnostics.shape == (n_steps, 3, n_samples, dim)


class TestOptimizationCorrectness:
    """Tests to verify optimization correctness."""
    
    def test_gaussian_energy_correctness(self):
        """Verify GaussianModel produces correct energies after optimization."""
        device = "cpu"
        dim = 5
        batch_size = 10
        
        mean = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
        cov = torch.eye(dim, device=device) * 2.0
        model = GaussianModel(mean=mean, cov=cov).to(device)
        
        # Test with known input
        x = torch.zeros(batch_size, dim, device=device)
        energy = model(x)
        
        # For x=0, energy = 0.5 * mean^T @ cov_inv @ mean
        # cov_inv = 0.5 * I, so energy = 0.5 * 0.5 * (1+4+9+16+25) = 0.25 * 55 = 13.75
        expected_energy = 0.25 * (1 + 4 + 9 + 16 + 25)
        assert torch.allclose(energy, torch.full((batch_size,), expected_energy, device=device), rtol=1e-5)
        
    def test_gradient_dtype_preservation(self):
        """Verify gradient computation preserves dtype without unnecessary conversions."""
        device = "cpu"
        model = DoubleWellModel(barrier_height=2.0).to(device)
        
        # Test with float32
        x_f32 = torch.randn(10, 5, device=device, dtype=torch.float32)
        grad_f32 = model.gradient(x_f32)
        assert grad_f32.dtype == torch.float32
        
        # Test with float64
        x_f64 = torch.randn(10, 5, device=device, dtype=torch.float64)
        model_f64 = DoubleWellModel(barrier_height=2.0, dtype=torch.float64).to(device)
        grad_f64 = model_f64.gradient(x_f64)
        assert grad_f64.dtype == torch.float64
        
    def test_hmc_diagnostics_shape_correctness(self):
        """Verify HMC diagnostics have correct shapes after broadcasting optimization."""
        device = "cpu"
        dim = 10
        n_samples = 20
        n_steps = 15
        
        mean = torch.zeros(dim, device=device)
        cov = torch.eye(dim, device=device)
        model = GaussianModel(mean=mean, cov=cov).to(device)
        
        sampler = HamiltonianMonteCarlo(
            model=model,
            step_size=0.1,
            n_leapfrog_steps=5,
            device=device
        )
        
        samples, diagnostics = sampler.sample(
            dim=dim,
            n_steps=n_steps,
            n_samples=n_samples,
            return_diagnostics=True
        )
        
        # Check shapes
        assert diagnostics.shape == (n_steps, 4, n_samples, dim)
        
        # Verify diagnostics contain valid values
        assert torch.all(torch.isfinite(diagnostics))
        
    def test_langevin_diagnostics_shape_correctness(self):
        """Verify Langevin diagnostics have correct shapes after broadcasting optimization."""
        device = "cpu"
        dim = 10
        n_samples = 30
        n_steps = 20
        
        mean = torch.zeros(dim, device=device)
        cov = torch.eye(dim, device=device)
        model = GaussianModel(mean=mean, cov=cov).to(device)
        
        sampler = LangevinDynamics(
            model=model,
            step_size=0.01,
            device=device
        )
        
        samples, diagnostics = sampler.sample(
            dim=dim,
            n_steps=n_steps,
            n_samples=n_samples,
            return_diagnostics=True
        )
        
        # Check shapes
        assert diagnostics.shape == (n_steps, 3, n_samples, dim)
        
        # Verify diagnostics contain valid values
        assert torch.all(torch.isfinite(diagnostics))
