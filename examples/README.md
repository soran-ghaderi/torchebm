# TorchEBM Examples

This directory contains example code demonstrating the use of TorchEBM for energy-based modeling and sampling.

## Directory Structure

The examples are organized into the following categories:

### Core

Energy function examples that demonstrate the core functionality:

#### Energy Functions
- `core/energy_functions/landscape_2d.py` - Basic 2D visualization of energy landscapes
- `core/energy_functions/multimodal.py` - Visualization of multimodal energy functions
- `core/energy_functions/parametric.py` - Interactive parametric energy function exploration

### Samplers

Examples demonstrating different sampling methods:

#### Hamiltonian Monte Carlo (HMC)
- `samplers/hmc/gaussian_sampling.py` - Basic HMC sampling from Gaussian distributions
- `samplers/hmc/advanced.py` - Advanced HMC techniques and diagnostics
- `samplers/hmc/mass_matrix.py` - HMC with custom mass matrix specification

#### Langevin Dynamics
- `samplers/langevin/gaussian_sampling.py` - Basic Langevin sampling from Gaussian distributions
- `samplers/langevin/multimodal_sampling.py` - Sampling from multimodal distributions
- `samplers/langevin/visualization_trajectory.py` - Visualization of Langevin dynamics trajectories
- `samplers/langevin/advanced.py` - Advanced Langevin dynamics techniques

### Visualization

Visualization utilities and examples:

#### Basic Visualizations
- `visualization/basic/contour_plots.py` - Simple energy landscape contour plots
- `visualization/basic/distribution_comparison.py` - Comparing sampled vs. true distributions

#### Advanced Visualizations
- `visualization/advanced/trajectory_animation.py` - Animated sampling trajectory visualization
- `visualization/advanced/parallel_chains.py` - Visualizing multiple sampling chains
- `visualization/advanced/energy_over_time.py` - Energy evolution during sampling process

### Utils
Utility functions and performance tools:
- `utils/performance_benchmark.py` - Benchmarking sampler performance
- `utils/convergence_diagnostics.py` - Tools for diagnosing sampling convergence

## Common Visualization Tools
- `visualization/utils.py` - Shared visualization functions for energy landscapes and samplers

## Running Examples

To run any example, navigate to the repository root directory and execute:

```bash
# List all available examples
python examples/main.py --list

# Run a specific example
python examples/main.py samplers/langevin/gaussian_sampling

# Or run the Python file directly
python examples/core/energy_functions/landscape_2d.py
```

## Example Entry Points

Each example is designed as a standalone script that can be executed directly, with the main visualization or sampling functionality in the `if __name__ == "__main__":` block. 