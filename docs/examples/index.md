---
title: Examples
description: Practical examples demonstrating how to use TorchEBM
---

# TorchEBM Examples

This section contains practical examples that demonstrate how to use TorchEBM for energy-based modeling. Each example is fully tested and focuses on a specific use case or feature.

<div class="grid cards" markdown>

-   :material-terrain:{ .lg .middle } __Energy Landscape Visualization__

    ---

    Visualize energy functions to understand their landscapes and characteristics.

    [:octicons-arrow-right-24: Energy Visualization](energy_visualization.md)

-   :material-sigma:{ .lg .middle } __Langevin Dynamics Sampling__

    ---

    Sample from various distributions using Langevin dynamics.

    [:octicons-arrow-right-24: Langevin Dynamics](langevin_dynamics.md)

-   :material-axis-arrow:{ .lg .middle } __Hamiltonian Monte Carlo__

    ---

    Learn to use Hamiltonian Monte Carlo for efficient sampling.

    [:octicons-arrow-right-24: Hamiltonian Monte Carlo](hmc.md)

-   :material-chart-line:{ .lg .middle } __Langevin Sampler Trajectory__

    ---

    Visualize sampling trajectories on multimodal energy landscapes.

    [:octicons-arrow-right-24: Langevin Trajectory](langevin_trajectory.md)

</div>

## Example Structure

!!! info "Example Format"
    Each example follows a consistent structure to help you understand and apply the concepts:
    
    1. **Overview**: Brief explanation of the example and its purpose
    2. **Code**: Complete, runnable code for the example
    3. **Explanation**: Detailed explanation of key concepts and code sections
    4. **Extensions**: Suggestions for extending or modifying the example

## Running the Examples

All examples can be run directly from the command line:

```bash
# Clone the repository
git clone https://github.com/soran-ghaderi/torchebm.git
cd torchebm

# Set up your environment
pip install -e .

# Run an example
python examples/energy_fn_visualization.py
```

<div class="grid" markdown>
<div markdown>

## Prerequisites

To run these examples, you'll need:

- Python 3.7+
- PyTorch 1.9+
- NumPy
- Matplotlib

If you haven't installed TorchEBM yet, see the [Installation](../getting_started.md) guide.

</div>
<div markdown>

## GPU Acceleration

Most examples support GPU acceleration and will automatically use CUDA if available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
energy_fn = GaussianEnergy(mean, cov).to(device)
```

</div>
</div>

## Example Files

You can find all example files in the [examples directory](https://github.com/soran-ghaderi/torchebm/tree/master/examples) of the TorchEBM repository:

| File | Description |
|------|-------------|
| `energy_fn_visualization.py` | Visualizes various energy function landscapes |
| `langevin_dynamics_sampling.py` | Demonstrates Langevin dynamics sampling |
| `hmc_examples.py` | Shows usage of Hamiltonian Monte Carlo |
| `lagevin_sampler_trajectory.py` | Visualizes sampling trajectories |

## Additional Resources

For more in-depth information about the concepts demonstrated in these examples, see:

- [Energy Functions Guide](../guides/energy_functions.md)
- [Samplers Guide](../guides/samplers.md)
- [API Reference](../api/index.md)

## What's Next?

After exploring these examples, you might want to:

1. Check out the [API Reference](../api/index.md) for detailed documentation
2. Read the [Developer Guide](../developer_guide/index.md) to learn about contributing
3. Look at the [roadmap](../index.md#features--roadmap) for upcoming features 