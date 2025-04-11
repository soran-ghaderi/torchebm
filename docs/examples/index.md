---
title: Examples
description: Practical examples demonstrating how to use TorchEBM
---

# TorchEBM Examples

This section contains practical examples that demonstrate how to use TorchEBM for energy-based modeling. Each example is fully tested and focuses on a specific use case or feature.

<div class="grid cards" markdown>

-   :material-terrain:{ .lg .middle } __Energy Functions__

    ---

    Explore and visualize various energy functions and their properties.

    [:octicons-arrow-right-24: Energy Functions](core/energy_functions.md)

-   :material-sigma:{ .lg .middle } __Langevin Dynamics__

    ---

    Sample from various distributions using Langevin dynamics.

    [:octicons-arrow-right-24: Langevin Dynamics](samplers/langevin.md)

-   :material-axis-arrow:{ .lg .middle } __Hamiltonian Monte Carlo__

    ---

    Learn to use Hamiltonian Monte Carlo for efficient sampling.

    [:octicons-arrow-right-24: Hamiltonian Monte Carlo](samplers/hmc.md)

-   :material-chart-line:{ .lg .middle } __Visualization Tools__

    ---

    Advanced visualization tools for energy landscapes and sampling results.

    [:octicons-arrow-right-24: Visualization](visualization/index.md)

</div>

## Example Structure

!!! info "Example Format"
    Each example follows a consistent structure to help you understand and apply the concepts:
    
    1. **Overview**: Brief explanation of the example and its purpose
    2. **Code**: Complete, runnable code for the example
    3. **Explanation**: Detailed explanation of key concepts and code sections
    4. **Extensions**: Suggestions for extending or modifying the example

## Running the Examples

All examples can be run using the examples main.py script:

```bash
# Clone the repository
git clone https://github.com/soran-ghaderi/torchebm.git
cd torchebm

# Set up your environment
pip install -e .

# List all available examples
python examples/main.py --list

# Run a specific example
python examples/main.py samplers/langevin/visualization_trajectory
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

## Key Example Files

You'll find examples organized into the following categories:

| Category | Description | Key Files |
|----------|-------------|-----------|
| `core/energy_functions/` | Energy function visualization and properties | `landscape_2d.py`, `multimodal.py`, `parametric.py` |
| `samplers/langevin/` | Langevin dynamics sampling examples | `gaussian_sampling.py`, `multimodal_sampling.py`, `visualization_trajectory.py` |
| `samplers/hmc/` | Hamiltonian Monte Carlo examples | `gaussian_sampling.py`, `advanced.py`, `mass_matrix.py` |
| `visualization/` | Advanced visualization tools | `basic/contour_plots.py`, `advanced/trajectory_animation.py`, `advanced/parallel_chains.py` |

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