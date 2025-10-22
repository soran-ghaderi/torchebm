---
title: Examples
description: Practical examples demonstrating how to use TorchEBM
icon: fontawesome/regular/image
---

# TorchEBM Examples

This section contains practical examples that demonstrate how to use TorchEBM for energy-based modeling. Each example is fully tested and focuses on a specific use case or feature.

<div class="grid cards" markdown>

- [:material-function-variant:{ .lg .middle } Models and Energy Functions](models/index.md)
- [:material-database-search:{ .lg .middle } Datasets](datasets/index.md)
- [:material-chart-scatter-plot:{ .lg .middle } Samplers](samplers/index.md)
- [:material-chart-line:{ .lg .middle } Training an EBM](training/index.md)
- [:material-chart-bar:{ .lg .middle } Visualization](visualization/index.md)

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

If you haven't installed TorchEBM yet, see the [Installation](../tutorials/index.md) guide.

</div>
<div markdown>

## GPU Acceleration

Most examples support GPU acceleration and will automatically use CUDA if available:

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
energy_fn = GaussianEnergy(mean, cov).to(device)
```

</div>
</div>

## What's Next?

After exploring these examples, you might want to:

1. Check out the [API Reference](../api/index.md) for detailed documentation
2. Read the [Developer Guide](../developer_guide/index.md) to learn about contributing
3. Look at the [roadmap](../index.md#features--roadmap) for upcoming features 