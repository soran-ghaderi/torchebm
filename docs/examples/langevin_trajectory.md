---
title: Langevin Sampler Trajectory
description: Visualizing sampling trajectories on energy landscapes
---

# Langevin Sampler Trajectory

This example demonstrates how to visualize the trajectories of Langevin dynamics samplers on multimodal energy landscapes.

!!! abstract "Key Concepts Covered"
    - Creating custom energy functions
    - Visualizing energy landscapes
    - Tracking and plotting sampling trajectories
    - Working with multimodal distributions

## Overview

Visualizing sampling trajectories helps understand how different sampling algorithms explore the energy landscape. This example creates a multimodal energy function and visualizes multiple sampling chains as they traverse the landscape.

## Multimodal Energy Function

First, we define a custom energy function with multiple local minima:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchebm.samplers.langevin_dynamics import LangevinDynamics

class MultimodalEnergy:
    """
    A 2D energy function with multiple local minima to demonstrate sampling behavior.
    """

    def __init__(self, device=None, dtype=torch.float32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Define centers and weights for multiple Gaussian components
        self.centers = torch.tensor(
            [[-1.0, -1.0], [1.0, 1.0], [-0.5, 1.0], [1.0, -0.5]],
            device=self.device,
            dtype=self.dtype,
        )

        self.weights = torch.tensor(
            [1.0, 0.8, 0.6, 0.7], device=self.device, dtype=self.dtype
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has correct dtype and shape
        x = x.to(dtype=self.dtype)
        if x.dim() == 1:
            x = x.view(1, -1)

        # Calculate distance to each center
        dists = torch.cdist(x, self.centers)

        # Calculate energy as negative log of mixture of Gaussians
        energy = -torch.log(
            torch.sum(self.weights * torch.exp(-0.5 * dists.pow(2)), dim=-1)
        )

        return energy

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has correct dtype and shape
        x = x.to(dtype=self.dtype)
        if x.dim() == 1:
            x = x.view(1, -1)

        # Calculate distances and Gaussian components
        diff = x.unsqueeze(1) - self.centers
        exp_terms = torch.exp(-0.5 * torch.sum(diff.pow(2), dim=-1))
        weights_exp = self.weights * exp_terms

        # Calculate gradient
        normalizer = torch.sum(weights_exp, dim=-1, keepdim=True)
        gradient = -torch.sum(
            weights_exp.unsqueeze(-1) * diff / normalizer.unsqueeze(-1), dim=1
        )

        return gradient.squeeze()  # Ensure consistent output shape

    def to(self, device):
        self.device = device
        self.centers = self.centers.to(device)
        self.weights = self.weights.to(device)
        return self
```

## Modified Langevin Dynamics Sampler

Next, we create a slightly modified version of the Langevin dynamics sampler to ensure consistent tensor shapes during trajectory tracking:

```python
class ModifiedLangevinDynamics(LangevinDynamics):
    """
    Modified version of LangevinDynamics to ensure consistent tensor shapes
    """

    def sample(
        self, initial_state, n_steps, return_trajectory=False, return_diagnostics=False
    ):
        current_state = initial_state.clone()

        if return_trajectory:
            trajectory = [current_state.view(1, -1)]  # Ensure consistent shape

        diagnostics = {"energies": []} if return_diagnostics else None

        for _ in range(n_steps):
            # Calculate gradient
            grad = self.energy_function.gradient(current_state)

            # Add noise
            noise = torch.randn_like(current_state) * self.noise_scale

            # Update state
            current_state = current_state - self.step_size * grad + noise

            if return_trajectory:
                trajectory.append(current_state.view(1, -1))  # Ensure consistent shape

            if return_diagnostics:
                diagnostics["energies"].append(
                    self.energy_function(current_state).item()
                )

        if return_trajectory:
            result = torch.cat(trajectory, dim=0)  # Use cat instead of stack
        else:
            result = current_state

        if return_diagnostics:
            diagnostics["energies"] = torch.tensor(diagnostics["energies"])

        return result, diagnostics if return_diagnostics else None
```

## Visualization Function

Finally, we write a function to visualize the energy landscape and sampling trajectories:

```python
def visualize_energy_landscape_and_sampling():
    # Set up device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Create energy function with explicit dtype
    energy_fn = MultimodalEnergy(device=device, dtype=dtype)

    # Create modified sampler
    sampler = ModifiedLangevinDynamics(
        energy_function=energy_fn, step_size=0.01, noise_scale=0.1, device=device
    )

    # Create grid for energy landscape visualization
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # Calculate energy values with explicit dtype
    grid_points = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1), device=device, dtype=dtype
    )

    energy_values = energy_fn(grid_points).cpu().numpy().reshape(X.shape)

    # Generate samples with trajectory tracking
    n_chains = 5
    initial_states = torch.randn(n_chains, 2, device=device, dtype=dtype) * 2

    trajectories = []
    for init_state in initial_states:
        trajectory, _ = sampler.sample(
            initial_state=init_state, n_steps=200, return_trajectory=True
        )
        trajectories.append(trajectory.cpu().numpy())

    # Plotting
    plt.figure(figsize=(12, 10))

    # Plot energy landscape
    contour = plt.contour(X, Y, energy_values, levels=20, cmap="viridis")
    plt.colorbar(contour, label="Energy")

    # Plot sampling trajectories
    colors = plt.cm.rainbow(np.linspace(0, 1, n_chains))
    for idx, (trajectory, color) in enumerate(zip(trajectories, colors)):
        plt.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            "o-",
            color=color,
            alpha=0.5,
            markersize=2,
            label=f"Chain {idx+1}",
        )
        plt.plot(trajectory[0, 0], trajectory[0, 1], "o", color=color, markersize=8)
        plt.plot(
            trajectory[-1, 0], trajectory[-1, 1], "*", color=color, markersize=12
        )

    plt.title("Energy Landscape and Langevin Dynamics Sampling Trajectories")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.grid(True)
    plt.legend()
    plt.show()
```

## Running the Example

To run the example, simply execute:

```python
if __name__ == "__main__":
    print("Running energy landscape visualization...")
    visualize_energy_landscape_and_sampling()
```

## Expected Results

<div class="grid" markdown>
<div markdown>

When you run this example, you'll see a contour plot of the energy landscape with multiple chains of Langevin dynamics samples overlaid. The visualization shows:

- **Energy landscape**: Contour lines representing the multimodal energy function
- **Multiple sampling chains**: Different colored trajectories starting from random initial points
- **Trajectory progression**: You can see how samples move from high-energy regions to low-energy regions

</div>
<div markdown>

The key insights from this visualization:

1. Sampling chains are attracted to areas of low energy (high probability)
2. Chains can get trapped in local minima and have difficulty crossing energy barriers
3. The stochastic nature of Langevin dynamics helps chains occasionally escape local minima
4. Sampling efficiency depends on starting points and energy landscape geometry

</div>
</div>

## Understanding Multimodal Sampling

Multimodal distributions present special challenges for sampling algorithms:

!!! info "Challenges in Multimodal Sampling"
    1. **Energy barriers**: Chains must overcome barriers between modes
    2. **Mode-hopping**: Chains may have difficulty transitioning between distant modes
    3. **Mixing time**: The time required to adequately explore all modes increases
    4. **Mode coverage**: Some modes may be missed entirely during finite sampling

The visualization helps understand these challenges by showing:

- How chains explore the space around each mode
- Whether chains successfully transition between modes
- If certain modes are favored over others
- The impact of initialization on the final sampling distribution

## Extensions and Variations

This example can be extended in various ways:

1. **Compare different samplers**: Add HMC or other samplers for comparison
2. **Vary step size and noise**: Show the impact of different parameters
3. **Use more complex energy functions**: Create energy functions with more challenging landscapes
4. **Implement adaptive step sizes**: Show how adaptive methods improve sampling efficiency
5. **Add diagnostics visualization**: Plot energy evolution and other metrics alongside trajectories

## Conclusion

Visualizing sampling trajectories provides valuable insights into the behavior of sampling algorithms and the challenges they face when exploring complex energy landscapes. This understanding can help in selecting and tuning appropriate sampling methods for different problems.

[//]: # (## Visualization Results)

[//]: # ()
[//]: # (When running the example, you'll see a visualization of the energy landscape with multiple sampling chains:)

[//]: # ()
[//]: # (![Energy Landscape and Langevin Sampling Trajectories]&#40;../assets/images/examples/langevin_trajectory.png&#41;)

[//]: # ()
[//]: # (*This visualization shows a multimodal energy landscape &#40;contour lines&#41; with five independent Langevin dynamics sampling chains &#40;colored trajectories&#41;. Each chain starts from a different random position &#40;marked by a circle&#41; and evolves through 200 steps &#40;ending at the stars&#41;. The trajectories show how the chains are attracted to the four local minima &#40;dark blue regions&#41; of the energy function. Note how some chains get trapped in local minima while others manage to explore multiple modes of the distribution.* )