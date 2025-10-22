---
title: Visualization
description: Techniques for visualizing energy landscapes, sampling trajectories, and model performance in TorchEBM.
icon: material/chart-bar
---

# Visualization in TorchEBM

Visualizing the behavior of energy-based models is essential for understanding and debugging them. This guide covers key visualization techniques for EBMs, focusing on energy landscapes and sampler trajectories.

## Visualizing 2D Energy Landscapes

For models that operate on 2D data, we can directly visualize the energy function as a surface or contour plot. This shows us where the model has learned to assign low energy (high probability).

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchebm.core import DoubleWellModel

model = DoubleWellModel(barrier_height=2.0)

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)

with torch.no_grad():
    energy_values = model(grid_points).numpy().reshape(X.shape)

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, energy_values, levels=50, cmap='viridis')
plt.colorbar(label='Energy')
plt.title('Energy Landscape of a Double Well Model')
plt.show()
```

<figure markdown>
  ![Double Well Energy Function](../../assets/images/e_functions/double_well.png){ width="500" }
  <figcaption>A 2D contour plot of the `DoubleWellModel` energy landscape.</figcaption>
</figure>

## Visualizing Sampling Trajectories

To understand how samplers explore the state space, we can plot their trajectories on top of the energy landscape. This is particularly insightful for complex, multimodal distributions.

```python
from torchebm.samplers import LangevinDynamics
from torchebm.core import MultimodalModel # A custom model with 4 modes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalModel().to(device)
sampler = LangevinDynamics(model=model, step_size=0.1)

initial_particles = torch.zeros(5, 2, device=device) # 5 chains starting at the origin
trajectory = sampler.sample(x=initial_particles, n_steps=200, return_trajectory=True)

# Plotting (on top of the energy landscape from the previous example)
# (Contour plot code omitted for brevity)
colors = plt.cm.viridis(np.linspace(0, 1, 5))
for i in range(5):
    traj_chain = trajectory[i].cpu().numpy()
    plt.plot(traj_chain[:, 0], traj_chain[:, 1], color=colors[i], alpha=0.7)
    plt.scatter(traj_chain[0, 0], traj_chain[0, 1], color='red', s=50, zorder=3) # Start
    plt.scatter(traj_chain[-1, 0], traj_chain[-1, 1], color='blue', s=50, zorder=3) # End
plt.show()
```

<figure markdown>
  ![Langevin Dynamics Sampling Trajectories](../../assets/images/examples/langevin_trajectory.png){ width="500" }
  <figcaption>Trajectories of five Langevin Dynamics chains exploring a multimodal landscape.</figcaption>
</figure>

## Comparing Ground Truth and Model Samples

A critical evaluation is to compare the distribution of samples from the trained model against the real data distribution.

```python
# Assume `model_samples` are generated from a trained model
# Assume `real_samples` are from the dataset
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.set_title("Real Data Distribution")
ax1.scatter(real_samples[:, 0], real_samples[:, 1], s=10, alpha=0.5)

ax2.set_title("Model Sample Distribution")
ax2.scatter(model_samples[:, 0], model_samples[:, 1], s=10, alpha=0.5, c='red')

plt.show()
```
This side-by-side comparison provides a quick qualitative assessment of how well the model has learned the target distribution. For more quantitative measures, you can use metrics like Maximum Mean Discrepancy (MMD) or analyze summary statistics.