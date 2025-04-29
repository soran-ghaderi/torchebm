---
sidebar_position: 5
title: Visualization
description: Tools and techniques for visualizing energy landscapes and sampling results
---

# Visualization in TorchEBM

Data visualization is an essential tool for understanding, analyzing, and communicating the behavior of energy-based models. This guide covers various visualization techniques available in TorchEBM to help you gain insights into energy landscapes, sampling processes, and model performance.

## Energy Landscape Visualization

Visualizing energy landscapes is crucial for understanding the structure of the probability distribution you're working with. TorchEBM provides utilities to create both 2D and 3D visualizations of energy functions.

### Basic Energy Landscape Visualization

Here's a simple example to visualize a 2D energy function:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchebm.core import DoubleWellEnergy

# Create the energy function
energy_fn = DoubleWellEnergy(barrier_height=2.0)

# Create a grid for visualization
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Compute energy values
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32).unsqueeze(0)
        Z[i, j] = energy_fn(point).item()

# Create 3D surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Energy')
ax.set_title('Double Well Energy Landscape')
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()
```

![Basic Energy Landscape](../assets/images/visualization/basic_energy_landscape.png)

### Visualizing Energy as Probability Density

Often, it's more intuitive to visualize the probability density (exp(-Energy)) rather than the energy itself:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchebm.core import DoubleWellEnergy

# Create the energy function
energy_fn = DoubleWellEnergy(barrier_height=2.0)

# Create a grid for visualization
grid_size = 100
plot_range = 3.0
x_coords = np.linspace(-plot_range, plot_range, grid_size)
y_coords = np.linspace(-plot_range, plot_range, grid_size)
X, Y = np.meshgrid(x_coords, y_coords)
Z = np.zeros_like(X)

# Compute energy values
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32).unsqueeze(0)
        Z[i, j] = energy_fn(point).item()

# Convert energy to probability density (unnormalized)
# Subtract max for numerical stability before exponentiating
log_prob_values = -Z
log_prob_values = log_prob_values - np.max(log_prob_values)
prob_density = np.exp(log_prob_values)

# Create contour plot
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, prob_density, levels=50, cmap='viridis')
plt.colorbar(label='exp(-Energy) (unnormalized density)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Double Well Probability Density')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

![Probability Density](../assets/images/visualization/probability_density.png)

## Visualizing Model Training

When training EBMs, it's crucial to visualize both the energy landscape and the samples generated from the model. Here's a comprehensive example based on actual TorchEBM code used in practice:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchebm.core import BaseEnergyFunction
from torchebm.samplers import LangevinDynamics

@torch.no_grad()
def plot_energy_and_samples(
    energy_fn: BaseEnergyFunction,
    real_samples: torch.Tensor,
    sampler: LangevinDynamics,
    epoch: int,
    device: torch.device,
    grid_size: int = 100,
    plot_range: float = 3.0,
    k_sampling: int = 100,
):
    """Plots the energy surface, real data, and model samples."""
    plt.figure(figsize=(8, 8))

    # Create grid for energy surface plot
    x_coords = torch.linspace(-plot_range, plot_range, grid_size, device=device)
    y_coords = torch.linspace(-plot_range, plot_range, grid_size, device=device)
    xv, yv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    grid = torch.stack([xv.flatten(), yv.flatten()], dim=1)

    # Calculate energy on the grid
    energy_fn.eval()
    energy_values = energy_fn(grid).cpu().numpy().reshape(grid_size, grid_size)
    
    # Plot energy surface (using probability density for better visualization)
    log_prob_values = -energy_values
    log_prob_values = log_prob_values - np.max(log_prob_values)
    prob_density = np.exp(log_prob_values)

    plt.contourf(
        xv.cpu().numpy(),
        yv.cpu().numpy(),
        prob_density,
        levels=50,
        cmap="viridis",
    )
    plt.colorbar(label="exp(-Energy) (unnormalized density)")

    # Generate samples from the current model for visualization
    vis_start_noise = torch.randn(
        500, real_samples.shape[1], device=device
    )
    model_samples_tensor = sampler.sample(x=vis_start_noise, n_steps=k_sampling)
    model_samples = model_samples_tensor.cpu().numpy()

    # Plot real and model samples
    real_samples_np = real_samples.cpu().numpy()
    plt.scatter(
        real_samples_np[:, 0],
        real_samples_np[:, 1],
        s=10,
        alpha=0.5,
        label="Real Data",
        c="white",
        edgecolors="k",
        linewidths=0.5,
    )
    plt.scatter(
        model_samples[:, 0],
        model_samples[:, 1],
        s=10,
        alpha=0.5,
        label="Model Samples",
        c="red",
        edgecolors="darkred",
        linewidths=0.5,
    )

    plt.xlim(-plot_range, plot_range)
    plt.ylim(-plot_range, plot_range)
    plt.title(f"Epoch {epoch}")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"docs/assets/images/visualization/ebm_training_epoch_{epoch}.png")
    plt.close()
```

Then in your training loop:

```python
# At appropriate intervals during training
if (epoch + 1) % VISUALIZE_EVERY == 0 or epoch == 0:
    plot_energy_and_samples(
        energy_fn=energy_model,
        real_samples=real_data_for_plotting,
        sampler=sampler,
        epoch=epoch + 1,
        device=device,
        plot_range=2.5,
        k_sampling=200,
    )
```

![Training Progress](../assets/images/visualization/ebm_training_epoch_100.png)

## Tracking Loss During Training

Visualizing the training loss can provide insights into the convergence of your model:

```python
import matplotlib.pyplot as plt
import numpy as np

# During training, collect loss values
losses = []
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    # Training loop
    # ...
    avg_epoch_loss = epoch_loss / len(dataloader)
    losses.append(avg_epoch_loss)
    
    # Print loss
    print(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_epoch_loss:.4f}")

# After training, plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)
plt.savefig('docs/assets/images/visualization/training_loss.png')
plt.show()
```

![Training Loss](../assets/images/visualization/training_loss.png)

## Sampling Trajectory Visualization

Visualizing the trajectory of sampling algorithms can provide insights into their behavior and convergence properties.

### Visualizing Langevin Dynamics Trajectories

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchebm.core import DoubleWellEnergy
from torchebm.samplers import LangevinDynamics

# Create energy function and sampler
energy_fn = DoubleWellEnergy(barrier_height=2.0)
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.01
)

# We'll manually track the trajectory in a 2D space
dim = 2
n_steps = 1000
initial_point = torch.tensor([[-2.0, 0.0]], dtype=torch.float32)

# Run sampling with return_trajectory=True to get the full trajectory
trajectory = sampler.sample(
    x=initial_point,
    n_steps=n_steps,
    return_trajectory=True
)

# Prepare the background energy landscape
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32).unsqueeze(0)
        Z[i, j] = energy_fn(point).item()

# Plot contour with trajectory
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.7)
plt.colorbar(label='Energy')

# Extract trajectory coordinates
traj_x = trajectory[0, :, 0].numpy()
traj_y = trajectory[0, :, 1].numpy()

# Plot trajectory with colormap based on step number
points = plt.scatter(
    traj_x, traj_y,
    c=np.arange(len(traj_x)),
    cmap='plasma',
    s=5,
    alpha=0.7
)
plt.colorbar(points, label='Sampling Step')

# Plot arrows to show direction of trajectory
step = 50  # Plot an arrow every 50 steps
plt.quiver(
    traj_x[:-1:step], traj_y[:-1:step],
    traj_x[1::step] - traj_x[:-1:step], traj_y[1::step] - traj_y[:-1:step],
    scale_units='xy', angles='xy', scale=1, color='red', alpha=0.7
)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Langevin Dynamics Sampling Trajectory on Double Well Potential')
plt.tight_layout()
plt.savefig('docs/assets/images/visualization/langevin_trajectory_updated.png')
plt.show()
```

![Langevin Dynamics Trajectory](../assets/images/visualization/langevin_trajectory_updated.png)

### Visualizing Multiple Chains

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchebm.core import RastriginEnergy
from torchebm.samplers import LangevinDynamics

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create energy function and sampler
energy_fn = RastriginEnergy(a=10.0)
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.01
)

# Parameters for sampling
dim = 2
n_steps = 1000
num_chains = 5

# Generate random starting points
initial_points = torch.randn(num_chains, dim) * 3

# Run sampling and get trajectory
trajectories = sampler.sample(
    x=initial_points,
    n_steps=n_steps,
    return_trajectory=True
)

# Create background contour
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32).unsqueeze(0)
        Z[i, j] = energy_fn(point).item()

# Plot contour
plt.figure(figsize=(12, 10))
contour = plt.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.7)
plt.colorbar(label='Energy')

# Plot each trajectory with a different color
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i in range(num_chains):
    traj_x = trajectories[i, :, 0].numpy()
    traj_y = trajectories[i, :, 1].numpy()

    plt.plot(traj_x, traj_y, alpha=0.7, linewidth=1, c=colors[i],
             label=f'Chain {i + 1}')

    # Mark start and end points
    plt.scatter(traj_x[0], traj_y[0], c='black', s=50, marker='o')
    plt.scatter(traj_x[-1], traj_y[-1], c=colors[i], s=100, marker='*')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Multiple Langevin Dynamics Sampling Chains on Rastrigin Potential')
plt.legend()
plt.tight_layout()
plt.savefig('docs/assets/images/visualization/multiple_chains_updated.png')
plt.show()
```

![Multiple Sampling Chains](../assets/images/visualization/multiple_chains_updated.png)

## Distribution Visualization

Visualizing the distribution of samples can help assess the quality of your sampling algorithm.

### Comparing Generated Samples with Ground Truth

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from torchebm.core import GaussianEnergy
from torchebm.samplers import LangevinDynamics

# Create a Gaussian energy function
mean = torch.tensor([1.0, -1.0])
cov = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
energy_fn = GaussianEnergy(mean=mean, cov=cov)

# Sample using Langevin dynamics
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.01
)

# Generate samples
n_samples = 5000
burn_in = 200

# Initialize random samples
x = torch.randn(n_samples, 2)

# Perform sampling
samples = sampler.sample(
    x=x,
    n_steps=1000,
    burn_in=burn_in,
    return_trajectory=False
)

# Convert to numpy for visualization
samples_np = samples.numpy()
mean_np = mean.numpy()
cov_np = cov.numpy()

# Create a grid for the ground truth density
x = np.linspace(-3, 5, 100)
y = np.linspace(-5, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Calculate multivariate normal PDF
rv = stats.multivariate_normal(mean_np, cov_np)
Z = rv.pdf(pos)

# Create figure with multiple plots
fig = plt.figure(figsize=(15, 5))

# Ground truth contour
ax1 = fig.add_subplot(131)
ax1.contourf(X, Y, Z, 50, cmap='Blues')
ax1.set_title('Ground Truth Density')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Sample density (using kernel density estimation)
ax2 = fig.add_subplot(132)
h = ax2.hist2d(samples_np[:, 0], samples_np[:, 1], bins=50, cmap='Reds', density=True)
plt.colorbar(h[3], ax=ax2, label='Density')
ax2.set_title('Sampled Distribution')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

# Scatter plot of samples
ax3 = fig.add_subplot(133)
ax3.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5, s=3)
ax3.set_title('Sample Points')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_xlim(ax2.get_xlim())
ax3.set_ylim(ax2.get_ylim())

plt.tight_layout()
plt.savefig('docs/assets/images/visualization/distribution_comparison_updated.png')
plt.show()
```

![Distribution Comparison](../assets/images/visualization/distribution_comparison_updated.png)

## Energy Evolution Visualization

Tracking how energy values evolve during sampling can help assess convergence.

```python
import torch
import matplotlib.pyplot as plt
from torchebm.core import DoubleWellEnergy
from torchebm.samplers import LangevinDynamics

# Create energy function and sampler
energy_fn = DoubleWellEnergy(barrier_height=2.0)
sampler = LangevinDynamics(
    energy_function=energy_fn,
    step_size=0.01
)

# Parameters for sampling
dim = 2
n_steps = 1000
initial_point = torch.tensor([[-2.0, 0.0]], dtype=torch.float32)

# Track the energy during sampling
energy_values = []
current_sample = initial_point.clone()

# Run the sampling steps and store each energy
for i in range(n_steps):
    current_sample = sampler.step(current_sample)
    energy_values.append(energy_fn(current_sample).item())

# Convert to numpy for plotting
energy_values_np = np.array(energy_values)

# Plot energy evolution
plt.figure(figsize=(10, 6))
plt.plot(energy_values_np)
plt.xlabel('Step')
plt.ylabel('Energy')
plt.title('Energy Evolution During Langevin Dynamics Sampling')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('docs/assets/images/visualization/energy_evolution_updated.png')
plt.show()
```

![Energy Evolution](../assets/images/visualization/energy_evolution_updated.png)

## Visualizing Different Parameter Settings

Here's an example showing energy landscapes with different parameters:

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchebm.core import DoubleWellEnergy

# Create a grid for visualization
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Create barrier height values
barrier_heights = [0.5, 1.0, 2.0, 4.0]

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

# Calculate energy landscapes for different barrier heights
for i, barrier_height in enumerate(barrier_heights):
    # Create energy function with the specified barrier height
    energy_fn = DoubleWellEnergy(barrier_height=barrier_height)
    
    # Compute energy values
    for j in range(X.shape[0]):
        for k in range(X.shape[1]):
            point = torch.tensor([X[j, k], Y[j, k]], dtype=torch.float32).unsqueeze(0)
            Z[j, k] = energy_fn(point).item()
    
    # Create contour plot
    contour = axes[i].contourf(X, Y, Z, 50, cmap='viridis')
    fig.colorbar(contour, ax=axes[i], label='Energy')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')
    axes[i].set_title(f'Double Well Energy (Barrier Height = {barrier_height})')

plt.tight_layout()
plt.savefig('docs/assets/images/visualization/parameter_comparison.png')
plt.show()
```

![Parameter Comparison](../assets/images/visualization/parameter_comparison.png)

## Visualizing Training Progress with Different Loss Functions

You can also visualize how different loss functions affect the training dynamics:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchebm.core import BaseEnergyFunction
from torchebm.losses import ContrastiveDivergence, ScoreMatching
from torchebm.samplers import LangevinDynamics
from torchebm.datasets import TwoMoonsDataset
import torch.nn as nn
import torch.optim as optim

# Define a simple MLP energy function
class MLPEnergy(BaseEnergyFunction):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze(-1)

# Training function
def train_and_record_loss(loss_type, n_epochs=100):
    # Reset model
    energy_model = MLPEnergy(input_dim=2, hidden_dim=32).to(device)
    
    # Setup sampler and loss
    sampler = LangevinDynamics(
        energy_function=energy_model,
        step_size=0.1,
        device=device
    )
    
    if loss_type == 'CD':
        loss_fn = ContrastiveDivergence(
            energy_function=energy_model,
            sampler=sampler,
            k_steps=10,
            persistent=True
        )
    elif loss_type == 'SM':
        loss_fn = ScoreMatching(
            energy_function=energy_model,
            hutchinson_samples=5
        )
    
    optimizer = optim.Adam(energy_model.parameters(), lr=0.001)
    
    # Record loss
    losses = []
    
    # Train
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            loss = loss_fn(batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"{loss_type} - Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    return losses

# Setup data and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = TwoMoonsDataset(n_samples=1000, noise=0.1, device=device)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Train with different loss functions
cd_losses = train_and_record_loss('CD')
sm_losses = train_and_record_loss('SM')

# Plot losses
plt.figure(figsize=(10, 6))
plt.plot(cd_losses, label='Contrastive Divergence')
plt.plot(sm_losses, label='Score Matching')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('docs/assets/images/visualization/loss_comparison.png')
plt.show()
```

![Loss Comparison](../assets/images/visualization/loss_comparison.png)

## Conclusion

Effective visualization is key to understanding and debugging energy-based models. TorchEBM provides tools for visualizing energy landscapes, sampling trajectories, and model performance. These visualizations can help you gain insights into your models and improve their design and performance.

Remember to adapt these examples to your specific needs - you might want to visualize higher-dimensional spaces using dimensionality reduction techniques, or create specialized plots for your particular application. 