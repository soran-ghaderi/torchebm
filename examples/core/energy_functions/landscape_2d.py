import numpy as np
import torch
from matplotlib import pyplot as plt

from torchebm.core.base_energy_function import (
    RosenbrockEnergy,
    AckleyEnergy,
    RastriginEnergy,
    DoubleWellEnergy,
    GaussianEnergy,
    HarmonicEnergy,
)


def plot_energy_function(energy_fn, x_range, y_range, title):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32).unsqueeze(0)
            Z[i, j] = energy_fn(point).item()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Energy")
    plt.show()


energy_functions = [
    (RosenbrockEnergy(), [-2, 2], [-1, 3], "Rosenbrock Energy Function"),
    (AckleyEnergy(), [-5, 5], [-5, 5], "Ackley Energy Function"),
    (RastriginEnergy(), [-5, 5], [-5, 5], "Rastrigin Energy Function"),
    (DoubleWellEnergy(), [-2, 2], [-2, 2], "Double Well Energy Function"),
    (
        GaussianEnergy(
            torch.tensor([0.0, 0.0]), torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        ),
        [-3, 3],
        [-3, 3],
        "Gaussian Energy Function",
    ),
    (HarmonicEnergy(), [-3, 3], [-3, 3], "Harmonic Energy Function"),
]

# Plot each energy function
for energy_fn, x_range, y_range, title in energy_functions:
    plot_energy_function(energy_fn, x_range, y_range, title)
