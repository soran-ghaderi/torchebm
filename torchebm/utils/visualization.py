import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torchebm.core.base_energy_function import BaseEnergyFunction
from typing import Optional, Tuple, List, Union
import os


def plot_2d_energy_landscape(
    energy_fn: BaseEnergyFunction,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    resolution: int = 100,
    log_scale: bool = False,
    cmap: str = "viridis",
    title: Optional[str] = None,
    show_colorbar: bool = True,
    save_path: Optional[str] = None,
    fig_size: Tuple[int, int] = (8, 6),
    contour: bool = True,
    contour_levels: int = 20,
    device: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a 2D energy landscape.

    Args:
        energy_fn: The energy function to visualize
        x_range: Range for the x-axis
        y_range: Range for the y-axis
        resolution: Number of points in each dimension
        log_scale: Whether to use log scale for the energy values
        cmap: Colormap to use
        title: Title of the plot
        show_colorbar: Whether to show a colorbar
        save_path: Path to save the figure, if not None
        fig_size: Size of the figure
        contour: Whether to overlay contour lines
        contour_levels: Number of contour levels
        device: Device to use for computation

    Returns:
        The matplotlib figure object
    """
    # Create the grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Convert to pytorch tensor
    grid = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32
    )
    if device is not None:
        grid = grid.to(device)
        energy_fn = energy_fn.to(device)

    # Compute energy values
    with torch.no_grad():
        Z = energy_fn(grid).cpu().numpy()
    Z = Z.reshape(X.shape)

    # Apply log scale if requested
    if log_scale:
        # Add a small constant to avoid log(0)
        Z = np.log(Z + 1e-10)

    # Create the figure
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot the surface
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto")

    # Overlay contour lines if requested
    if contour:
        contour_plot = ax.contour(
            X, Y, Z, levels=contour_levels, colors="white", alpha=0.5, linewidths=0.5
        )

    # Add colorbar if requested
    if show_colorbar:
        fig.colorbar(im, ax=ax, label="Energy" if not log_scale else "Log Energy")

    # Set title and labels
    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Save figure if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_3d_energy_landscape(
    energy_fn: BaseEnergyFunction,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    resolution: int = 50,
    log_scale: bool = False,
    cmap: str = "viridis",
    title: Optional[str] = None,
    show_colorbar: bool = True,
    save_path: Optional[str] = None,
    fig_size: Tuple[int, int] = (10, 8),
    alpha: float = 0.9,
    elev: float = 30,
    azim: float = -45,
    device: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a 3D surface visualization of a 2D energy landscape.

    Args:
        energy_fn: The energy function to visualize
        x_range: Range for the x-axis
        y_range: Range for the y-axis
        resolution: Number of points in each dimension
        log_scale: Whether to use log scale for the energy values
        cmap: Colormap to use
        title: Title of the plot
        show_colorbar: Whether to show a colorbar
        save_path: Path to save the figure, if not None
        fig_size: Size of the figure
        alpha: Transparency of the surface
        elev: Elevation angle for the 3D view
        azim: Azimuth angle for the 3D view
        device: Device to use for computation

    Returns:
        The matplotlib figure object
    """
    # Create the grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Convert to pytorch tensor
    grid = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32
    )
    if device is not None:
        grid = grid.to(device)
        energy_fn = energy_fn.to(device)

    # Compute energy values
    with torch.no_grad():
        Z = energy_fn(grid).cpu().numpy()
    Z = Z.reshape(X.shape)

    # Apply log scale if requested
    if log_scale:
        # Add a small constant to avoid log(0)
        Z = np.log(Z + 1e-10)

    # Create the figure
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface
    surf = ax.plot_surface(
        X, Y, Z, cmap=cmap, alpha=alpha, linewidth=0, antialiased=True
    )

    # Add colorbar if requested
    if show_colorbar:
        fig.colorbar(
            surf,
            ax=ax,
            shrink=0.5,
            aspect=5,
            label="Energy" if not log_scale else "Log Energy",
        )

    # Set title and labels
    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Energy" if not log_scale else "Log Energy")

    # Set view angle
    ax.view_init(elev=elev, azim=azim)

    # Save figure if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_samples_on_energy(
    energy_fn: BaseEnergyFunction,
    samples: torch.Tensor,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    resolution: int = 100,
    log_scale: bool = False,
    cmap: str = "viridis",
    title: Optional[str] = None,
    show_colorbar: bool = True,
    save_path: Optional[str] = None,
    fig_size: Tuple[int, int] = (8, 6),
    contour: bool = True,
    contour_levels: int = 20,
    sample_color: str = "red",
    sample_alpha: float = 0.5,
    sample_size: float = 5,
    device: Optional[str] = None,
) -> plt.Figure:
    """
    Plot samples on a 2D energy landscape.

    Args:
        energy_fn: The energy function to visualize
        samples: Tensor of samples with shape [n_samples, 2]
        x_range: Range for the x-axis
        y_range: Range for the y-axis
        resolution: Number of points in each dimension
        log_scale: Whether to use log scale for the energy values
        cmap: Colormap to use
        title: Title of the plot
        show_colorbar: Whether to show a colorbar
        save_path: Path to save the figure, if not None
        fig_size: Size of the figure
        contour: Whether to overlay contour lines
        contour_levels: Number of contour levels
        sample_color: Color of the samples
        sample_alpha: Transparency of the samples
        sample_size: Size of the sample markers
        device: Device to use for computation

    Returns:
        The matplotlib figure object
    """
    fig = plot_2d_energy_landscape(
        energy_fn=energy_fn,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        log_scale=log_scale,
        cmap=cmap,
        title=title,
        show_colorbar=show_colorbar,
        fig_size=fig_size,
        contour=contour,
        contour_levels=contour_levels,
        device=device,
    )

    # Get the current axis
    ax = plt.gca()

    # Plot the samples
    samples_np = samples.detach().cpu().numpy()
    ax.scatter(
        samples_np[:, 0],
        samples_np[:, 1],
        color=sample_color,
        alpha=sample_alpha,
        s=sample_size,
    )

    # Save figure if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_sample_trajectories(
    trajectories: torch.Tensor,
    energy_fn: Optional[BaseEnergyFunction] = None,
    x_range: Tuple[float, float] = None,
    y_range: Tuple[float, float] = None,
    resolution: int = 100,
    log_scale: bool = False,
    cmap: str = "viridis",
    title: Optional[str] = None,
    show_colorbar: bool = True,
    save_path: Optional[str] = None,
    fig_size: Tuple[int, int] = (8, 6),
    trajectory_colors: Optional[List[str]] = None,
    trajectory_alpha: float = 0.7,
    line_width: float = 1.0,
    device: Optional[str] = None,
) -> plt.Figure:
    """
    Plot sample trajectories, optionally on an energy landscape background.

    Args:
        trajectories: Tensor of trajectories with shape [n_chains, n_steps, 2]
        energy_fn: The energy function to visualize as background (optional)
        x_range: Range for the x-axis (if None, determined from data)
        y_range: Range for the y-axis (if None, determined from data)
        resolution: Number of points in each dimension for energy grid
        log_scale: Whether to use log scale for the energy values
        cmap: Colormap to use for energy background
        title: Title of the plot
        show_colorbar: Whether to show a colorbar
        save_path: Path to save the figure, if not None
        fig_size: Size of the figure
        trajectory_colors: List of colors for trajectories (if None, automatically chosen)
        trajectory_alpha: Transparency of the trajectory lines
        line_width: Width of the trajectory lines
        device: Device to use for computation

    Returns:
        The matplotlib figure object
    """
    # Determine plotting ranges if not provided
    if x_range is None or y_range is None:
        all_data = trajectories.detach().cpu().numpy().reshape(-1, 2)
        data_min = all_data.min(axis=0)
        data_max = all_data.max(axis=0)
        padding = (data_max - data_min) * 0.1  # Add 10% padding

        if x_range is None:
            x_range = (data_min[0] - padding[0], data_max[0] + padding[0])
        if y_range is None:
            y_range = (data_min[1] - padding[1], data_max[1] + padding[1])

    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot energy landscape if provided
    if energy_fn is not None:
        # Create the grid
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        # Convert to pytorch tensor
        grid = torch.tensor(
            np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32
        )
        if device is not None:
            grid = grid.to(device)
            energy_fn = energy_fn.to(device)

        # Compute energy values
        with torch.no_grad():
            Z = energy_fn(grid).cpu().numpy()
        Z = Z.reshape(X.shape)

        # Apply log scale if requested
        if log_scale:
            # Add a small constant to avoid log(0)
            Z = np.log(Z + 1e-10)

        # Plot the surface
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto")

        # Add colorbar if requested
        if show_colorbar:
            fig.colorbar(im, ax=ax, label="Energy" if not log_scale else "Log Energy")

    # Plot trajectories
    n_chains = trajectories.shape[0]
    if trajectory_colors is None:
        trajectory_colors = plt.cm.tab10(np.linspace(0, 1, n_chains))

    trajectories_np = trajectories.detach().cpu().numpy()
    for i, trajectory in enumerate(trajectories_np):
        color = trajectory_colors[i] if i < len(trajectory_colors) else "gray"
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=color,
            alpha=trajectory_alpha,
            linewidth=line_width,
        )
        ax.scatter(
            trajectory[0, 0],
            trajectory[0, 1],
            color=color,
            marker="o",
            s=30,
            label=f"Start {i+1}",
        )
        ax.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            color=color,
            marker="x",
            s=50,
            label=f"End {i+1}",
        )

    # Set title and labels
    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Set limits
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # Add legend for first chain only to avoid cluttering
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        # Only show legend for the first trajectory's start/end
        ax.legend(handles[:2], labels[:2], loc="best")

    # Save figure if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
