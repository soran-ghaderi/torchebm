import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torchebm.core.base_model import BaseModel
from typing import Optional, Tuple, List, Union
import os


def plot_2d_energy_landscape(
    model: BaseModel,
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
    Plots a 2D energy landscape of a model.

    Args:
        model (BaseModel): The model to visualize.
        x_range (Tuple[float, float]): The range for the x-axis.
        y_range (Tuple[float, float]): The range for the y-axis.
        resolution (int): The number of points in each dimension.
        log_scale (bool): Whether to use a log scale for the energy values.
        cmap (str): The colormap to use.
        title (Optional[str]): The title of the plot.
        show_colorbar (bool): Whether to show a colorbar.
        save_path (Optional[str]): The path to save the figure.
        fig_size (Tuple[int, int]): The size of the figure.
        contour (bool): Whether to overlay contour lines.
        contour_levels (int): The number of contour levels.
        device (Optional[str]): The device to use for computation.

    Returns:
        plt.Figure: The matplotlib figure object.
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
        model = model.to(device)

    # Compute energy values
    with torch.no_grad():
        Z = model(grid).cpu().numpy()
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
    model: BaseModel,
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
    Plots a 3D surface visualization of a 2D energy landscape.

    Args:
        model (BaseModel): The model to visualize.
        x_range (Tuple[float, float]): The range for the x-axis.
        y_range (Tuple[float, float]): The range for the y-axis.
        resolution (int): The number of points in each dimension.
        log_scale (bool): Whether to use a log scale for the energy values.
        cmap (str): The colormap to use.
        title (Optional[str]): The title of the plot.
        show_colorbar (bool): Whether to show a colorbar.
        save_path (Optional[str]): The path to save the figure.
        fig_size (Tuple[int, int]): The size of the figure.
        alpha (float): The transparency of the surface.
        elev (float): The elevation angle for the 3D view.
        azim (float): The azimuth angle for the 3D view.
        device (Optional[str]): The device to use for computation.

    Returns:
        plt.Figure: The matplotlib figure object.
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
        model = model.to(device)

    # Compute energy values
    with torch.no_grad():
        Z = model(grid).cpu().numpy()
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
    model: BaseModel,
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
    Plots samples on a 2D energy landscape.

    Args:
        model (BaseModel): The model to visualize.
        samples (torch.Tensor): A tensor of samples of shape `(n_samples, 2)`.
        x_range (Tuple[float, float]): The range for the x-axis.
        y_range (Tuple[float, float]): The range for the y-axis.
        resolution (int): The number of points in each dimension.
        log_scale (bool): Whether to use a log scale for the energy values.
        cmap (str): The colormap to use.
        title (Optional[str]): The title of the plot.
        show_colorbar (bool): Whether to show a colorbar.
        save_path (Optional[str]): The path to save the figure.
        fig_size (Tuple[int, int]): The size of the figure.
        contour (bool): Whether to overlay contour lines.
        contour_levels (int): The number of contour levels.
        sample_color (str): The color of the samples.
        sample_alpha (float): The transparency of the samples.
        sample_size (float): The size of the sample markers.
        device (Optional[str]): The device to use for computation.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    fig = plot_2d_energy_landscape(
        model=model,
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
    model: Optional[BaseModel] = None,
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
    Plots sample trajectories, optionally on an energy landscape background.

    Args:
        trajectories (torch.Tensor): A tensor of trajectories of shape `(n_chains, n_steps, 2)`.
        model (Optional[BaseModel]): The model to visualize as a background.
        x_range (Optional[Tuple[float, float]]): The range for the x-axis. If `None`, it is
            inferred from the data.
        y_range (Optional[Tuple[float, float]]): The range for the y-axis. If `None`, it is
            inferred from the data.
        resolution (int): The number of points in each dimension for the energy grid.
        log_scale (bool): Whether to use a log scale for the energy values.
        cmap (str): The colormap to use for the energy background.
        title (Optional[str]): The title of the plot.
        show_colorbar (bool): Whether to show a colorbar.
        save_path (Optional[str]): The path to save the figure.
        fig_size (Tuple[int, int]): The size of the figure.
        trajectory_colors (Optional[List[str]]): A list of colors for the trajectories.
        trajectory_alpha (float): The transparency of the trajectory lines.
        line_width (float): The width of the trajectory lines.
        device (Optional[str]): The device to use for computation.

    Returns:
        plt.Figure: The matplotlib figure object.
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
    if model is not None:
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
            model = model.to(device)

        # Compute energy values
        with torch.no_grad():
            Z = model(grid).cpu().numpy()
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
