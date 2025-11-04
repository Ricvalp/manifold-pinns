from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D





def plot_u0(x, y, ics, name=None):

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(x, y, s=3, c=ics)
    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()


def _select_snapshot_indices(num_items: int, num_targets: int) -> np.ndarray:
    """Pick evenly spaced indices for visualization."""
    if num_items <= 0:
        return np.array([], dtype=int)
    num_targets = min(max(num_targets, 1), num_items)
    if num_targets == 1:
        return np.array([num_items - 1], dtype=int)
    return np.linspace(0, num_items - 1, num_targets, dtype=int)


def plot_results(
    x: np.ndarray,
    y: np.ndarray,
    times: Sequence[float],
    predictions: Sequence[Sequence[float]],
    *,
    save_dir,
    prefix: str = "solution",
    num_snapshots: int = 10,
    log_to_wandb: bool = False,
    wandb_key: str = "solution_snapshots",
    wandb_step: int | None = None,
) -> list[Path]:
    """Save and optionally log solution snapshots with per-plot color scales."""
    times = np.asarray(times)
    preds = np.asarray(predictions)
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    indices = _select_snapshot_indices(len(times), num_snapshots)
    saved_paths: list[Path] = []
    wandb_images = []

    for rank, idx in enumerate(indices):
        t_val = times[idx]
        values = preds[idx]

        fig, ax = plt.subplots(figsize=(4, 4))
        scatter = ax.scatter(
            x,
            y,
            c=values,
            cmap="viridis",
            s=6,
            vmin=float(np.min(values)),
            vmax=float(np.max(values)),
        )
        ax.set_title(f"t = {t_val:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        filename = f"{prefix}_snapshot_{rank:02d}.png"
        output_path = save_dir / filename
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

        saved_paths.append(output_path)

        if log_to_wandb:
            import wandb  # Local import to avoid mandatory dependency.

            wandb_images.append(wandb.Image(str(output_path), caption=f"t = {t_val:.2f}"))

    if log_to_wandb and wandb_images:
        import wandb

        wandb.log({wandb_key: wandb_images}, step=wandb_step)

    return saved_paths


def plot_domains(x, y, boundaries_x, boundaries_y, ics, name=None):
    num_plots = len(x)
    cols = 4  # You can adjust the number of columns based on your preference
    rows = (num_plots + cols - 1) // cols  # Calculate required rows

    fig, ax = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    # Ensure ax is a 2D array for easy indexing
    if num_plots == 1:
        ax = [[ax]]
    elif cols == 1 or rows == 1:
        ax = ax.reshape(-1, cols)

    for i in range(num_plots):
        # Calculate row and column index for the plot
        row, col = divmod(i, cols)

        ax[row][col].set_title(f"Chart {i}")
        ax[row][col].scatter(x[i], y[i], s=3, c=ics[i])

        # Plot boundaries for current chart
        if i in boundaries_x:
            for other_chart, boundary_x in boundaries_x[i].items():
                ax[row][col].scatter(
                    boundary_x,
                    boundaries_y[i][other_chart],
                    s=10,
                    label=f"boundary {i}-{other_chart}",
                )

        ax[row][col].legend(loc="best")

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_domains_with_metric(x, y, sqrt_det_g, conditionings, name=None):
    num_plots = min(len(conditionings), 16)
    
    cols = 4
    rows = (num_plots + cols - 1) // cols

    fig, ax = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_plots == 1:
        ax = [[ax]]
    elif cols == 1 or rows == 1:
        ax = ax.reshape(-1, cols)

    for i in range(num_plots):
        row, col = divmod(i, cols)

        ax[row][col].set_title(f"Chart {i}")
        color_values = sqrt_det_g(conditionings[i], jnp.stack([x, y], axis=1))
        scatter = ax[row][col].scatter(x, y, s=3, c=color_values, cmap="viridis")
        fig.colorbar(scatter, ax=ax[row][col], orientation="vertical")

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_domains_3d(x, y, ics, decoder, d_params, name=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    decoder_params = [jax.tree_map(lambda x: x[i], d_params) for i in range(len(x))]
    for i in range(len(x)):
        p = decoder.apply({"params": decoder_params[i]}, np.stack([x[i], y[i]], axis=1))
        color = plt.cm.tab10(i)
        ax.scatter(
            p[:, 0],
            p[:, 1],
            p[:, 2],
            c=ics[i],
        )
    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_domains_3d_with_metric(x, y, decoder, sqrt_det_g, d_params, name=None):
    num_plots = len(x)
    cols = 2
    rows = (num_plots + cols - 1) // cols

    fig = plt.figure(figsize=(15, 5 * rows))

    decoder_params = [jax.tree_map(lambda x: x[i], d_params) for i in range(len(x))]
    for i in range(num_plots):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.set_title(f"Chart {i}")
        points_3d = decoder.apply(
            {"params": decoder_params[i]}, jnp.stack([x[i], y[i]], axis=1)
        )
        x_3d, y_3d, z_3d = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

        color_values = sqrt_det_g(decoder_params[i], jnp.stack([x[i], y[i]], axis=1))

        scatter = ax.scatter(x_3d, y_3d, z_3d, c=color_values, cmap="viridis", s=3)

        ax.legend(loc="best")
        fig.colorbar(scatter, ax=ax, orientation="vertical")

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_combined_3d_with_metric(x, y, decoder, sqrt_det_g, d_params, name=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Combined 3D Plot with Metric Coloring")

    decoder_params = [jax.tree_map(lambda x: x[i], d_params) for i in range(len(x))]
    for i in range(len(x)):
        # Decode the 2D points to 3D using the decoder function
        points_3d = decoder.apply(
            {"params": decoder_params[i]}, jnp.stack([x[i], y[i]], axis=1)
        )
        x_3d, y_3d, z_3d = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

        # Calculate the color values using sqrt_det_gs
        color_values = sqrt_det_g(decoder_params[i], jnp.stack([x[i], y[i]], axis=1))

        scatter = ax.scatter(
            x_3d, y_3d, z_3d, c=color_values, cmap="viridis", s=3, label=f"Chart {i}"
        )

    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, orientation="vertical")

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()



def plot_charts_sequence(charts, ts, name=None):
    num_charts = len(charts)
    cols = min(5, num_charts)
    rows = (num_charts + cols - 1) // cols
    
    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    
    for i, chart in enumerate(charts[::10]):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.set_title(f"t={ts[i]:.2f}")
        ax.scatter(
            chart[:, 0],
            chart[:, 1],
            chart[:, 2],
            s=3
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_zlim(-1, 1)
    
    plt.tight_layout()
    
    if name is not None:
        plt.savefig(name)
    plt.show()
    plt.close()


def plot_charts_sequence_with_solution(charts, ts, values=None, figsize=(15, 10), dpi=300, 
                         cmap='viridis', alpha=0.8, s=5, view_angle=(30, 45),
                         save_path=None, every_n=10, zlim=None, 
                         show_colorbar=False, show_time=True, vmin=None, vmax=None,
                         max_time_fraction=1.0):
    """
    Plot a sequence of 3D charts over time with publication-quality styling.
    
    Parameters:
    -----------
    charts : list of numpy.ndarray
        List of point clouds at different time steps
    ts : numpy.ndarray
        Time values corresponding to each chart
    values : list of numpy.ndarray, optional
        Values to color the points by at each time step (if None, uses z-coordinate)
    figsize : tuple, optional
        Figure size in inches (width, height)
    dpi : int, optional
        Resolution in dots per inch
    cmap : str, optional
        Colormap for the points (default: 'viridis')
    alpha : float, optional
        Transparency of points (0 to 1)
    s : float, optional
        Point size
    view_angle : tuple, optional
        Elevation and azimuth angles for the 3D view
    save_path : str, optional
        Path to save the figure (if None, figure is not saved)
    every_n : int, optional
        Plot every n-th chart to reduce the number of subplots
    zlim : tuple, optional
        Limits for z-axis (min, max)
    show_colorbar : bool, optional
        Whether to show the colorbar
    show_time : bool, optional
        Whether to show time labels in subplot titles
    vmin : float, optional
        Minimum value for color scaling
    vmax : float, optional
        Maximum value for color scaling
    max_time_fraction : float, optional
        Fraction of the time range to plot (default: 1.0 = full range)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    axes : array of mpl_toolkits.mplot3d.Axes3D
        The 3D axes objects
    """
    # Set the style for publication quality
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.usetex'] = False
    
    # Limit to the specified time fraction
    max_time_idx = int(len(ts) * max_time_fraction)
    charts = charts[:max_time_idx]
    ts = ts[:max_time_idx]
    if values is not None:
        values = values[:max_time_idx]
    
    # Force exactly 5 plots per row
    cols = 5
    
    # Calculate how many plots to show (must be a multiple of 5)
    num_plots = (len(charts) // cols) * cols
    if num_plots > 10:  # Limit to 10 plots (2 rows)
        num_plots = 10
    
    rows = num_plots // cols
    
    # Calculate indices to select evenly spaced plots
    if len(charts) > num_plots:
        indices = np.linspace(0, len(charts) - 1, num_plots, dtype=int)
        selected_charts = [charts[i] for i in indices]
        selected_ts = ts[indices]
        if values is not None:
            selected_values = [values[i] for i in indices]
    else:
        # If we have fewer charts than needed, use all of them
        selected_charts = charts[:num_plots]
        selected_ts = ts[:num_plots]
        if values is not None:
            selected_values = values[:num_plots]
    
    # Create figure without using tight_layout
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Find global min/max for consistent color scaling and z-limits
    if zlim is None:
        all_z = np.concatenate([chart[:, 2] for chart in selected_charts])
        zlim = (np.min(all_z), np.max(all_z))
    
    if values is not None and vmin is None and vmax is None:
        all_values = np.concatenate(selected_values)
        vmin, vmax = np.min(all_values), np.max(all_values)
    
    # Calculate custom grid positions for maximum plot size
    # Define even smaller margins
    left_margin = 0.01
    right_margin = 0.01 if not show_colorbar else 0.10
    bottom_margin = 0.01
    top_margin = 0.01
    
    # Calculate available space
    available_width = 1.0 - left_margin - right_margin
    available_height = 1.0 - bottom_margin - top_margin
    
    # Calculate plot size and spacing - make plots overlap slightly
    plot_width = available_width / cols * 1.02  # Slightly larger than allocated space
    plot_height = available_height / rows * 1.02
    
    # Create subplots
    axes = []
    for i in range(num_plots):
        row = i // cols
        col = i % cols
        
        # Calculate position for this subplot - allow slight overlap
        left = left_margin + col * (available_width / cols) - 0.005
        bottom = 1.0 - top_margin - (row + 1) * (available_height / rows) - 0.005
        width = plot_width + 0.01  # Slightly larger to create overlap
        height = plot_height + 0.01
        
        # Create custom positioned subplot
        ax = fig.add_axes([left, bottom, width, height], projection="3d")
        axes.append(ax)
        
        chart = selected_charts[i]
        t = selected_ts[i]
        
        # Extract coordinates
        x, y, z = chart[:, 0], chart[:, 1], chart[:, 2]
        
        # Color by values if provided, otherwise by z-coordinate
        if values is not None:
            scatter = ax.scatter(
                x, y, z,
                c=selected_values[i],
                cmap=cmap,
                s=s,
                alpha=alpha,
                edgecolors='none',
                vmin=vmin,
                vmax=vmax
            )
        else:
            scatter = ax.scatter(
                x, y, z,
                c=z,
                cmap=cmap,
                s=s,
                alpha=alpha,
                edgecolors='none',
                vmin=zlim[0] if vmin is None else vmin,
                vmax=zlim[1] if vmax is None else vmax
            )
        
        # Set view angle
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Set title with time - larger font size and minimal padding
        if show_time:
            ax.set_title(f"t = {t:.2f}", fontsize=18, pad=2)
        
        # Set consistent z-limits
        ax.set_zlim(zlim)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Make the panes transparent
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Make the panes (box faces) completely invisible
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        
        # Turn off the grid
        ax.grid(False)
        
        # Hide all axes
        ax.set_axis_off()
    
    # Add a single colorbar for all subplots if requested
    if show_colorbar:
        cbar_ax = fig.add_axes([1.0 - right_margin + 0.01, bottom_margin, 0.02, available_height])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=12)
        if values is not None:
            cbar.set_label('Value', fontsize=14, rotation=270, labelpad=20)
        else:
            cbar.set_label('Z Coordinate', fontsize=14, rotation=270, labelpad=20)
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        # Also save as PDF
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
    
    return fig, axes


def plot_charts_individually(charts, ts, values=None, figsize=(8, 8), dpi=300, 
                         cmap='viridis', alpha=0.8, s=5, view_angle=(30, 45),
                         save_dir=None, zlim=None, vmin=None, vmax=None,
                         show_time=True, file_prefix="chart_t_", every_n=1):
    """
    Plot each 3D chart individually and save to separate files.
    
    Parameters:
    -----------
    charts : list of numpy.ndarray
        List of point clouds at different time steps
    ts : numpy.ndarray
        Time values corresponding to each chart
    values : list of numpy.ndarray, optional
        Values to color the points by at each time step (if None, uses z-coordinate)
    figsize : tuple, optional
        Figure size in inches (width, height)
    dpi : int, optional
        Resolution in dots per inch
    cmap : str, optional
        Colormap for the points (default: 'viridis')
    alpha : float, optional
        Transparency of points (0 to 1)
    s : float, optional
        Point size
    view_angle : tuple, optional
        Elevation and azimuth angles for the 3D view
    save_dir : str, optional
        Directory to save the figures (if None, figures are not saved)
    zlim : tuple, optional
        Limits for z-axis (min, max)
    vmin : float, optional
        Minimum value for color scaling
    vmax : float, optional
        Maximum value for color scaling
    show_time : bool, optional
        Whether to show time labels in subplot titles
    file_prefix : str, optional
        Prefix for saved files
    every_n : int, optional
        Save only one chart every N steps (default: 1 = save all charts)
    
    Returns:
    --------
    None
    """
    # Set the style for publication quality
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.usetex'] = False
    
    # Create save directory if it doesn't exist
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Select charts at regular intervals
    selected_indices = range(0, len(charts), every_n)
    selected_charts = [charts[i] for i in selected_indices]
    selected_ts = ts[selected_indices]
    if values is not None:
        selected_values = [values[i] for i in selected_indices]
    
    # Find global min/max for consistent color scaling and z-limits
    if zlim is None:
        all_z = np.concatenate([chart[:, 2] for chart in selected_charts])
        zlim = (np.min(all_z), np.max(all_z))
    
    if values is not None and vmin is None and vmax is None:
        all_values = np.concatenate(selected_values)
        vmin, vmax = np.min(all_values), np.max(all_values)
    
    # Process each selected chart individually
    for i, (chart, t) in enumerate(zip(selected_charts, selected_ts)):
        # Create a new figure for each chart
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        
        # Extract coordinates
        x, y, z = chart[:, 0], chart[:, 1], chart[:, 2]
        
        # Color by values if provided, otherwise by z-coordinate
        if values is not None:
            scatter = ax.scatter(
                x, y, z,
                c=selected_values[i],
                cmap=cmap,
                s=s,
                alpha=alpha,
                edgecolors='none',
                vmin=vmin,
                vmax=vmax
            )
        else:
            scatter = ax.scatter(
                x, y, z,
                c=z,
                cmap=cmap,
                s=s,
                alpha=alpha,
                edgecolors='none',
                vmin=zlim[0] if vmin is None else vmin,
                vmax=zlim[1] if vmax is None else vmax
            )
        
        # Set view angle
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Set title with time
        if show_time:
            ax.set_title(f"t = {t:.2f}", fontsize=18)
        
        # Set consistent z-limits
        ax.set_zlim(zlim)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Make the panes transparent
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Make the panes (box faces) completely invisible
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        
        # Turn off the grid
        ax.grid(False)
        
        # Hide all axes
        ax.set_axis_off()
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=12)
        if values is not None:
            cbar.set_label('Value', fontsize=14, rotation=270, labelpad=20)
        else:
            cbar.set_label('Z Coordinate', fontsize=14, rotation=270, labelpad=20)
        
        # Tight layout
        plt.tight_layout()
        
        # Save the figure if a directory is provided
        if save_dir:
            # Format the time value for the filename
            time_str = f"{t:.3f}".replace('.', '_')
            filename = f"{file_prefix}{time_str}.png"
            filepath = Path(save_dir) / filename
            plt.savefig(filepath, bbox_inches='tight', dpi=dpi)
            
            # Also save as PDF
            pdf_path = str(filepath).rsplit('.', 1)[0] + '.pdf'
            plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
        
        # Close the figure to free memory
        plt.close(fig)
