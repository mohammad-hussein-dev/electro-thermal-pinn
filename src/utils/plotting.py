"""
Plotting utilities for visualizing PINN results.

This module provides professional-grade plotting functions for visualizing
predicted fields (E, H, T), spatially-varying parameters (alpha, sigma),
training history, and comparison summaries. All plots are styled for
publication-quality presentations and reports.

Functions:
    - plot_electro_thermal_results: Plot predicted E, H, T fields.
    - plot_varying_parameters: Plot predicted alpha(x) and sigma(x).
    - plot_training_history: Plot loss curves and learning rate.
    - plot_comparison_summary: Comprehensive comparison with error plots.
    - set_plot_style: Configure global matplotlib style.
    - save_figure: Save figure with high quality.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any

# =============================================================================
#  GLOBAL STYLE CONFIGURATION
# =============================================================================

def set_plot_style() -> None:
    """
    Configure global plotting style for professional-looking figures.

    This function sets matplotlib parameters for:
        - Font sizes (axes, titles, legends)
        - Figure DPI and face color
        - Grid appearance
        - Save settings (high DPI, tight bounding box)

    The style is optimized for presentations, publications, and reports.
    """
    # Use seaborn style if available, fallback to ggplot
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        plt.style.use('ggplot')

    # Font settings
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # Figure settings
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['figure.facecolor'] = 'white'

    # Axes settings
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    # Legend settings
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.edgecolor'] = 'none'

    # Save settings
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.facecolor'] = 'white'


def setup_figure(
    n_rows: int = 1,
    n_cols: int = 1,
    figsize: Optional[Tuple[float, float]] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a figure with professional styling.

    Args:
        n_rows (int): Number of rows in the subplot grid. Defaults to 1.
        n_cols (int): Number of columns in the subplot grid. Defaults to 1.
        figsize (Optional[Tuple[float, float]]): Figure size in inches.
            If None, auto-calculated based on n_rows and n_cols.

    Returns:
        Tuple[plt.Figure, np.ndarray]: Figure and axes objects.
            Axes is always a 2D array for consistent indexing.

    Example:
        >>> fig, axes = setup_figure(2, 3, figsize=(18, 10))
        >>> axes[0, 0].plot(x, y)
    """
    set_plot_style()

    if figsize is None:
        figsize = (12 * n_cols, 6 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Ensure axes is always a 2D array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    return fig, axes


def save_figure(fig: plt.Figure, save_path: str, dpi: int = 300) -> None:
    """
    Save a figure with high quality, creating parent directories if needed.

    Args:
        fig (plt.Figure): Figure object to save.
        save_path (str): Path to save the figure.
        dpi (int, optional): Resolution in dots per inch. Defaults to 300.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {save_path}")
    plt.close(fig)


def _compute_field_metrics(
    pred: np.ndarray,
    exact: Optional[np.ndarray]
) -> Dict[str, float]:
    """
    Compute error metrics for a single field.

    Args:
        pred (np.ndarray): Predicted values.
        exact (Optional[np.ndarray]): Exact values.

    Returns:
        Dict[str, float]: Dictionary containing L2, MAE, Max, and MSE.
    """
    metrics = {}

    if exact is not None:
        # Relative L2 error
        l2 = np.sqrt(np.mean((pred - exact) ** 2))
        l2_norm = l2 / (np.sqrt(np.mean(exact ** 2)) + 1e-12)
        metrics['L2_relative'] = l2_norm
        metrics['L2_absolute'] = l2

        # MAE
        metrics['MAE'] = np.mean(np.abs(pred - exact))

        # Max error
        metrics['Max'] = np.max(np.abs(pred - exact))

        # MSE
        metrics['MSE'] = np.mean((pred - exact) ** 2)

    return metrics


def _add_metrics_box(
    ax: plt.Axes,
    metrics: Dict[str, float],
    label: str,
    color: str
) -> None:
    """
    Add a metrics text box to an axes.

    Args:
        ax (plt.Axes): Matplotlib axes object.
        metrics (Dict[str, float]): Metrics dictionary.
        label (str): Field label (E, H, T).
        color (str): Color for the text box border.
    """
    if not metrics:
        return

    text = f"{label} Metrics:\n"
    if 'L2_relative' in metrics:
        text += f"L2: {metrics['L2_relative']:.2e}\n"
    if 'MAE' in metrics:
        text += f"MAE: {metrics['MAE']:.2e}\n"
    if 'Max' in metrics:
        text += f"Max: {metrics['Max']:.2e}"

    ax.text(
        0.02, 0.98,
        text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            edgecolor=color,
            alpha=0.85
        ),
        family='monospace'
    )


# =============================================================================
#  MAIN PLOTTING FUNCTIONS
# =============================================================================

def plot_electro_thermal_results(
    x_test: np.ndarray,
    E_pred: np.ndarray,
    H_pred: np.ndarray,
    T_pred: np.ndarray,
    E_exact: Optional[np.ndarray] = None,
    H_exact: Optional[np.ndarray] = None,
    T_exact: Optional[np.ndarray] = None,
    save_dir: str = "experiments/figures",
    show: bool = True,
) -> None:
    """
    Plot the predicted electric field, magnetic field, and temperature.

    This function creates three subplots comparing predicted vs exact solutions
    for E, H, and T fields with professional styling. If exact solutions are
    provided, error bands and metric boxes are displayed.

    Args:
        x_test (np.ndarray): Spatial coordinates for evaluation, shape (N, 1).
        E_pred (np.ndarray): Predicted electric field, shape (N, 1).
        H_pred (np.ndarray): Predicted magnetic field, shape (N, 1).
        T_pred (np.ndarray): Predicted temperature, shape (N, 1).
        E_exact (Optional[np.ndarray]): Exact electric field.
        H_exact (Optional[np.ndarray]): Exact magnetic field.
        T_exact (Optional[np.ndarray]): Exact temperature.
        save_dir (str, optional): Directory to save the figure.
            Defaults to "experiments/figures".
        show (bool, optional): Whether to display the figure. Defaults to True.

    Raises:
        ValueError: If input arrays have incompatible shapes.

    Example:
        >>> plot_electro_thermal_results(
        ...     x_test, E_pred, H_pred, T_pred,
        ...     E_exact, H_exact, T_exact
        ... )
    """
    if len(x_test) != len(E_pred) or len(x_test) != len(H_pred) or len(x_test) != len(T_pred):
        raise ValueError("All input arrays must have the same length.")

    set_plot_style()

    fig, axes = setup_figure(1, 3, figsize=(15, 5))

    # Field configurations
    fields: List[Tuple[str, np.ndarray, Optional[np.ndarray], str, str]] = [
        ('E', E_pred, E_exact, 'Electric Field (E)', '#1f77b4'),
        ('H', H_pred, H_exact, 'Magnetic Field (H)', '#ff7f0e'),
        ('T', T_pred, T_exact, 'Temperature (T)', '#2ca02c'),
    ]

    for idx, (label, pred, exact, title, color) in enumerate(fields):
        ax = axes[0, idx]

        # Plot predicted
        ax.plot(x_test, pred, color=color, linewidth=2.5, label=f'PINN {label}')

        # Plot exact if available
        if exact is not None:
            ax.plot(x_test, exact, 'k--', linewidth=2, alpha=0.7, label=f'Exact {label}')

            # Add error band
            error = np.abs(pred - exact)
            ax.fill_between(
                x_test.flatten(),
                pred.flatten() - error.flatten(),
                pred.flatten() + error.flatten(),
                color=color, alpha=0.15, label='± Error'
            )

            # Add metrics box
            metrics = _compute_field_metrics(pred, exact)
            _add_metrics_box(ax, metrics, label, color)

        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel(title, fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'electro_thermal_results.png')
    save_figure(fig, save_path)

    if show:
        plt.show()


def plot_varying_parameters(
    x_test: np.ndarray,
    alpha_pred: np.ndarray,
    sigma_pred: np.ndarray,
    alpha_exact: Optional[np.ndarray] = None,
    sigma_exact: Optional[np.ndarray] = None,
    save_dir: str = "experiments/figures",
    show: bool = True,
) -> None:
    """
    Plot the predicted spatially-varying parameters alpha(x) and sigma(x).

    This function creates two subplots for thermal diffusivity and electrical
    conductivity, with optional exact solutions for comparison.

    Args:
        x_test (np.ndarray): Spatial coordinates for evaluation, shape (N, 1).
        alpha_pred (np.ndarray): Predicted thermal diffusivity, shape (N, 1).
        sigma_pred (np.ndarray): Predicted electrical conductivity, shape (N, 1).
        alpha_exact (Optional[np.ndarray]): Exact thermal diffusivity.
        sigma_exact (Optional[np.ndarray]): Exact electrical conductivity.
        save_dir (str, optional): Directory to save the figure.
            Defaults to "experiments/figures".
        show (bool, optional): Whether to display the figure. Defaults to True.

    Example:
        >>> plot_varying_parameters(
        ...     x_test, alpha_pred, sigma_pred,
        ...     alpha_exact, sigma_exact
        ... )
    """
    set_plot_style()

    fig, axes = setup_figure(1, 2, figsize=(14, 5))

    # Alpha plot
    ax1 = axes[0, 0]
    ax1.plot(x_test, alpha_pred, 'b-', linewidth=2.5, label='PINN α(x)')
    if alpha_exact is not None:
        ax1.plot(x_test, alpha_exact, 'k--', linewidth=2, alpha=0.7, label='Exact α(x)')
        metrics = _compute_field_metrics(alpha_pred, alpha_exact)
        _add_metrics_box(ax1, metrics, 'α', 'blue')
    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel(r'$\alpha(x)$', fontsize=14)
    ax1.set_title('Thermal Diffusivity', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=12)

    # Sigma plot
    ax2 = axes[0, 1]
    ax2.plot(x_test, sigma_pred, 'r-', linewidth=2.5, label='PINN σ(x)')
    if sigma_exact is not None:
        ax2.plot(x_test, sigma_exact, 'k--', linewidth=2, alpha=0.7, label='Exact σ(x)')
        metrics = _compute_field_metrics(sigma_pred, sigma_exact)
        _add_metrics_box(ax2, metrics, 'σ', 'red')
    ax2.set_xlabel('x', fontsize=14)
    ax2.set_ylabel(r'$\sigma(x)$', fontsize=14)
    ax2.set_title('Electrical Conductivity', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=12)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'varying_parameters.png')
    save_figure(fig, save_path)

    if show:
        plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_dir: str = "experiments/figures",
    show: bool = True,
) -> None:
    """
    Plot training history including loss curves and learning rate.

    Args:
        history (Dict[str, List[float]]): Dictionary containing training metrics.
            Expected keys: 'total_loss', 'loss_pde', 'loss_bc', 'loss_data',
            'uncertainty', 'learning_rate'. All values are lists of floats.
        save_dir (str, optional): Directory to save the figure.
            Defaults to "experiments/figures".
        show (bool, optional): Whether to display the figure. Defaults to True.

    Example:
        >>> history = {
        ...     'total_loss': [1.0, 0.5, 0.1],
        ...     'loss_pde': [0.8, 0.4, 0.08],
        ...     'loss_bc': [0.2, 0.1, 0.02],
        ... }
        >>> plot_training_history(history)
    """
    set_plot_style()

    fig, axes = setup_figure(2, 2, figsize=(14, 10))

    # Total loss
    ax1 = axes[0, 0]
    if 'total_loss' in history and history['total_loss']:
        ax1.semilogy(history['total_loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Total Loss', fontsize=14)
    ax1.set_title('Training Loss', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Individual loss components
    ax2 = axes[0, 1]
    loss_keys = [
        ('loss_pde', 'PDE', 'r'),
        ('loss_bc', 'BC', 'g'),
        ('loss_data', 'Data', 'm'),
    ]
    has_loss_data = False
    for key, label, color in loss_keys:
        if key in history and history[key]:
            ax2.semilogy(history[key], color=color, linewidth=2, label=label)
            has_loss_data = True

    if has_loss_data:
        ax2.legend(loc='best', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=14)
    ax2.set_title('Loss Components', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Uncertainty (if available)
    ax3 = axes[1, 0]
    if 'uncertainty' in history and history['uncertainty']:
        ax3.plot(history['uncertainty'], 'b-', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=14)
        ax3.set_ylabel('Uncertainty', fontsize=14)
        ax3.set_title('Uncertainty Quantification', fontsize=16, fontweight='bold')
        ax3.grid(True, alpha=0.3)

    # Learning rate (if available)
    ax4 = axes[1, 1]
    if 'learning_rate' in history and history['learning_rate']:
        ax4.semilogy(history['learning_rate'], 'r-', linewidth=2)
        ax4.set_xlabel('Epoch', fontsize=14)
        ax4.set_ylabel('Learning Rate', fontsize=14)
        ax4.set_title('Learning Rate Schedule', fontsize=16, fontweight='bold')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'training_history.png')
    save_figure(fig, save_path)

    if show:
        plt.show()


def plot_comparison_summary(
    x_test: np.ndarray,
    E_pred: np.ndarray,
    H_pred: np.ndarray,
    T_pred: np.ndarray,
    E_exact: np.ndarray,
    H_exact: np.ndarray,
    T_exact: np.ndarray,
    save_dir: str = "experiments/figures",
    show: bool = True,
) -> None:
    """
    Create a comprehensive comparison summary with prediction and error plots.

    This function creates a 2x3 grid showing:
        - Top row: Predicted vs exact for E, H, T
        - Bottom row: Absolute error for E, H, T

    Args:
        x_test (np.ndarray): Spatial coordinates, shape (N, 1).
        E_pred, H_pred, T_pred (np.ndarray): Predicted fields.
        E_exact, H_exact, T_exact (np.ndarray): Exact fields.
        save_dir (str, optional): Directory to save the figure.
            Defaults to "experiments/figures".
        show (bool, optional): Whether to display the figure. Defaults to True.

    Example:
        >>> plot_comparison_summary(
        ...     x_test, E_pred, H_pred, T_pred,
        ...     E_exact, H_exact, T_exact
        ... )
    """
    set_plot_style()

    fig, axes = setup_figure(2, 3, figsize=(18, 10))

    fields: List[Tuple[str, np.ndarray, np.ndarray, str, str]] = [
        ('E', E_pred, E_exact, 'Electric Field', '#1f77b4'),
        ('H', H_pred, H_exact, 'Magnetic Field', '#ff7f0e'),
        ('T', T_pred, T_exact, 'Temperature', '#2ca02c'),
    ]

    for idx, (label, pred, exact, title, color) in enumerate(fields):
        # Predicted vs exact (top row)
        ax_pred = axes[0, idx]
        ax_pred.plot(x_test, pred, color=color, linewidth=2.5, label=f'PINN {label}')
        ax_pred.plot(x_test, exact, 'k--', linewidth=2, alpha=0.7, label=f'Exact {label}')
        ax_pred.set_xlabel('x', fontsize=14)
        ax_pred.set_ylabel(title, fontsize=14)
        ax_pred.set_title(f'{title} - Prediction', fontsize=16, fontweight='bold')
        ax_pred.grid(True, alpha=0.3)
        ax_pred.legend(loc='best', fontsize=12)

        # Error (bottom row)
        ax_err = axes[1, idx]
        error = np.abs(pred - exact)
        ax_err.fill_between(x_test.flatten(), 0, error.flatten(), color=color, alpha=0.3)
        ax_err.plot(x_test, error, color=color, linewidth=2)
        ax_err.set_xlabel('x', fontsize=14)
        ax_err.set_ylabel('Absolute Error', fontsize=14)
        ax_err.set_title(f'{title} - Error', fontsize=16, fontweight='bold')
        ax_err.grid(True, alpha=0.3)
        ax_err.set_ylim(bottom=0)

        # Add max error annotation
        max_err = np.max(error)
        max_idx = np.argmax(error)
        ax_err.annotate(
            f'Max: {max_err:.2e}',
            xy=(x_test[max_idx], max_err),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
        )

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'comparison_summary.png')
    save_figure(fig, save_path)

    if show:
        plt.show()
