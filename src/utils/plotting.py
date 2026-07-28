"""
Plotting utilities for visualizing PINN results.
Plots are saved to disk only (no interactive display).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any


def set_plot_style() -> None:
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        plt.style.use('ggplot')
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.edgecolor'] = 'none'
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.facecolor'] = 'white'


def setup_figure(n_rows=1, n_cols=1, figsize=None):
    set_plot_style()
    if figsize is None:
        figsize = (12 * n_cols, 6 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    return fig, axes


def save_figure(fig, save_path, dpi=300):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {save_path}")
    plt.close(fig)


def _compute_field_metrics(pred, exact):
    metrics = {}
    if exact is not None:
        l2 = np.sqrt(np.mean((pred - exact) ** 2))
        l2_norm = l2 / (np.sqrt(np.mean(exact ** 2)) + 1e-12)
        metrics['L2_relative'] = l2_norm
        metrics['L2_absolute'] = l2
        metrics['MAE'] = np.mean(np.abs(pred - exact))
        metrics['Max'] = np.max(np.abs(pred - exact))
        metrics['MSE'] = np.mean((pred - exact) ** 2)
    return metrics


def _add_metrics_box(ax, metrics, label, color):
    if not metrics:
        return
    text = f"{label} Metrics:\n"
    if 'L2_relative' in metrics:
        text += f"L2: {metrics['L2_relative']:.2e}\n"
    if 'MAE' in metrics:
        text += f"MAE: {metrics['MAE']:.2e}\n"
    if 'Max' in metrics:
        text += f"Max: {metrics['Max']:.2e}"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.85),
            family='monospace')


def plot_electro_thermal_results(x_test, E_pred, H_pred, T_pred,
                                  E_exact=None, H_exact=None, T_exact=None,
                                  save_dir="experiments/figures", show=True):
    set_plot_style()
    fig, axes = setup_figure(1, 3, figsize=(15, 5))
    fields = [
        ('E', E_pred, E_exact, 'Electric Field (E)', '#1f77b4'),
        ('H', H_pred, H_exact, 'Magnetic Field (H)', '#ff7f0e'),
        ('T', T_pred, T_exact, 'Temperature (T)', '#2ca02c'),
    ]
    for idx, (label, pred, exact, title, color) in enumerate(fields):
        ax = axes[0, idx]
        ax.plot(x_test, pred, color=color, linewidth=2.5, label=f'PINN {label}')
        if exact is not None:
            ax.plot(x_test, exact, 'k--', linewidth=2, alpha=0.7, label=f'Exact {label}')
            error = np.abs(pred - exact)
            ax.fill_between(x_test.flatten(), pred.flatten() - error.flatten(),
                            pred.flatten() + error.flatten(), color=color, alpha=0.15, label='± Error')
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


def plot_varying_parameters(x_test, alpha_pred, sigma_pred,
                            alpha_exact=None, sigma_exact=None,
                            save_dir="experiments/figures", show=True):
    set_plot_style()
    fig, axes = setup_figure(1, 2, figsize=(14, 5))
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


def plot_training_history(history, save_dir="experiments/figures", show=True):
    set_plot_style()
    fig, axes = setup_figure(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    if 'total_loss' in history and history['total_loss']:
        ax1.semilogy(history['total_loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Total Loss', fontsize=14)
    ax1.set_title('Training Loss', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    loss_keys = [('loss_pde', 'PDE', 'r'), ('loss_bc', 'BC', 'g'), ('loss_data', 'Data', 'm')]
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

    ax3 = axes[1, 0]
    if 'uncertainty' in history and history['uncertainty']:
        ax3.plot(history['uncertainty'], 'b-', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=14)
        ax3.set_ylabel('Uncertainty', fontsize=14)
        ax3.set_title('Uncertainty Quantification', fontsize=16, fontweight='bold')
        ax3.grid(True, alpha=0.3)

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


def plot_comparison_summary(x_test, E_pred, H_pred, T_pred,
                            E_exact, H_exact, T_exact,
                            save_dir="experiments/figures", show=True):
    set_plot_style()
    fig, axes = setup_figure(2, 3, figsize=(18, 10))
    fields = [
        ('E', E_pred, E_exact, 'Electric Field', '#1f77b4'),
        ('H', H_pred, H_exact, 'Magnetic Field', '#ff7f0e'),
        ('T', T_pred, T_exact, 'Temperature', '#2ca02c'),
    ]
    for idx, (label, pred, exact, title, color) in enumerate(fields):
        ax_pred = axes[0, idx]
        ax_pred.plot(x_test, pred, color=color, linewidth=2.5, label=f'PINN {label}')
        ax_pred.plot(x_test, exact, 'k--', linewidth=2, alpha=0.7, label=f'Exact {label}')
        ax_pred.set_xlabel('x', fontsize=14)
        ax_pred.set_ylabel(title, fontsize=14)
        ax_pred.set_title(f'{title} - Prediction', fontsize=16, fontweight='bold')
        ax_pred.grid(True, alpha=0.3)
        ax_pred.legend(loc='best', fontsize=12)

        ax_err = axes[1, idx]
        error = np.abs(pred - exact)
        ax_err.fill_between(x_test.flatten(), 0, error.flatten(), color=color, alpha=0.3)
        ax_err.plot(x_test, error, color=color, linewidth=2)
        ax_err.set_xlabel('x', fontsize=14)
        ax_err.set_ylabel('Absolute Error', fontsize=14)
        ax_err.set_title(f'{title} - Error', fontsize=16, fontweight='bold')
        ax_err.grid(True, alpha=0.3)
        ax_err.set_ylim(bottom=0)

        max_err = np.max(error)
        max_idx = np.argmax(error)
        ax_err.annotate(f'Max: {max_err:.2e}', xy=(x_test[max_idx], max_err),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'comparison_summary.png')
    save_figure(fig, save_path)
