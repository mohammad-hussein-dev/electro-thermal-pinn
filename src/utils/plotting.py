"""
Plotting utilities for visualizing PINN results.

This module provides functions to plot the predicted fields (E, H, T)
and the spatially-varying parameters (alpha, sigma).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_electro_thermal_results(
    x_test: np.ndarray,
    E_pred: np.ndarray,
    H_pred: np.ndarray,
    T_pred: np.ndarray,
    save_dir: str = "experiments/figures",
) -> None:
    """
    Plot the predicted electric field, magnetic field, and temperature.

    Args:
        x_test (np.ndarray): Spatial coordinates for evaluation, shape (N, 1).
        E_pred (np.ndarray): Predicted electric field, shape (N, 1).
        H_pred (np.ndarray): Predicted magnetic field, shape (N, 1).
        T_pred (np.ndarray): Predicted temperature, shape (N, 1).
        save_dir (str, optional): Directory to save the figure.
            Defaults to "experiments/figures".

    Note:
        The figure is saved as 'electro_thermal_results.png' in the save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Electric field
    axes[0].plot(x_test, E_pred, 'b-', linewidth=2)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('E (Electric Field)', fontsize=12)
    axes[0].set_title('Predicted Electric Field', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Magnetic field
    axes[1].plot(x_test, H_pred, 'r-', linewidth=2)
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('H (Magnetic Field)', fontsize=12)
    axes[1].set_title('Predicted Magnetic Field', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    # Temperature
    axes[2].plot(x_test, T_pred, 'g-', linewidth=2)
    axes[2].set_xlabel('x', fontsize=12)
    axes[2].set_ylabel('T (Temperature)', fontsize=12)
    axes[2].set_title('Predicted Temperature', fontsize=14)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'electro_thermal_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.show()


def plot_varying_parameters(
    x_test: np.ndarray,
    alpha_pred: np.ndarray,
    sigma_pred: np.ndarray,
    save_dir: str = "experiments/figures",
) -> None:
    """
    Plot the predicted spatially-varying parameters alpha(x) and sigma(x).

    Args:
        x_test (np.ndarray): Spatial coordinates for evaluation, shape (N, 1).
        alpha_pred (np.ndarray): Predicted thermal diffusivity, shape (N, 1).
        sigma_pred (np.ndarray): Predicted electrical conductivity, shape (N, 1).
        save_dir (str, optional): Directory to save the figure.
            Defaults to "experiments/figures".

    Note:
        The figure is saved as 'varying_parameters.png' in the save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Thermal diffusivity
    ax1.plot(x_test, alpha_pred, 'b-', linewidth=2)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel(r'$\alpha(x)$', fontsize=12)
    ax1.set_title('Predicted Thermal Diffusivity', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Electrical conductivity
    ax2.plot(x_test, sigma_pred, 'r-', linewidth=2)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel(r'$\sigma(x)$', fontsize=12)
    ax2.set_title('Predicted Electrical Conductivity', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'varying_parameters.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.show()
