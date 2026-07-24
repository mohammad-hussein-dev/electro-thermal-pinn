"""
1D heat equation with Joule heating source term.

This module provides the residual of the heat equation coupled with
an electric field through Joule heating.
"""

import torch
import torch.autograd as autograd


def heat_pde_with_source(
    T: torch.Tensor,
    E: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float = 1.0,
    rho_c: float = 1.0,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Compute the residual of the 1D heat equation with Joule heating.

    The heat equation with Joule heating source is:
        ∂T/∂t = α * ∂²T/∂x² + (σ / ρc) * |E|²

    Where:
        - α is the thermal diffusivity
        - σ is the electrical conductivity
        - ρc is the volumetric heat capacity
        - E is the electric field (source term)

    Args:
        T (torch.Tensor): Temperature field, requires gradient.
        E (torch.Tensor): Electric field (for Joule heating calculation).
        x (torch.Tensor): Spatial coordinates, requires gradient.
        t (torch.Tensor): Temporal coordinates, requires gradient.
        alpha (float, optional): Thermal diffusivity. Defaults to 1.0.
        rho_c (float, optional): Volumetric heat capacity. Defaults to 1.0.
        sigma (float, optional): Electrical conductivity. Defaults to 1.0.

    Returns:
        torch.Tensor: Residual of the heat equation:
            residual = ∂T/∂t - α * ∂²T/∂x² - (σ/ρc) * E²
            The residual should be zero for the true physical solution.

    Note:
        All input tensors must have `requires_grad=True` for automatic
        differentiation to work correctly.

    Raises:
        RuntimeError: If automatic differentiation fails.
    """
    # First-order time derivative
    T_t = autograd.grad(
        T, t,
        grad_outputs=torch.ones_like(T),
        create_graph=True,
        retain_graph=True
    )[0]

    # First-order spatial derivative (needed for second derivative)
    T_x = autograd.grad(
        T, x,
        grad_outputs=torch.ones_like(T),
        create_graph=True,
        retain_graph=True
    )[0]

    # Second-order spatial derivative
    T_xx = autograd.grad(
        T_x, x,
        grad_outputs=torch.ones_like(T_x),
        create_graph=True,
        retain_graph=True
    )[0]

    # Joule heating source term: (σ / ρc) * E²
    source = (sigma / rho_c) * (E ** 2)

    # Heat equation residual
    residual = T_t - alpha * T_xx - source

    return residual
