"""
Maxwell's equations in 1D for the electro-thermal PINN.

This module provides functions to compute the residual of Maxwell's equations
and the Joule heating source term for coupling with the heat equation.
"""

import torch
import torch.autograd as autograd
from typing import Tuple


def maxwell_1d_residual(
    E: torch.Tensor,
    H: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    mu: float = 1.0,
    epsilon: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the residual of the 1D Maxwell's equations (source-free form).

    The 1D Maxwell's equations are:
        ∂E/∂t + (1/ε) * ∂H/∂x = 0
        ∂H/∂t + (1/μ) * ∂E/∂x = 0

    Args:
        E (torch.Tensor): Electric field, requires gradient.
        H (torch.Tensor): Magnetic field, requires gradient.
        x (torch.Tensor): Spatial coordinates, requires gradient.
        t (torch.Tensor): Temporal coordinates, requires gradient.
        mu (float, optional): Magnetic permeability. Defaults to 1.0.
        epsilon (float, optional): Electric permittivity. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - residual_1 (torch.Tensor): ∂E/∂t + (1/ε) * ∂H/∂x
            - residual_2 (torch.Tensor): ∂H/∂t + (1/μ) * ∂E/∂x

    Note:
        All input tensors must have `requires_grad=True` for automatic
        differentiation to compute the residuals correctly.

    Raises:
        RuntimeError: If automatic differentiation fails.
    """
    # Compute time derivatives using automatic differentiation
    E_t = autograd.grad(
        E, t,
        grad_outputs=torch.ones_like(E),
        create_graph=True,
        retain_graph=True
    )[0]

    H_t = autograd.grad(
        H, t,
        grad_outputs=torch.ones_like(H),
        create_graph=True,
        retain_graph=True
    )[0]

    # Compute spatial derivatives
    E_x = autograd.grad(
        E, x,
        grad_outputs=torch.ones_like(E),
        create_graph=True,
        retain_graph=True
    )[0]

    H_x = autograd.grad(
        H, x,
        grad_outputs=torch.ones_like(H),
        create_graph=True,
        retain_graph=True
    )[0]

    # Maxwell's equations residuals
    residual_1 = E_t + (1.0 / epsilon) * H_x
    residual_2 = H_t + (1.0 / mu) * E_x

    return residual_1, residual_2


def joule_heating_source(E: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute the Joule heating source term.

    Joule heating (Ohmic heating) is given by: Q = σ * |E|²

    Args:
        E (torch.Tensor): Electric field.
        sigma (float, optional): Electrical conductivity. Defaults to 1.0.

    Returns:
        torch.Tensor: Joule heating source term (scalar field).
    """
    return sigma * (E ** 2)
