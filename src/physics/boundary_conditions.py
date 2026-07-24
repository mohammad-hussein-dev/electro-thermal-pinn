"""
Dirichlet boundary conditions for the electro-thermal PINN.

This module defines Dirichlet boundary conditions for temperature (T)
at x = 0 and x = 1.
"""

import torch
import torch.nn as nn


def dirichlet_bc_0(model: nn.Module, xi_b0: torch.Tensor) -> torch.Tensor:
    """
    Compute the residual for the left boundary condition at x = 0.

    The temperature at the left boundary is fixed to 0 (dimensionless).

    Args:
        model (nn.Module): The PINN model (takes x and t as inputs).
        xi_b0 (torch.Tensor): Boundary points at x = 0, shape (N, 1).
            These points represent the time coordinate t.

    Returns:
        torch.Tensor: Boundary condition residual: T(x=0) - 0.

    Note:
        At x = 0, the input to the model is (x=0, t).
    """
    # At x=0, the input is only time
    t = xi_b0
    x = torch.zeros_like(t)

    _, _, T = model(x, t)

    # Dirichlet condition: T(0, t) = 0
    return T - 0.0


def dirichlet_bc_1(model: nn.Module, xi_b1: torch.Tensor) -> torch.Tensor:
    """
    Compute the residual for the right boundary condition at x = 1.

    The temperature at the right boundary is fixed to 1 (dimensionless).

    Args:
        model (nn.Module): The PINN model (takes x and t as inputs).
        xi_b1 (torch.Tensor): Boundary points at x = 1, shape (N, 1).
            These points represent the time coordinate t.

    Returns:
        torch.Tensor: Boundary condition residual: T(x=1) - 1.

    Note:
        At x = 1, the input to the model is (x=1, t).
    """
    # At x=1, the input is only time
    t = xi_b1
    x = torch.ones_like(t)

    _, _, T = model(x, t)

    # Dirichlet condition: T(1, t) = 1
    return T - 1.0
