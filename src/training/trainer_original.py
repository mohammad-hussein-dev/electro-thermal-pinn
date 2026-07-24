"""
Original training utilities for the forward problem with constant parameters.

This module provides the loss function for the forward problem used to train
the reference model for data generation in inverse problems.
"""

import torch
import torch.nn as nn
from typing import Tuple

from physics.maxwell_pde import maxwell_1d_residual
from physics.heat_pde import heat_pde_with_source
from physics.boundary_conditions import dirichlet_bc_0, dirichlet_bc_1


def compute_loss_original(
    model: nn.Module,
    xi_f: torch.Tensor,
    xi_b0: torch.Tensor,
    xi_b1: torch.Tensor,
    alpha: float = 1.0,
    sigma: float = 1.0,
    rho_c: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the loss for the forward problem with constant parameters.

    This loss combines PDE residuals and boundary conditions only.
    It is used to train the reference model for generating synthetic data.

    Args:
        model (nn.Module): The PINN model for E, H, T.
        xi_f (torch.Tensor): Interior collocation points (x, t).
        xi_b0 (torch.Tensor): Left boundary points (x = 0).
        xi_b1 (torch.Tensor): Right boundary points (x = 1).
        alpha (float, optional): Thermal diffusivity. Defaults to 1.0.
        sigma (float, optional): Electrical conductivity. Defaults to 1.0.
        rho_c (float, optional): Volumetric heat capacity. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - total_loss (torch.Tensor): Sum of PDE and BC losses.
            - loss_pde (torch.Tensor): PDE residual loss (for monitoring).
            - loss_bc (torch.Tensor): Boundary condition loss (for monitoring).
    """
    x_f = xi_f[:, 0:1]
    t_f = xi_f[:, 1:2]

    # Predict fields at interior points
    E_f, H_f, T_f = model(x_f, t_f)

    # ---- Physics-informed (PDE) loss ----
    res_m1, res_m2 = maxwell_1d_residual(E_f, H_f, x_f, t_f)
    res_heat = heat_pde_with_source(T_f, E_f, x_f, t_f, alpha, rho_c, sigma)

    loss_pde = (
        torch.mean(res_m1 ** 2) +
        torch.mean(res_m2 ** 2) +
        torch.mean(res_heat ** 2)
    )

    # ---- Boundary condition loss ----
    r0 = dirichlet_bc_0(model, xi_b0)
    r1 = dirichlet_bc_1(model, xi_b1)
    loss_bc = 0.5 * (torch.mean(r0 ** 2) + torch.mean(r1 ** 2))

    # ---- Total loss ----
    total_loss = loss_pde + loss_bc

    return total_loss, loss_pde.detach(), loss_bc.detach()
