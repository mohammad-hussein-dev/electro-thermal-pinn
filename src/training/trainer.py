"""
Training utilities for the Electro-Thermal PINN with uncertainty and varying parameters.

This module provides the loss function that combines PDE residuals,
boundary conditions, and data loss for inverse problems.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from physics.maxwell_pde import maxwell_1d_residual
from physics.heat_pde import heat_pde_with_source
from physics.boundary_conditions import dirichlet_bc_0, dirichlet_bc_1


def compute_loss(
    model: nn.Module,
    param_net: nn.Module,
    xi_f: torch.Tensor,
    xi_b0: torch.Tensor,
    xi_b1: torch.Tensor,
    lambda_pde: float = 1.0,
    lambda_bc: float = 1.0,
    lambda_data: float = 1.0,
    data_tuple: Optional[Tuple] = None,
    use_uncertainty: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the total loss for the Electro-Thermal PINN.

    The total loss is a weighted sum of three components:
        1. PDE residual loss (Maxwell's equations + Heat equation)
        2. Boundary condition loss (Dirichlet BCs)
        3. Data loss (for inverse problems, optional)

    Args:
        model (nn.Module): The PINN model for E, H, T.
        param_net (nn.Module): The network for spatially-varying parameters.
        xi_f (torch.Tensor): Interior collocation points (x, t).
        xi_b0 (torch.Tensor): Left boundary points (x = 0).
        xi_b1 (torch.Tensor): Right boundary points (x = 1).
        lambda_pde (float, optional): Weight for PDE loss. Defaults to 1.0.
        lambda_bc (float, optional): Weight for BC loss. Defaults to 1.0.
        lambda_data (float, optional): Weight for data loss. Defaults to 1.0.
        data_tuple (Optional[Tuple], optional): Tuple of (x_data, t_data, E_target,
            H_target, T_target). Defaults to None.
        use_uncertainty (bool, optional): Whether to compute uncertainty.
            Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple containing:
                - total_loss (torch.Tensor): Weighted sum of all losses.
                - loss_pde (torch.Tensor): PDE residual loss (for monitoring).
                - loss_bc (torch.Tensor): Boundary condition loss (for monitoring).
                - loss_data (torch.Tensor): Data loss (for monitoring).
                - uncertainty (torch.Tensor): Uncertainty metric (if enabled).
    """
    # Extract spatial and temporal coordinates
    x_f = xi_f[:, 0:1]
    t_f = xi_f[:, 1:2]

    # Predict fields at interior points
    E_f, H_f, T_f = model(x_f, t_f)

    # Predict spatially-varying parameters at interior points
    params = param_net(x_f)
    alpha = params[:, 0:1]  # Thermal diffusivity
    sigma = params[:, 1:2]  # Electrical conductivity

    # ---- Physics-informed (PDE) loss ----
    # Residuals of Maxwell's equations
    res_m1, res_m2 = maxwell_1d_residual(E_f, H_f, x_f, t_f)

    # Residual of the heat equation with Joule heating
    res_heat = heat_pde_with_source(
        T_f, E_f, x_f, t_f,
        alpha=alpha.squeeze(),
        sigma=sigma.squeeze()
    )

    # Mean squared error of PDE residuals
    loss_pde = (
        torch.mean(res_m1 ** 2) +
        torch.mean(res_m2 ** 2) +
        torch.mean(res_heat ** 2)
    )

    # ---- Boundary condition loss ----
    # Dirichlet BCs: T(0) = 0, T(1) = 1
    r0 = dirichlet_bc_0(model, xi_b0)
    r1 = dirichlet_bc_1(model, xi_b1)
    loss_bc = 0.5 * (torch.mean(r0 ** 2) + torch.mean(r1 ** 2))

    # ---- Data loss (for inverse problems) ----
    loss_data = torch.tensor(0.0, device=xi_f.device)
    if data_tuple is not None:
        x_data, t_data, E_target, H_target, T_target = data_tuple
        E_pred, H_pred, T_pred = model(x_data, t_data)

        # Mean squared error between predictions and observations
        loss_data = (
            torch.mean((E_pred - E_target) ** 2) +
            torch.mean((H_pred - H_target) ** 2) +
            torch.mean((T_pred - T_target) ** 2)
        )

    # ---- Uncertainty quantification ----
    uncertainty = torch.tensor(0.0, device=xi_f.device)
    if use_uncertainty:
        # Use variance of parameters as a proxy for uncertainty
        uncertainty = torch.var(params, dim=0).mean()

    # ---- Total loss ----
    total_loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_data * loss_data

    return total_loss, loss_pde.detach(), loss_bc.detach(), loss_data.detach(), uncertainty.detach()
