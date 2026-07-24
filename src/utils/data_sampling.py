"""
Data sampling utilities for collocation and boundary points.

This module provides functions to generate random points for PINN training
within the domain [0, 1] x [0, 1] and on the boundaries.
"""

import torch
from typing import Tuple


def sample_interior_points(n_points: int, device: torch.device) -> torch.Tensor:
    """
    Sample random interior points within the domain [0, 1] x [0, 1].

    The domain is defined as:
        x ∈ [0, 1]  (spatial dimension)
        t ∈ [0, 1]  (temporal dimension)

    Args:
        n_points (int): Number of points to sample.
        device (torch.device): Device to place the tensor on.

    Returns:
        torch.Tensor: Sampled points with shape (n_points, 2).
            Column 0: x coordinates
            Column 1: t coordinates

    Note:
        The tensor has `requires_grad=True` for automatic differentiation.
    """
    # Sample uniformly in [0, 1] for both dimensions
    xi = torch.rand(n_points, 2, device=device, requires_grad=True)
    return xi


def boundary_points(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate boundary points for Dirichlet boundary conditions.

    The boundary points are:
        - x = 0 (left boundary): 100 random points in time
        - x = 1 (right boundary): 100 random points in time

    Args:
        device (torch.device): Device to place the tensor on.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - xi_b0 (torch.Tensor): Left boundary points (x=0), shape (100, 1)
            - xi_b1 (torch.Tensor): Right boundary points (x=1), shape (100, 1)

    Note:
        Both tensors have `requires_grad=True` for automatic differentiation.
    """
    n_bc = 100

    # Left boundary: x = 0, t ∈ [0, 1]
    xi_b0 = torch.rand(n_bc, 1, device=device, requires_grad=True)

    # Right boundary: x = 1, t ∈ [0, 1]
    xi_b1 = torch.rand(n_bc, 1, device=device, requires_grad=True)

    return xi_b0, xi_b1
