"""
Electro-Thermal Physics-Informed Neural Network (PINN) model.

This module defines a fully-connected neural network with three outputs:
electric field (E), magnetic field (H), and temperature (T).
The network is designed for solving coupled electro-thermal problems.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class ElectroThermalPINN(nn.Module):
    """
    A fully-connected neural network for coupled electro-thermal problems.

    The network takes spatial (x) and temporal (t) coordinates as inputs and
    predicts three physical fields: electric field (E), magnetic field (H),
    and temperature (T). The architecture uses Tanh activation functions and
    Xavier initialization for stable training.

    Attributes:
        layers (List[int]): List of layer sizes, e.g., [2, 50, 50, 50, 50, 3].
        activation (nn.Module): Activation function (Tanh).
        linears (nn.ModuleList): List of linear layers.
    """

    def __init__(self, layers: Optional[List[int]] = None) -> None:
        """
        Initialize the ElectroThermalPINN model.

        Args:
            layers (Optional[List[int]]): List of layer sizes.
                Defaults to [2, 50, 50, 50, 50, 3] where:
                - Input layer: 2 (x, t)
                - Hidden layers: 4 layers of 50 neurons each
                - Output layer: 3 (E, H, T)
        """
        super(ElectroThermalPINN, self).__init__()

        if layers is None:
            layers = [2, 50, 50, 50, 50, 3]

        self.layers = layers
        self.activation = nn.Tanh()

        # Build the network layers
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i + 1]))

        # Apply Xavier initialization for better convergence
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Spatial coordinates, shape (N, 1).
            t (torch.Tensor): Temporal coordinates, shape (N, 1).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - E (torch.Tensor): Electric field, shape (N, 1)
                - H (torch.Tensor): Magnetic field, shape (N, 1)
                - T (torch.Tensor): Temperature field, shape (N, 1)

        Note:
            The output layer has no activation function to allow unbounded outputs.
        """
        # Concatenate inputs: (x, t) -> (N, 2)
        inputs = torch.cat([x, t], dim=1)

        # Forward pass through hidden layers with Tanh activation
        u = inputs
        for i in range(len(self.linears) - 1):
            u = self.activation(self.linears[i](u))

        # Output layer (no activation for unbounded outputs)
        output = self.linears[-1](u)

        # Split into three physical fields
        E = output[:, 0:1]  # Electric field
        H = output[:, 1:2]  # Magnetic field
        T = output[:, 2:3]  # Temperature

        return E, H, T
