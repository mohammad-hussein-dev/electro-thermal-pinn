"""
Neural network for spatially-varying physical parameters.

This module defines a ParameterNetwork that predicts parameters like
alpha(x) and sigma(x) as functions of spatial coordinates.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ParameterNetwork(nn.Module):
    """
    Neural network for predicting spatially-varying physical parameters.

    The network takes spatial coordinate x as input and outputs parameters
    such as thermal diffusivity alpha(x) and electrical conductivity sigma(x).

    A Softplus activation is used on the final layer to ensure positive outputs.

    Attributes:
        network (nn.Sequential): The sequential network architecture.
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 2,
        hidden_layers: Optional[List[int]] = None,
    ):
        """
        Initialize the ParameterNetwork.

        Args:
            input_dim (int, optional): Input dimension (1 for x). Defaults to 1.
            output_dim (int, optional): Number of parameters to predict.
                Defaults to 2 (alpha and sigma).
            hidden_layers (Optional[List[int]], optional): List of hidden layer sizes.
                Defaults to [20, 20, 20] if None.
        """
        super(ParameterNetwork, self).__init__()

        if hidden_layers is None:
            hidden_layers = [20, 20, 20]

        layers = []
        prev_dim = input_dim

        # Build hidden layers with Tanh activation
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim

        # Output layer with Softplus for positive outputs
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softplus())

        self.network = nn.Sequential(*layers)

        # Initialize weights for better convergence
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict spatially-varying parameters.

        Args:
            x (torch.Tensor): Spatial coordinates, shape (N, 1).

        Returns:
            torch.Tensor: Predicted parameters, shape (N, output_dim).
                The columns correspond to [alpha(x), sigma(x)].
        """
        return self.network(x)
