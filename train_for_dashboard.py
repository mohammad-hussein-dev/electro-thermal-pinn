"""
Quick training script to generate a model for the dashboard.
Run this if you don't have a trained model yet.
"""

import os
import torch
import numpy as np
from src.models.electro_thermal_pinn import ElectroThermalPINN
from physics.varying_parameters import ParameterNetwork
from training.trainer_original import compute_loss_original
from utils.data_sampling import sample_interior_points, boundary_points
from utils.config_loader import load_config

def train_quick():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("src/configs/config.yaml")
    layers = config["network"]["layers"]

    model = ElectroThermalPINN(layers).to(device)
    param_net = ParameterNetwork(input_dim=1, output_dim=2).to(device)

    xi_f = sample_interior_points(500, device)
    xi_b0, xi_b1 = boundary_points(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training model for dashboard...")
    for it in range(500):
        optimizer.zero_grad()
        loss, _, _ = compute_loss_original(model, xi_f, xi_b0, xi_b1)
        loss.backward()
        optimizer.step()

        if it % 100 == 0:
            print(f"Iter {it}: Loss = {loss.item():.4e}")

    os.makedirs("experiments/saved_models", exist_ok=True)
    torch.save(model.state_dict(), "experiments/saved_models/electro_thermal_pinn.pt")
    torch.save(param_net.state_dict(), "experiments/saved_models/param_net.pt")
    print("✅ Model saved to experiments/saved_models/")

if __name__ == "__main__":
    train_quick()
