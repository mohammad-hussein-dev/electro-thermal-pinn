"""
Electro-Thermal PINN: Main training script for inverse problems.

This script trains a Physics-Informed Neural Network (PINN) to solve
the coupled electro-thermal problem:
    1. Maxwell's equations (1D) for electric and magnetic fields
    2. Heat equation with Joule heating source term

It also supports inverse problems to identify spatially-varying
physical parameters from limited observational data.

The script includes:
    - Reproducibility via fixed random seeds
    - Early stopping to prevent overfitting
    - Learning rate scheduling for stable convergence
    - L-BFGS refinement for final optimization
    - Comprehensive evaluation metrics (L2 error, MAE, Max error)
    - Visualization of all predicted fields and parameters
"""

import os
import random
import numpy as np
import torch

# ---- Set seeds for reproducibility ----
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from models.electro_thermal_pinn import ElectroThermalPINN
from physics.varying_parameters import ParameterNetwork
from training.trainer import compute_loss
from training.trainer_original import compute_loss_original
from utils.data_sampling import sample_interior_points, boundary_points
from utils.plotting import plot_electro_thermal_results, plot_varying_parameters
from utils.config_loader import load_config


def generate_synthetic_data_with_varying_params(
    model,
    param_net,
    n_points: int = 100,
    device: torch.device = None,
    noise_level: float = 0.02,
) -> tuple:
    """
    Generate synthetic observational data from a reference model.

    This function simulates experimental data by evaluating a pre-trained
    reference model at random points and optionally adding Gaussian noise.

    Args:
        model (nn.Module): The reference PINN model.
        param_net (nn.Module): The reference parameter network (not used directly).
        n_points (int): Number of data points to generate.
        device (torch.device): Device to use.
        noise_level (float): Standard deviation of Gaussian noise.

    Returns:
        tuple: (x_data, t_data, E_target, H_target, T_target)
    """
    x_data = torch.rand(n_points, 1, device=device)
    t_data = torch.rand(n_points, 1, device=device)

    with torch.no_grad():
        E_target, H_target, T_target = model(x_data, t_data)

    # Add Gaussian noise if specified
    if noise_level > 0:
        E_target += noise_level * torch.randn_like(E_target)
        H_target += noise_level * torch.randn_like(H_target)
        T_target += noise_level * torch.randn_like(T_target)

    return x_data, t_data, E_target, H_target, T_target


def train_reference_model(device, layers, N_f=2000, epochs=2000):
    """
    Train a reference model on the forward problem.

    This model is used to generate synthetic data for the inverse problem.
    It solves the forward problem with constant parameters.

    Args:
        device (torch.device): Device to use.
        layers (list): Network architecture.
        N_f (int): Number of interior collocation points.
        epochs (int): Number of training epochs.

    Returns:
        nn.Module: Trained reference model.
    """
    model = ElectroThermalPINN(layers).to(device)
    xi_f = sample_interior_points(N_f, device)
    xi_b0, xi_b1 = boundary_points(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training reference model...")
    for it in range(epochs):
        optimizer.zero_grad()
        total_loss, _, _ = compute_loss_original(model, xi_f, xi_b0, xi_b1)
        total_loss.backward()
        optimizer.step()

        if it % max(1, epochs // 10) == 0:
            print(f"Reference model: Iter {it:6d}, Loss: {total_loss.item():.3e}")

    return model


def evaluate_model(model, param_net, x_test, t_test, device):
    """
    Evaluate the model on test points and return predictions.

    Args:
        model (nn.Module): The PINN model.
        param_net (nn.Module): The parameter network.
        x_test (torch.Tensor): Test spatial coordinates.
        t_test (torch.Tensor): Test temporal coordinates.
        device (torch.device): Device to use.

    Returns:
        tuple: (E_pred, H_pred, T_pred, alpha_pred, sigma_pred)
    """
    model.eval()
    param_net.eval()

    with torch.no_grad():
        E_pred, H_pred, T_pred = model(x_test, t_test)
        E_pred = E_pred.cpu().numpy()
        H_pred = H_pred.cpu().numpy()
        T_pred = T_pred.cpu().numpy()

        params = param_net(x_test)
        alpha_pred = params[:, 0].cpu().numpy()
        sigma_pred = params[:, 1].cpu().numpy()

    return E_pred, H_pred, T_pred, alpha_pred, sigma_pred


def compute_eval_metrics(E_pred, H_pred, T_pred, E_exact, H_exact, T_exact):
    """
    Compute evaluation metrics: L2 error, MAE, and max error.

    Args:
        E_pred (np.ndarray): Predicted electric field.
        H_pred (np.ndarray): Predicted magnetic field.
        T_pred (np.ndarray): Predicted temperature.
        E_exact (np.ndarray): Exact electric field.
        H_exact (np.ndarray): Exact magnetic field.
        T_exact (np.ndarray): Exact temperature.

    Returns:
        dict: Dictionary containing all metrics.
    """
    # Relative L2 error (with small epsilon to avoid division by zero)
    l2_E = np.sqrt(np.mean((E_pred - E_exact) ** 2)) / (np.sqrt(np.mean(E_exact ** 2)) + 1e-8)
    l2_H = np.sqrt(np.mean((H_pred - H_exact) ** 2)) / (np.sqrt(np.mean(H_exact ** 2)) + 1e-8)
    l2_T = np.sqrt(np.mean((T_pred - T_exact) ** 2)) / (np.sqrt(np.mean(T_exact ** 2)) + 1e-8)

    # Mean Absolute Error
    mae_E = np.mean(np.abs(E_pred - E_exact))
    mae_H = np.mean(np.abs(H_pred - H_exact))
    mae_T = np.mean(np.abs(T_pred - T_exact))

    # Maximum Absolute Error
    max_E = np.max(np.abs(E_pred - E_exact))
    max_H = np.max(np.abs(H_pred - H_exact))
    max_T = np.max(np.abs(T_pred - T_exact))

    return {
        'L2_E': l2_E, 'L2_H': l2_H, 'L2_T': l2_T,
        'MAE_E': mae_E, 'MAE_H': mae_H, 'MAE_T': mae_T,
        'Max_E': max_E, 'Max_H': max_H, 'Max_T': max_T
    }


def main() -> None:
    """
    Main training routine.
    """
    # ---- Load configuration ----
    this_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(this_dir, "configs", "config.yaml")
    config = load_config(config_path)

    # Problem parameters
    N_f = config["problem"]["N_f"]
    layers = config["network"]["layers"]

    # Training parameters
    adam_lr = float(config["training"]["adam_lr"])
    adam_iters = config["training"]["adam_iters"]
    use_lbfgs = config["training"]["use_lbfgs"]

    # Inverse problem settings
    inverse_config = config.get("inverse", {})
    enable_inverse = inverse_config.get("enable", False)
    lambda_data = inverse_config.get("lambda_data", 10.0)
    n_data_points = inverse_config.get("n_data_points", 300)
    noise_level = inverse_config.get("noise_level", 0.02)
    use_uncertainty = inverse_config.get("use_uncertainty", True)
    varying_params = inverse_config.get("varying_params", True)

    # ---- Device setup ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Sample collocation and boundary points ----
    xi_f = sample_interior_points(N_f, device)
    xi_b0, xi_b1 = boundary_points(device)

    # ---- Build models ----
    model = ElectroThermalPINN(layers).to(device)
    param_net = ParameterNetwork(input_dim=1, output_dim=2).to(device)

    print("\n=== Inverse Problem with Varying Parameters ===")
    print(f"Varying parameters: {varying_params}")
    print(f"Use uncertainty: {use_uncertainty}")
    print(f"Data weight (lambda_data): {lambda_data}")

    # ---- Generate synthetic data for inverse problem ----
    data_tuple = None
    if enable_inverse:
        print("\n" + "="*50)
        print("Training reference model for data generation...")
        print("="*50)
        reference_model = train_reference_model(device, layers, N_f=2000, epochs=2000)
        reference_param_net = ParameterNetwork(input_dim=1, output_dim=2).to(device)

        print("\nGenerating synthetic data...")
        data_tuple = generate_synthetic_data_with_varying_params(
            reference_model,
            reference_param_net,
            n_points=n_data_points,
            device=device,
            noise_level=noise_level,
        )
        print(f"Generated {n_data_points} data points with noise level {noise_level}")

    # ---- Optimizer with learning rate scheduler ----
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(param_net.parameters()),
        lr=adam_lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=150
    )

    # ---- Training loop with early stopping ----
    print("\n=== Adam Training ===")
    print("="*50)
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 200

    for it in range(adam_iters):
        optimizer.zero_grad()
        total_loss, loss_pde, loss_bc, loss_data, uncertainty = compute_loss(
            model,
            param_net,
            xi_f,
            xi_b0,
            xi_b1,
            lambda_pde=1.0,
            lambda_bc=1.0,
            lambda_data=lambda_data if enable_inverse else 0.0,
            data_tuple=data_tuple,
            use_uncertainty=use_uncertainty,
        )
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        # Early stopping: save best model
        if total_loss < best_loss:
            best_loss = total_loss
            patience_counter = 0
            best_model_path = os.path.join(os.path.dirname(this_dir), "experiments", "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        # Print progress
        if it % max(1, adam_iters // 10) == 0:
            print(
                f"[Adam] Iter {it:6d} | "
                f"Total = {total_loss.item():.3e} | "
                f"PDE = {loss_pde.item():.3e} | "
                f"BC = {loss_bc.item():.3e} | "
                f"Data = {loss_data.item():.3e} | "
                f"UQ = {uncertainty.item():.3e} | "
                f"LR = {optimizer.param_groups[0]['lr']:.2e}"
            )

        # Early stopping condition
        if patience_counter > patience_limit:
            print(f"Early stopping at iteration {it} (patience {patience_limit})")
            break

    # ---- L-BFGS refinement ----
    if use_lbfgs:
        print("\n=== L-BFGS Refinement ===")
        print("="*50)

        # Load the best model from early stopping
        best_model_path = os.path.join(os.path.dirname(this_dir), "experiments", "best_model.pt")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
        else:
            print("Best model not found, using current model.")

        def closure():
            optimizer_lbfgs.zero_grad()
            total_loss, _, _, _, _ = compute_loss(
                model,
                param_net,
                xi_f,
                xi_b0,
                xi_b1,
                lambda_pde=1.0,
                lambda_bc=1.0,
                lambda_data=lambda_data if enable_inverse else 0.0,
                data_tuple=data_tuple,
                use_uncertainty=use_uncertainty,
            )
            total_loss.backward()
            return total_loss

        optimizer_lbfgs = torch.optim.LBFGS(
            list(model.parameters()) + list(param_net.parameters()),
            lr=1.0,
            max_iter=300,
            max_eval=300,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        optimizer_lbfgs.step(closure)
        final_loss, _, _, _, _ = compute_loss(
            model,
            param_net,
            xi_f,
            xi_b0,
            xi_b1,
            lambda_pde=1.0,
            lambda_bc=1.0,
            lambda_data=lambda_data if enable_inverse else 0.0,
            data_tuple=data_tuple,
            use_uncertainty=use_uncertainty,
        )
        print(f"Final loss after L-BFGS: {final_loss.item():.3e}")

    # ---- Evaluation on test set ----
    print("\n=== Evaluation on Test Set ===")
    print("="*50)

    x_test_np = np.linspace(0.0, 1.0, 300).reshape(-1, 1)
    t_test_np = np.ones_like(x_test_np) * 0.5

    x_test = torch.tensor(x_test_np, dtype=torch.float32, device=device)
    t_test = torch.tensor(t_test_np, dtype=torch.float32, device=device)

    E_pred, H_pred, T_pred, alpha_pred, sigma_pred = evaluate_model(
        model, param_net, x_test, t_test, device
    )

    # Generate exact solution from reference model
    if enable_inverse and data_tuple is not None:
        with torch.no_grad():
            E_exact, H_exact, T_exact = reference_model(x_test, t_test)
            E_exact = E_exact.cpu().numpy()
            H_exact = H_exact.cpu().numpy()
            T_exact = T_exact.cpu().numpy()
    else:
        # Fallback analytical solution
        E_exact = x_test_np
        H_exact = 0.5 * x_test_np
        T_exact = 0.5 * x_test_np ** 2

    # Compute metrics
    metrics = compute_eval_metrics(E_pred, H_pred, T_pred, E_exact, H_exact, T_exact)

    print("\n📊 Evaluation Metrics:")
    print(f"  Relative L2 Error:")
    print(f"    E: {metrics['L2_E']:.4e}")
    print(f"    H: {metrics['L2_H']:.4e}")
    print(f"    T: {metrics['L2_T']:.4e}")
    print(f"\n  Mean Absolute Error (MAE):")
    print(f"    E: {metrics['MAE_E']:.4e}")
    print(f"    H: {metrics['MAE_H']:.4e}")
    print(f"    T: {metrics['MAE_T']:.4e}")
    print(f"\n  Max Absolute Error:")
    print(f"    E: {metrics['Max_E']:.4e}")
    print(f"    H: {metrics['Max_H']:.4e}")
    print(f"    T: {metrics['Max_T']:.4e}")

    # ---- Plot results ----
    print("\n=== Plotting Results ===")
    plot_electro_thermal_results(x_test_np, E_pred, H_pred, T_pred)
    plot_varying_parameters(x_test_np, alpha_pred, sigma_pred)

    # ---- Save results ----
    exp_dir = os.path.join(os.path.dirname(this_dir), "experiments")
    os.makedirs(exp_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(exp_dir, "model.pt"))
    torch.save(param_net.state_dict(), os.path.join(exp_dir, "param_net.pt"))

    np.save(os.path.join(exp_dir, "E_pred.npy"), E_pred)
    np.save(os.path.join(exp_dir, "H_pred.npy"), H_pred)
    np.save(os.path.join(exp_dir, "T_pred.npy"), T_pred)
    np.save(os.path.join(exp_dir, "alpha_pred.npy"), alpha_pred)
    np.save(os.path.join(exp_dir, "sigma_pred.npy"), sigma_pred)
    np.save(os.path.join(exp_dir, "E_exact.npy"), E_exact)
    np.save(os.path.join(exp_dir, "H_exact.npy"), H_exact)
    np.save(os.path.join(exp_dir, "T_exact.npy"), T_exact)

    # Save metrics
    with open(os.path.join(exp_dir, "eval_metrics.txt"), "w") as f:
        f.write("Evaluation Metrics:\n")
        f.write("="*50 + "\n")
        f.write(f"Relative L2 Error:\n")
        f.write(f"  E: {metrics['L2_E']:.6e}\n")
        f.write(f"  H: {metrics['L2_H']:.6e}\n")
        f.write(f"  T: {metrics['L2_T']:.6e}\n")
        f.write(f"\nMean Absolute Error (MAE):\n")
        f.write(f"  E: {metrics['MAE_E']:.6e}\n")
        f.write(f"  H: {metrics['MAE_H']:.6e}\n")
        f.write(f"  T: {metrics['MAE_T']:.6e}\n")
        f.write(f"\nMax Absolute Error:\n")
        f.write(f"  E: {metrics['Max_E']:.6e}\n")
        f.write(f"  H: {metrics['Max_H']:.6e}\n")
        f.write(f"  T: {metrics['Max_T']:.6e}\n")

    print(f"\nResults saved to: {exp_dir}")
    print("="*50)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*50)


if __name__ == "__main__":
    main()
