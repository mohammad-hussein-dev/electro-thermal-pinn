# Electro-Thermal PINN: Physics-Informed Neural Networks for Coupled Electromagnetic-Thermal Problems

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **Physics-Informed Neural Network (PINN)** framework for solving coupled electromagnetic-thermal problems. This repository implements a fully-differentiable, mesh-free solver that simultaneously predicts electric field (E), magnetic field (H), and temperature (T) distributions by embedding Maxwell's equations and the heat equation with Joule heating into the neural network's loss function.

## 📌 Overview

Traditional numerical solvers (FEM, FVM) require mesh generation and iterative solutions for each set of parameters. This PINN-based approach offers:

- **Mesh-free** modeling with automatic differentiation
- **Physics-constrained** learning without labeled data
- **Forward and inverse** problem capabilities
- **Spatially-varying parameter** identification from limited observations

### Key Features

| Feature | Description |
|---------|-------------|
| **Forward Problem** | Solve coupled PDEs directly from physics constraints |
| **Inverse Problem** | Identify unknown physical parameters from observational data |
| **Spatially-Varying Parameters** | Predict `α(x)` and `σ(x)` as functions of space |
| **Uncertainty Quantification** | Variance-based uncertainty estimation |
| **Reproducibility** | Fixed random seeds for consistent results |
| **Early Stopping** | Prevents overfitting with patience-based stopping |
| **Learning Rate Scheduling** | Adaptive LR for stable convergence |
| **L-BFGS Refinement** | Final optimization for higher accuracy |
| **Comprehensive Metrics** | L2 error, MAE, Max error |

## 🔬 Governing Equations

### Maxwell's Equations (1D, source-free)

```
∂E/∂t + (1/ε) · ∂H/∂x = 0
∂H/∂t + (1/μ) · ∂E/∂x = 0
```

### Heat Equation with Joule Heating

```
∂T/∂t = α · ∂²T/∂x² + (σ/ρc) · E²
```

### Joule Heating Source

```
Q = σ · E²
```

**Where:**
- `E` = Electric field
- `H` = Magnetic field
- `T` = Temperature
- `α` = Thermal diffusivity
- `σ` = Electrical conductivity
- `ρc` = Volumetric heat capacity
- `μ` = Magnetic permeability
- `ε` = Electric permittivity

## 🧠 Methodology

### Physics-Informed Neural Networks

PINNs embed the governing partial differential equations (PDEs) directly into the neural network's loss function. The network learns to approximate the solution `(E, H, T)(x, t)` by minimizing:

```
L_total = λ_pde · L_pde + λ_bc · L_bc + λ_data · L_data
```

Where:
- `L_pde` = Residual of Maxwell's equations + Heat equation
- `L_bc` = Dirichlet boundary condition residuals
- `L_data` = Data mismatch (for inverse problems)

All derivatives are computed via **automatic differentiation** (PyTorch's `autograd`), eliminating the need for mesh generation or numerical differentiation.

### Network Architecture

| Component | Architecture | Details |
|-----------|--------------|---------|
| **Main PINN** | Fully-Connected | Input: (x, t) → Output: (E, H, T) |
| **Parameter Network** | Fully-Connected | Input: (x) → Output: (α(x), σ(x)) |
| **Activation** | Tanh | Hidden layers |
| **Output Activation** | Linear (main) / Softplus (params) | Unbounded / Positive outputs |
| **Initialization** | Xavier | Stable training |

### Training Strategy

1. **Adam Optimizer** (initial training)
2. **ReduceLROnPlateau** (adaptive learning rate)
3. **Early Stopping** (prevents overfitting)
4. **L-BFGS** (final refinement for higher accuracy)

## 📊 Results

### Evaluation Metrics

| Metric | E (Electric) | H (Magnetic) | T (Temperature) |
|--------|--------------|--------------|-----------------|
| **Relative L2 Error** | 9.23e-04 | 6.39e-03 | 1.53e-03 |
| **MAE** | 2.47e-04 | 3.15e-04 | 7.70e-04 |
| **Max Error** | 5.61e-04 | 7.45e-04 | 2.03e-03 |

### Visualizations

The training script automatically generates:

1. **Electro-Thermal Results**: Plots of E, H, T fields
2. **Varying Parameters**: Plots of α(x) and σ(x)

All figures are saved in `experiments/figures/`.

## 📂 Project Structure

```
electro-thermal-pinn/
├── src/
│   ├── models/
│   │   └── electro_thermal_pinn.py      # Main PINN architecture
│   ├── physics/
│   │   ├── maxwell_pde.py               # Maxwell's equations residuals
│   │   ├── heat_pde.py                  # Heat equation with Joule heating
│   │   ├── boundary_conditions.py       # Dirichlet BCs
│   │   └── varying_parameters.py        # Parameter network for α(x), σ(x)
│   ├── training/
│   │   ├── trainer.py                   # Loss function with UQ
│   │   └── trainer_original.py          # Forward problem trainer
│   ├── utils/
│   │   ├── data_sampling.py             # Collocation & boundary points
│   │   ├── plotting.py                  # Visualization utilities
│   │   └── config_loader.py             # YAML configuration loader
│   ├── configs/
│   │   └── config.yaml                  # Hyperparameter configuration
│   └── main.py                          # Main training script
├── experiments/
│   ├── figures/                         # Generated plots
│   ├── results/                         # .npy prediction files
│   └── saved_models/                    # Trained model weights
├── README.md
├── LICENSE
└── requirements.txt
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/electro-thermal-pinn.git
cd electro-thermal-pinn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
pyyaml>=6.0
```

## 🚀 Usage

### Run Training

```bash
python src/main.py
```

### Configuration

All hyperparameters are managed via `src/configs/config.yaml`:

```yaml
problem:
  N_f: 2000              # Number of collocation points

network:
  layers: [2, 50, 50, 50, 50, 3]  # Architecture

training:
  adam_lr: 1e-3
  adam_iters: 3000
  use_lbfgs: true

inverse:
  enable: true
  lambda_data: 100.0     # Weight for data loss
  n_data_points: 500
  noise_level: 0.005
  use_uncertainty: true
  varying_params: true
```

### Customization

- **Forward Problem Only**: Set `enable: false` under `inverse`
- **Adjust Accuracy**: Increase `N_f` and `adam_iters` for higher precision
- **Different Physics**: Modify PDE residuals in `src/physics/`

## 🔬 Inverse Problem Capabilities

This framework supports **inverse problems** to identify unknown physical parameters from limited observational data:

- **Constant Parameters**: Identify global `α` and `σ` values
- **Spatially-Varying Parameters**: Reconstruct `α(x)` and `σ(x)` fields
- **Uncertainty Quantification**: Variance-based confidence estimation

This capability is critical for applications where direct measurement of material properties is expensive or impossible.

## 📈 Example Output

```
📊 Evaluation Metrics:
  Relative L2 Error:
    E: 9.2266e-04
    H: 6.3861e-03
    T: 1.5291e-03

  Mean Absolute Error (MAE):
    E: 2.4665e-04
    H: 3.1498e-04
    T: 7.7009e-04
```

## 🎯 Applications

This framework is suitable for:

- **Electronic Packaging**: Thermal management of ICs and power electronics
- **Battery Design**: Electro-thermal modeling of Li-ion cells
- **Induction Heating**: Coupled field simulation
- **Electroslag Remelting**: Electromagnetic field prediction
- **MHD Flows**: Joule heating effects in magnetohydrodynamics

## 📚 References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics.

2. Karniadakis, G. E., et al. (2021). *Physics-informed machine learning.* Nature Reviews Physics.

3. *Physics-Informed Neural Networks for Multiphysics Simulations: Application to Coupled Electromagnetic-Thermal Modeling.* IEEE Access, 2023.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for:
- New physics modules
- Performance improvements
- Additional test cases
- Documentation enhancements

## 📧 Contact

For questions, collaborations, or feedback:
- **GitHub Issues**: [Open an issue](https://github.com/YOUR_USERNAME/electro-thermal-pinn/issues)
- **Email**: YOUR_EMAIL@example.com

---

⭐ If you find this project useful for your research or work, please consider giving it a star on GitHub!
