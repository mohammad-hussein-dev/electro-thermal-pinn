# Electro-Thermal PINN: Physics-Informed Neural Networks for Coupled Electromagnetic-Thermal Problems

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-PDF-success)](https://github.com/mohammad-hussein-dev/electro-thermal-pinn/blob/main/docs/Electro-Thermal_PINN_Technical_Report.pdf)

A **Physics-Informed Neural Network (PINN)** framework for solving coupled electromagnetic-thermal problems. This repository implements a fully-differentiable, mesh-free solver that simultaneously predicts electric field (E), magnetic field (H), and temperature (T) distributions by embedding Maxwell's equations and the heat equation with Joule heating into the neural network's loss function.

The framework supports **three neural network architectures** with an interactive selection menu at startup, allowing users to choose the best trade-off between speed, accuracy, and memory usage.

---

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
| **Model Selection** | Interactive menu to choose between MLP, MLPPINN, or TransformerPINN |
| **Reproducibility** | Fixed random seeds for consistent results |
| **Early Stopping** | Prevents overfitting with patience-based stopping |
| **Learning Rate Scheduling** | Adaptive LR for stable convergence |
| **L-BFGS Refinement** | Final optimization for higher accuracy |
| **Comprehensive Metrics** | L2 error, MAE, Max error |
| **Interactive Dashboard** | Streamlit-based UI for live parameter exploration and visualization |

---

## 📄 Technical Report

A comprehensive technical report detailing the mathematical formulation, architecture design, training methodology, and extensive results is available:

[**Download the full PDF report**](https://github.com/mohammad-hussein-dev/electro-thermal-pinn/blob/main/docs/Electro-Thermal_PINN_Technical_Report.pdf)

The report covers:
- Governing equations and non-dimensionalization
- PINN architecture and loss function design
- Three model architectures (MLP, MLPPINN, TransformerPINN)
- Comprehensive error analysis and convergence study
- Industrial applications and future work

---

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

---

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

---

## 🏗️ Model Architectures

The framework provides **three architectures** with an interactive selection menu at startup:

| # | Model | Description | Speed | Memory | Accuracy |
|:-:|-------|-------------|:-----:|:------:|:--------:|
| 1 | **MLP** (ElectroThermalPINN) | Fully-connected network with Tanh activation. Lightweight baseline. | ⚡⚡⚡⚡⚡ | 💾💾 | 🎯🎯🎯 |
| 2 | **MLPPINN** (Lightweight Transformer) | Transformer-style without attention. Uses Fourier features, RMSNorm, and SwiGLU. | ⚡⚡⚡⚡ | 💾💾💾 | 🎯🎯🎯🎯 |
| 3 | **TransformerPINN** (Full Attention) | Full Transformer with attention, RoPE, GQA. Highest accuracy, memory-intensive. | ⚡⚡ | 💾💾💾💾💾 | 🎯🎯🎯🎯🎯 |
| 0 | **config.yaml** | Load architecture from configuration file. | — | — | — |

---

## 📊 Results

### Evaluation Metrics (MLP – Default)

| Metric | E (Electric) | H (Magnetic) | T (Temperature) |
|--------|--------------|--------------|-----------------|
| **Relative L2 Error** | 9.09e-04 | 2.43e-03 | 1.42e-03 |
| **MAE** | 1.96e-04 | 3.24e-04 | 7.02e-04 |
| **Max Error** | 3.29e-04 | 6.78e-04 | 2.11e-03 |

### Evaluation Metrics (MLPPINN – Lightweight Transformer)

| Metric | E (Electric) | H (Magnetic) | T (Temperature) |
|--------|--------------|--------------|-----------------|
| **Relative L2 Error** | 2.15e-04 | 8.75e-04 | 1.52e-03 |
| **MAE** | 6.25e-05 | 2.10e-04 | 4.20e-04 |
| **Max Error** | 1.74e-03 | 5.90e-04 | 1.70e-03 |

### Evaluation Metrics (TransformerPINN – Full Attention)

| Metric | E (Electric) | H (Magnetic) | T (Temperature) |
|--------|--------------|--------------|-----------------|
| **Relative L2 Error** | 8.71e-05 | 3.11e-04 | 5.67e-04 |
| **MAE** | 2.53e-05 | 8.70e-05 | 1.80e-04 |
| **Max Error** | 1.20e-04 | 4.20e-04 | 8.90e-04 |

> **Note**: MLP is fastest and performs well for E-field. MLPPINN offers improved accuracy for H and T with moderate cost. TransformerPINN achieves the highest accuracy for all fields at the cost of increased memory and compute time.

---

## 📂 Project Structure

```
electro-thermal-pinn/
├── app.py                             # Streamlit dashboard
├── train_for_dashboard.py             # Quick training script for dashboard models
├── src/
│   ├── main.py                        # Unified entry point (menu for dashboard/terminal)
│   ├── main_terminal.py               # Terminal-based training with model selection
│   ├── models/
│   │   └── electro_thermal_pinn.py    # All three model architectures
│   ├── physics/
│   │   ├── maxwell_pde.py             # Maxwell's equations residuals
│   │   ├── heat_pde.py                # Heat equation with Joule heating
│   │   ├── boundary_conditions.py     # Dirichlet BCs
│   │   └── varying_parameters.py      # Parameter network for α(x), σ(x)
│   ├── training/
│   │   ├── trainer.py                 # Loss function with UQ
│   │   └── trainer_original.py        # Forward problem trainer
│   ├── utils/
│   │   ├── data_sampling.py           # Collocation & boundary points
│   │   ├── plotting.py                # Visualization utilities
│   │   └── config_loader.py           # YAML configuration loader
│   └── configs/
│       └── config.yaml                # Hyperparameter configuration
├── experiments/
│   ├── figures/                       # Generated plots
│   ├── results/                       # .npy prediction files
│   └── saved_models/                  # Trained model weights
├── README.md
├── LICENSE
└── requirements.txt
```

---

## 🛠 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/mohammad-hussein-dev/electro-thermal-pinn.git
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
streamlit>=1.28.0
```

---

## 🚀 Usage

### Main Menu

```bash
python src/main.py
```

This command launches a unified menu with two execution modes:

```
======================================================================
  ⚡ ELECTRO-THERMAL PINN - MAIN MENU
======================================================================

  +---+--------------------------------------------+
  | # | Mode                                       |
  +---+--------------------------------------------+
  | 1 | Dashboard (Streamlit)                      |
  |   | Interactive UI for employers & clients    |
  +---+--------------------------------------------+
  | 2 | Terminal Mode                              |
  |   | Command-line with architecture selection  |
  +---+--------------------------------------------+
  | 0 | Exit                                      |
  +---+--------------------------------------------+
```

- **Option 1**: Launches the Streamlit dashboard for interactive parameter exploration and visualization.
- **Option 2**: Enters the terminal-based training mode with architecture selection.
- **Option 0**: Exits the application.

### Terminal Mode

After selecting option `2`, you will see the architecture selection menu:

```
----------------------------------------------------------------------
  PINN MODEL SELECTION
----------------------------------------------------------------------

  Select the neural network architecture:

  +---+------------------------+------------------------------------------+
  | # | Model                  | Description / Use Case                   |
  +---+------------------------+------------------------------------------+
  | 1 | MLP                    | Fastest, lowest memory, good baseline   |
  |   | (ElectroThermalPINN)   | Quick tests, prototyping                |
  +---+------------------------+------------------------------------------+
  | 2 | MLPPINN                | Better accuracy, moderate speed         |
  |   | (Lightweight Trans.)   | High accuracy on CPU                    |
  +---+------------------------+------------------------------------------+
  | 3 | TransformerPINN        | Highest accuracy, memory-intensive      |
  |   | (Full Attention)       | Research, GPU-accelerated               |
  +---+------------------------+------------------------------------------+
  | 0 | config.yaml            | Load from configuration file            |
  +---+------------------------+------------------------------------------+
```

### Configuration

All hyperparameters are managed via `src/configs/config.yaml`:

```yaml
problem:
  N_f: 500                  # Number of collocation points

network:
  layers: [2, 50, 50, 50, 50, 3]  # MLP architecture

training:
  adam_lr: 1e-3
  adam_iters: 3000
  use_lbfgs: true

inverse:
  enable: true
  lambda_data: 100.0
  n_data_points: 500
  noise_level: 0.005
  use_uncertainty: true
  varying_params: true

model:
  type: "transformer"       # "mlp" or "transformer" (MLPPINN)
  transformer:
    hidden_size: 128
    num_hidden_layers: 2
    use_fourier_features: false
    max_seq_len: 512
```

### Customization

- **Model Selection**: Use the interactive menu or set `model.type` in `config.yaml`
- **Accuracy vs Speed**: Adjust `N_f`, `hidden_size`, and `num_hidden_layers`
- **Forward Problem Only**: Set `enable: false` under `inverse`
- **Different Physics**: Modify PDE residuals in `src/physics/`

---

## 📊 Interactive Dashboard

The Streamlit dashboard (`app.py`) provides a visual interface for exploring model predictions. It is launched from the main menu (Option 1).

**Features:**
- **Model Architecture Selection**: Choose between MLP, MLPPINN, and TransformerPINN.
- **Parameter Adjustment**: Adjust spatial domain (`x_min`, `x_max`) and time (`t`).
- **Live Inference**: Run the model with the selected parameters and see plots for:
    - Electric field (E)
    - Magnetic field (H)
    - Temperature (T)
    - Spatially-varying parameters (α(x), σ(x))
- **Metrics Display**: View mean, standard deviation, min, and max for each field.

---

## 🔬 Inverse Problem Capabilities

This framework supports **inverse problems** to identify unknown physical parameters from limited observational data:

- **Constant Parameters**: Identify global `α` and `σ` values
- **Spatially-Varying Parameters**: Reconstruct `α(x)` and `σ(x)` fields
- **Uncertainty Quantification**: Variance-based confidence estimation

This capability is critical for applications where direct measurement of material properties is expensive or impossible.

---

## 🎯 Industrial Applications

This framework is suitable for a wide range of industrial applications:

| Industry | Application | Impact |
|----------|-------------|--------|
| **Electronics** | Thermal management of ICs, PCB design, electronic packaging | Reduced prototyping costs, improved reliability |
| **Energy** | Battery cell design, electro-thermal modeling of Li-ion cells | Enhanced safety, extended battery life |
| **Manufacturing** | Induction heating, electroslag remelting, welding processes | Optimized process parameters, reduced energy consumption |
| **Aerospace** | Thermal protection systems, electromagnetic shielding | Improved safety margins, weight reduction |
| **Biomedical** | Hyperthermia treatment planning, RF ablation | Personalized treatment, reduced side effects |

---

## 🤝 Contributors

We would like to thank the following contributors for their valuable work on this project:

- **[SMNRFD](https://github.com/SMNRFD)** – Designed and implemented the full **TransformerPINN** architecture, including:
  - Rotary Position Embedding (RoPE)
  - Grouped-Query Attention (GQA)
  - SwiGLU gated MLP
  - Fourier feature coordinate embedding
  - RMSNorm with pre-norm residuals

This collaboration significantly enhanced the model's accuracy and scalability, especially for magnetic field (H) and temperature (T) predictions.

---

## 📚 References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics.

2. Karniadakis, G. E., et al. (2021). *Physics-informed machine learning.* Nature Reviews Physics.

3. *Physics-Informed Neural Networks for Multiphysics Simulations: Application to Coupled Electromagnetic-Thermal Modeling.* IEEE Access, 2023.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions, collaborations, or feedback:
- **GitHub Issues**: [Open an issue](https://github.com/mohammad-hussein-dev/electro-thermal-pinn/issues)
- **Email**: king.mohamd.09876@gmail.com
- **LinkedIn**: [mohammad-hussein-dev](https://linkedin.com/in/mohammad-hussein-dev)

---

⭐ If you find this project useful for your research or work, please consider giving it a star on GitHub!
