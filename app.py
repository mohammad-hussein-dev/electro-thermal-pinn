"""
Electro-Thermal PINN: Interactive Dashboard with Streamlit.
"""

import os
import sys
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models.electro_thermal_pinn import (
    ElectroThermalPINN,
    MLPPINN,
    TransformerPINN,
    TransformerConfig,
)
from physics.varying_parameters import ParameterNetwork
from utils.config_loader import load_config
from utils.plotting import set_plot_style


st.set_page_config(
    page_title="Electro-Thermal PINN Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model(model_type: str, config_path: str, device: str = "cpu"):
    """Load the trained model and parameter network."""
    config = load_config(config_path)
    device = torch.device(device)

    if model_type == "mlp":
        layers = config["network"]["layers"]
        model = ElectroThermalPINN(layers).to(device)
    elif model_type == "mlppinn":
        cfg = TransformerConfig(**config.get("model", {}).get("transformer", {}))
        model = MLPPINN(cfg).to(device)
    elif model_type == "transformer":
        cfg = TransformerConfig(**config.get("model", {}).get("transformer", {}))
        model = TransformerPINN(cfg).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model_path = "experiments/saved_models/electro_thermal_pinn.pt"
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception:
            pass

    param_net = ParameterNetwork(input_dim=1, output_dim=2).to(device)
    param_path = "experiments/saved_models/param_net.pt"
    if os.path.exists(param_path):
        try:
            param_net.load_state_dict(torch.load(param_path, map_location=device))
        except Exception:
            pass

    model.eval()
    param_net.eval()
    return model, param_net, device


def predict(model, param_net, x, t, device):
    """Run inference on the model."""
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
        E, H, T = model(x_tensor, t_tensor)
        E = E.cpu().numpy()
        H = H.cpu().numpy()
        T = T.cpu().numpy()
        params = param_net(x_tensor)
        alpha = params[:, 0].cpu().numpy()
        sigma = params[:, 1].cpu().numpy()
    return E, H, T, alpha, sigma


st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

model_type = st.sidebar.selectbox(
    "Model Architecture",
    options=["mlp", "mlppinn", "transformer"],
    format_func=lambda x: {
        "mlp": "MLP (ElectroThermalPINN) - Fastest",
        "mlppinn": "MLPPINN (Lightweight Transformer) - Better accuracy on CPU",
        "transformer": "TransformerPINN (Full Attention) - Highest accuracy",
    }.get(x, x),
)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Input Parameters")

x_min = st.sidebar.slider("x min", 0.0, 1.0, 0.0, 0.01)
x_max = st.sidebar.slider("x max", 0.0, 1.0, 1.0, 0.01)
t_fixed = st.sidebar.slider("Time (t)", 0.0, 1.0, 0.5, 0.01)
n_points = st.sidebar.slider("Number of points", 50, 500, 200, 10)
config_path = st.sidebar.text_input("Config Path", value="src/configs/config.yaml")
load_button = st.sidebar.button("🔄 Load Model", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Made with ❤️ using Streamlit")


st.title("⚡ Electro-Thermal PINN Dashboard")
st.markdown(
    """
    Interactive dashboard for **Physics-Informed Neural Network (PINN)** solving
    coupled Maxwell's equations and heat equation with Joule heating.
    """
)

if load_button or "model_loaded" not in st.session_state:
    with st.spinner("Loading model..."):
        try:
            model, param_net, device = load_model(model_type, config_path)
            st.session_state.model = model
            st.session_state.param_net = param_net
            st.session_state.device = device
            st.session_state.model_loaded = True
            st.session_state.model_type = model_type
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.stop()

if not st.session_state.get("model_loaded", False):
    st.info("👈 Click **'Load Model'** to start.")
    st.stop()

model = st.session_state.model
param_net = st.session_state.param_net
device = st.session_state.device

x_test = np.linspace(x_min, x_max, n_points).reshape(-1, 1)
t_test = np.full_like(x_test, t_fixed)

with st.spinner("Running inference..."):
    E_pred, H_pred, T_pred, alpha_pred, sigma_pred = predict(
        model, param_net, x_test, t_test, device
    )

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("⚡ Electric Field (E)", f"{E_pred.mean():.4f} ± {E_pred.std():.4f}")
with col2:
    st.metric("🧲 Magnetic Field (H)", f"{H_pred.mean():.4f} ± {H_pred.std():.4f}")
with col3:
    st.metric("🌡️ Temperature (T)", f"{T_pred.mean():.4f} ± {T_pred.std():.4f}")

st.markdown("---")
st.subheader("📊 Predicted Fields")

fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
set_plot_style()
colors = {"E": "#1f77b4", "H": "#ff7f0e", "T": "#2ca02c"}
axes1[0].plot(x_test, E_pred, color=colors["E"], linewidth=2.5)
axes1[0].set_xlabel("x")
axes1[0].set_ylabel("E (Electric Field)")
axes1[0].set_title("Electric Field")
axes1[0].grid(True, alpha=0.3)

axes1[1].plot(x_test, H_pred, color=colors["H"], linewidth=2.5)
axes1[1].set_xlabel("x")
axes1[1].set_ylabel("H (Magnetic Field)")
axes1[1].set_title("Magnetic Field")
axes1[1].grid(True, alpha=0.3)

axes1[2].plot(x_test, T_pred, color=colors["T"], linewidth=2.5)
axes1[2].set_xlabel("x")
axes1[2].set_ylabel("T (Temperature)")
axes1[2].set_title("Temperature")
axes1[2].grid(True, alpha=0.3)

st.pyplot(fig1)
plt.close(fig1)

st.subheader("📊 Spatially-Varying Parameters")
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
set_plot_style()
axes2[0].plot(x_test, alpha_pred, "b-", linewidth=2.5)
axes2[0].set_xlabel("x")
axes2[0].set_ylabel(r"$\alpha(x)$")
axes2[0].set_title("Thermal Diffusivity")
axes2[0].grid(True, alpha=0.3)

axes2[1].plot(x_test, sigma_pred, "r-", linewidth=2.5)
axes2[1].set_xlabel("x")
axes2[1].set_ylabel(r"$\sigma(x)$")
axes2[1].set_title("Electrical Conductivity")
axes2[1].grid(True, alpha=0.3)

st.pyplot(fig2)
plt.close(fig2)

st.subheader("📈 Statistics")
metrics = {
    "Field": ["E", "H", "T"],
    "Mean": [E_pred.mean(), H_pred.mean(), T_pred.mean()],
    "Std": [E_pred.std(), H_pred.std(), T_pred.std()],
    "Min": [E_pred.min(), H_pred.min(), T_pred.min()],
    "Max": [E_pred.max(), H_pred.max(), T_pred.max()],
}
st.dataframe(
    metrics,
    column_config={
        "Field": st.column_config.TextColumn("Field"),
        "Mean": st.column_config.NumberColumn("Mean", format="%.4f"),
        "Std": st.column_config.NumberColumn("Std", format="%.4f"),
        "Min": st.column_config.NumberColumn("Min", format="%.4f"),
        "Max": st.column_config.NumberColumn("Max", format="%.4f"),
    },
    hide_index=True,
    use_container_width=True,
)

st.markdown("---")
st.caption("Powered by PyTorch • Physics-Informed Neural Networks • Streamlit")
