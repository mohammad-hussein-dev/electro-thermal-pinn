"""
Electro-Thermal Physics-Informed Neural Network (PINN) models.

This module provides three architectures for solving coupled electro-thermal PDEs:
1. ElectroThermalPINN: Standard fully-connected MLP (default, lightweight).
2. TransformerPINN: Full Transformer-based model with attention (advanced, memory-intensive).
3. MLPPINN: Lightweight transformer-style model without attention (uses RMSNorm + SwiGLU MLP).

All models take spatial (x) and temporal (t) coordinates as inputs and predict
three physical fields: electric field (E), magnetic field (H), and temperature (T).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

# =============================================================================
#  PART 1: STANDARD MLP MODEL (ORIGINAL, LIGHTWEIGHT)
# =============================================================================

class ElectroThermalPINN(nn.Module):
    """
    Fully-connected neural network for coupled electro-thermal problems.

    This is the default PINN architecture, using a simple multi-layer perceptron
    with Tanh activations and Xavier initialization. It is lightweight, fast,
    and serves as a reliable baseline for most problems.

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
                Defaults to [2, 50, 50, 50, 50, 3], where:
                - Input: 2 (x, t)
                - Hidden: 4 layers of 50 neurons each
                - Output: 3 (E, H, T)
        """
        super(ElectroThermalPINN, self).__init__()

        if layers is None:
            layers = [2, 50, 50, 50, 50, 3]

        self.layers = layers
        self.activation = nn.Tanh()

        # Build fully-connected layers
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i + 1]))

        # Xavier initialization for stable training
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MLP network.

        Args:
            x (torch.Tensor): Spatial coordinates, shape (N, 1) or (B, S, 1).
            t (torch.Tensor): Temporal coordinates, same shape as x.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (E, H, T) fields.
                Each tensor has the same shape as the input (N, 1) or (B, S, 1).

        Note:
            The output layer has no activation function to allow unbounded outputs.
        """
        # Ensure inputs are at least 2D
        if x.dim() == 1:
            x = x.unsqueeze(-1)
            t = t.unsqueeze(-1)

        # Handle both batched and unbatched inputs
        single_batch = False
        if x.dim() == 2 and x.shape[-1] == 1:
            x = x.unsqueeze(0)
            t = t.unsqueeze(0)
            single_batch = True
        elif x.dim() == 2 and x.shape[-1] != 1:
            x = x.unsqueeze(-1)
            t = t.unsqueeze(-1)

        # Concatenate inputs: (x, t) -> (..., 2)
        inputs = torch.cat([x, t], dim=-1)

        # Forward pass through hidden layers
        u = inputs
        for i in range(len(self.linears) - 1):
            u = self.activation(self.linears[i](u))

        # Output layer (no activation)
        output = self.linears[-1](u)

        # Split into physical fields
        E = output[..., 0:1]
        H = output[..., 1:2]
        T = output[..., 2:3]

        # Remove batch dimension if it was added
        if single_batch:
            E = E.squeeze(0)
            H = H.squeeze(0)
            T = T.squeeze(0)

        return E, H, T


# =============================================================================
#  PART 2: TRANSFORMER-BASED MODELS (ADVANCED)
# =============================================================================

@dataclass
class TransformerConfig:
    """
    Configuration for the Transformer-based PINN models.

    Attributes:
        input_dim (int): Number of input coordinates (default 2 for x, t).
        output_dim (int): Number of output fields (default 3 for E, H, T).
        hidden_size (int): Hidden dimension of the transformer.
        intermediate_size (int): Intermediate dimension in the SwiGLU MLP.
        num_hidden_layers (int): Number of transformer decoder layers.
        num_attention_heads (int): Number of query attention heads.
        num_key_value_heads (int): Number of key/value heads for Grouped-Query Attention.
        head_dim (int): Dimension per attention head.
        rms_norm_eps (float): Epsilon for RMSNorm.
        hidden_act (str): Activation function for the gated MLP ('silu', 'gelu', 'tanh', 'relu').
        rope_theta (float): Base frequency for Rotary Position Embedding.
        rope_coord_scale (float): Scaling factor applied to coordinates before RoPE.
        use_fourier_features (bool): Whether to use Fourier feature mapping for inputs.
        fourier_num_frequencies (int): Number of frequency vectors in Fourier features.
        fourier_scale (float): Scale of the Gaussian distribution for Fourier frequencies.
        attention_bias (bool): Whether to use bias in attention projections.
        mlp_bias (bool): Whether to use bias in MLP projections.
        attention_dropout (float): Dropout rate for attention weights.
        is_causal (bool): If True, use causal attention; False for bidirectional (PINN).
        max_seq_len (int): Maximum sequence length before chunked processing.
        sigma (float): Electrical conductivity (physics parameter).
        mu (float): Magnetic permeability.
        epsilon (float): Permittivity.
        k_thermal (float): Thermal conductivity.
        rho (float): Mass density.
        cp (float): Specific heat capacity.
    """
    # Problem dimensions
    input_dim: int = 2
    output_dim: int = 3

    # Model dimensions
    hidden_size: int = 256
    intermediate_size: int = 512
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 32

    # Normalization
    rms_norm_eps: float = 1e-6

    # Activation
    hidden_act: str = "silu"

    # RoPE
    rope_theta: float = 100.0
    rope_coord_scale: float = 1.0

    # Coordinate embedding
    use_fourier_features: bool = True
    fourier_num_frequencies: int = 16
    fourier_scale: float = 10.0

    # Bias / dropout
    attention_bias: bool = False
    mlp_bias: bool = False
    attention_dropout: float = 0.0

    # Attention type — bidirectional for PINN
    is_causal: bool = False

    # Chunking for memory management
    max_seq_len: int = 2048

    # Physics parameters (normalized units)
    sigma: float = 1.0
    mu: float = 1.0
    epsilon: float = 1.0
    k_thermal: float = 1.0
    rho: float = 1.0
    cp: float = 1.0


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class CoordinateEmbedding(nn.Module):
    """Embeds continuous physical coordinates using Fourier features."""
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        use_fourier: bool = True,
        num_frequencies: int = 16,
        scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.num_frequencies = num_frequencies
            self.freqs = nn.Parameter(torch.randn(num_frequencies, input_dim) * scale)
            fourier_dim = num_frequencies * 2
            self.proj = nn.Linear(fourier_dim, hidden_size)
        else:
            self.proj = nn.Linear(input_dim, hidden_size)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if self.use_fourier:
            proj = torch.matmul(coords, self.freqs.T)
            sin_feat = torch.sin(proj)
            cos_feat = torch.cos(proj)
            features = torch.cat([sin_feat, cos_feat], dim=-1)
            return self.proj(features)
        return self.proj(coords)


class ContinuousRoPE(nn.Module):
    """Rotary Position Embedding for continuous physical coordinates."""
    def __init__(
        self,
        head_dim: int,
        theta: float = 100.0,
        num_coord_dims: int = 2,
        coord_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.num_coord_dims = num_coord_dims
        self.coord_scale = coord_scale
        dim_per_coord = head_dim // num_coord_dims
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim_per_coord, 2, dtype=torch.float) / dim_per_coord))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, _ = coords.shape
        dim_per_coord = self.head_dim // self.num_coord_dims
        cos_list, sin_list = [], []
        for d in range(self.num_coord_dims):
            coord_d = coords[:, :, d:d+1] * self.coord_scale
            freqs_d = coord_d * self.inv_freq.unsqueeze(0)
            freqs_d = torch.cat([freqs_d, freqs_d], dim=-1)
            cos_list.append(torch.cos(freqs_d))
            sin_list.append(torch.sin(freqs_d))
        cos = torch.cat(cos_list, dim=-1)
        sin = torch.cat(sin_list, dim=-1)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_continuous_rope(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, num_kv_heads, S, D = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(B, num_kv_heads, n_rep, S, D)
    return hidden_states.reshape(B, num_kv_heads * n_rep, S, D)


class Attention(nn.Module):
    """Multi-head attention with Grouped-Query Attention (GQA)."""
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = config.is_causal

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None):
        B, S, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_continuous_rope(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states = repeat_kv(value_states, self.num_kv_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if self.is_causal:
            causal_mask = torch.triu(torch.ones(S, S, device=hidden_states.device, dtype=torch.bool), diagonal=1)
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, S, -1)
        return self.o_proj(attn_output)


class MLP(nn.Module):
    """SwiGLU MLP."""
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = {"silu": F.silu, "gelu": F.gelu, "tanh": torch.tanh, "relu": F.relu}.get(config.hidden_act, F.silu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Transformer decoder layer with pre-normalization."""
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class TransformerPINN(nn.Module):
    """
    Advanced PINN with Transformer architecture (full attention).

    This model replaces the standard MLP backbone with a transformer,
    leveraging Fourier features, Continuous RoPE, Grouped-Query Attention,
    and SwiGLU MLP for improved performance on complex PDE problems.
    (Requires significant GPU memory for large N_f.)
    """

    def __init__(self, config: Optional[TransformerConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = TransformerConfig()
        self.config = config

        self.coord_embedding = CoordinateEmbedding(
            input_dim=config.input_dim,
            hidden_size=config.hidden_size,
            use_fourier=config.use_fourier_features,
            num_frequencies=config.fourier_num_frequencies,
            scale=config.fourier_scale,
        )

        self.rotary_emb = ContinuousRoPE(
            head_dim=config.head_dim,
            theta=config.rope_theta,
            num_coord_dims=config.input_dim,
            coord_scale=config.rope_coord_scale,
        )

        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_head = nn.Linear(config.hidden_size, config.output_dim, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _forward_impl(self, coords: torch.Tensor) -> torch.Tensor:
        hidden_states = self.coord_embedding(coords)
        position_embeddings = self.rotary_emb(coords)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings, attention_mask=None)
        hidden_states = self.norm(hidden_states)
        return self.output_head(hidden_states)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Normalize shapes
        if x.dim() == 1:
            x = x.unsqueeze(-1)
            t = t.unsqueeze(-1)

        single_batch = False
        if x.dim() == 2:
            if x.shape[-1] == 1:
                x = x.unsqueeze(0)
                t = t.unsqueeze(0)
                single_batch = True
            else:
                x = x.unsqueeze(-1)
                t = t.unsqueeze(-1)

        coords = torch.cat([x, t], dim=-1)  # (B, S, 2)
        B, S, _ = coords.shape
        max_len = self.config.max_seq_len

        if S > max_len:
            outputs = []
            for start in range(0, S, max_len):
                end = min(start + max_len, S)
                chunk = coords[:, start:end, :]
                outputs.append(self._forward_impl(chunk))
            output = torch.cat(outputs, dim=1)
        else:
            output = self._forward_impl(coords)

        E = output[..., 0:1]
        H = output[..., 1:2]
        T = output[..., 2:3]

        if single_batch:
            E = E.squeeze(0)
            H = H.squeeze(0)
            T = T.squeeze(0)

        return E, H, T

    def compute_pde_residuals(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute PDE residuals for the coupled electro-thermal equations."""
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        if not t.requires_grad:
            t = t.clone().requires_grad_(True)

        E, H, T = self.forward(x, t)
        ones = torch.ones_like(E)

        E_x = torch.autograd.grad(E, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        E_t = torch.autograd.grad(E, t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        H_x = torch.autograd.grad(H, x, grad_outputs=torch.ones_like(H), create_graph=True, retain_graph=True)[0]
        H_t = torch.autograd.grad(H, t, grad_outputs=torch.ones_like(H), create_graph=True, retain_graph=True)[0]
        T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True)[0]
        T_t = torch.autograd.grad(T, t, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True)[0]
        T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True, retain_graph=True)[0]

        cfg = self.config
        residual_E = E_x + cfg.mu * H_t
        residual_H = H_x + cfg.sigma * E - cfg.epsilon * E_t
        residual_T = cfg.rho * cfg.cp * T_t - cfg.k_thermal * T_xx - cfg.sigma * E.pow(2)

        return residual_E, residual_H, residual_T

    def compute_loss(
        self,
        x_collocation: torch.Tensor,
        t_collocation: torch.Tensor,
        x_boundary: torch.Tensor,
        t_boundary: torch.Tensor,
        E_bc: torch.Tensor,
        H_bc: torch.Tensor,
        T_bc: torch.Tensor,
        x_initial: torch.Tensor,
        t_initial: torch.Tensor,
        T_ic: torch.Tensor,
        lambda_pde: float = 1.0,
        lambda_bc: float = 10.0,
        lambda_ic: float = 10.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute total physics-informed loss."""
        r_E, r_H, r_T = self.compute_pde_residuals(x_collocation, t_collocation)
        loss_pde = torch.mean(r_E.pow(2)) + torch.mean(r_H.pow(2)) + torch.mean(r_T.pow(2))

        E_pred, H_pred, T_pred = self.forward(x_boundary, t_boundary)
        loss_bc = torch.mean((E_pred - E_bc).pow(2)) + torch.mean((H_pred - H_bc).pow(2)) + torch.mean((T_pred - T_bc).pow(2))

        _, _, T_pred_i = self.forward(x_initial, t_initial)
        loss_ic = torch.mean((T_pred_i - T_ic).pow(2))

        total_loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
        return {"total": total_loss, "pde": loss_pde, "bc": loss_bc, "ic": loss_ic}


# =============================================================================
#  PART 3: LIGHTWEIGHT TRANSFORMER VARIANT (NO ATTENTION)
# =============================================================================

class MLPPINN(nn.Module):
    """
    Lightweight PINN using transformer-style components without cross-point attention.

    This variant processes each collocation point independently through
    transformer-style blocks (RMSNorm + SwiGLU MLP + residual connections),
    but skips self-attention to achieve O(N) complexity instead of O(N²).

    It still uses Fourier features and RMSNorm for stability, making it a
    good middle-ground between pure MLP and full Transformer.

    Args:
        config (TransformerConfig): Configuration instance.
    """

    def __init__(self, config: Optional[TransformerConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = TransformerConfig()
        self.config = config

        self.coord_embedding = CoordinateEmbedding(
            input_dim=config.input_dim,
            hidden_size=config.hidden_size,
            use_fourier=config.use_fourier_features,
            num_frequencies=config.fourier_num_frequencies,
            scale=config.fourier_scale,
        )

        # Transformer-style blocks without attention
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                        "norm2": RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                        "mlp": MLP(config),
                    }
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_head = nn.Linear(config.hidden_size, config.output_dim, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Normalize shapes
        if x.dim() == 1:
            x = x.unsqueeze(-1)
            t = t.unsqueeze(-1)

        single_batch = False
        if x.dim() == 2:
            if x.shape[-1] == 1:
                x = x.unsqueeze(0)
                t = t.unsqueeze(0)
                single_batch = True
            else:
                x = x.unsqueeze(-1)
                t = t.unsqueeze(-1)

        coords = torch.cat([x, t], dim=-1)  # (B, S, 2)

        # Embed coordinates
        h = self.coord_embedding(coords)  # (B, S, hidden_size)

        # Process through blocks without attention
        for block in self.blocks:
            # Pre-norm MLP with residual (no attention)
            h = h + block["mlp"](block["norm1"](h))

        h = self.final_norm(h)
        output = self.output_head(h)

        E = output[..., 0:1]
        H = output[..., 1:2]
        T = output[..., 2:3]

        if single_batch:
            E = E.squeeze(0)
            H = H.squeeze(0)
            T = T.squeeze(0)

        return E, H, T

    def compute_pde_residuals(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same PDE residual computation as the other models."""
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        if not t.requires_grad:
            t = t.clone().requires_grad_(True)

        E, H, T = self.forward(x, t)
        ones = torch.ones_like(E)

        E_x = torch.autograd.grad(E, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        E_t = torch.autograd.grad(E, t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        H_x = torch.autograd.grad(H, x, grad_outputs=torch.ones_like(H), create_graph=True, retain_graph=True)[0]
        H_t = torch.autograd.grad(H, t, grad_outputs=torch.ones_like(H), create_graph=True, retain_graph=True)[0]
        T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True)[0]
        T_t = torch.autograd.grad(T, t, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True)[0]
        T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True, retain_graph=True)[0]

        cfg = self.config
        residual_E = E_x + cfg.mu * H_t
        residual_H = H_x + cfg.sigma * E - cfg.epsilon * E_t
        residual_T = cfg.rho * cfg.cp * T_t - cfg.k_thermal * T_xx - cfg.sigma * E.pow(2)

        return residual_E, residual_H, residual_T

    def compute_loss(
        self,
        x_collocation: torch.Tensor,
        t_collocation: torch.Tensor,
        x_boundary: torch.Tensor,
        t_boundary: torch.Tensor,
        E_bc: torch.Tensor,
        H_bc: torch.Tensor,
        T_bc: torch.Tensor,
        x_initial: torch.Tensor,
        t_initial: torch.Tensor,
        T_ic: torch.Tensor,
        lambda_pde: float = 1.0,
        lambda_bc: float = 10.0,
        lambda_ic: float = 10.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute total physics-informed loss."""
        r_E, r_H, r_T = self.compute_pde_residuals(x_collocation, t_collocation)
        loss_pde = torch.mean(r_E.pow(2)) + torch.mean(r_H.pow(2)) + torch.mean(r_T.pow(2))

        E_pred, H_pred, T_pred = self.forward(x_boundary, t_boundary)
        loss_bc = torch.mean((E_pred - E_bc).pow(2)) + torch.mean((H_pred - H_bc).pow(2)) + torch.mean((T_pred - T_bc).pow(2))

        _, _, T_pred_i = self.forward(x_initial, t_initial)
        loss_ic = torch.mean((T_pred_i - T_ic).pow(2))

        total_loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
        return {"total": total_loss, "pde": loss_pde, "bc": loss_bc, "ic": loss_ic}
