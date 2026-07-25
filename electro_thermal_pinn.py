"""
Advanced Physics-Informed Neural Network (PINN).

This module adapts key  transformer components — RMSNorm, Rotary Position
Embeddings (RoPE), Grouped-Query Attention (GQA), and SwiGLU gated MLP — for
solving coupled electro-thermal partial differential equations (PDEs).

The network takes spatial (x) and temporal (t) coordinates as inputs and predicts
three physical fields: electric field (E), magnetic field (H), and temperature (T).

Architecture overview:
    Input (x, t)
      → Coordinate Embedding  (Fourier features + linear projection)
      → Continuous RoPE        (rotary embeddings from physical coordinates)
      → N ×  Decoder Layer (Self-Attention + SwiGLU MLP, pre-norm residuals)
      → Final RMSNorm
      → Output Head → (E, H, T)

Physics-informed training computes PDE residuals via automatic differentiation:
    - Faraday's law:   ∂E/∂x + μ ∂H/∂t = 0
    - Ampere's law:    ∂H/∂x + σE − ε ∂E/∂t = 0
    - Heat equation:   ρcₚ ∂T/∂t − k ∂²T/∂x² − σE² = 0

Developers: Mohammad Hussein & Yasin Aryanfard (ysnrfd).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict



# Configuration

@dataclass
class PINNConfig:
    """Configuration for the -based PINN model.

    Attributes:
        input_dim: Number of input coordinates (default 2 for x, t).
        output_dim: Number of output fields (default 3 for E, H, T).
        hidden_size: Hidden dimension of the transformer.
        intermediate_size: Intermediate dimension in the SwiGLU MLP.
        num_hidden_layers: Number of  decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads (for GQA).
        head_dim: Dimension per attention head.
        rms_norm_eps: Epsilon for RMSNorm.
        hidden_act: Activation function for the gated MLP.
        rope_theta: Base frequency for RoPE.
        rope_coord_scale: Scaling factor applied to coordinates before RoPE.
        use_fourier_features: Whether to use Fourier feature mapping for inputs.
        fourier_num_frequencies: Number of frequency vectors in Fourier features.
        fourier_scale: Scale of the Gaussian distribution for Fourier frequencies.
        attention_bias: Whether to use bias in attention projections.
        mlp_bias: Whether to use bias in MLP projections.
        attention_dropout: Dropout rate for attention weights.
        is_causal: If True, use causal attention; False for bidirectional (PINN).
        max_seq_len: Maximum sequence length before chunked processing kicks in.
        sigma: Electrical conductivity (physics parameter).
        mu: Magnetic permeability.
        epsilon: Permittivity.
        k_thermal: Thermal conductivity.
        rho: Mass density.
        cp: Specific heat capacity.
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



#  RMSNorm

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (from ).

    Computes:  output = weight * x / sqrt(mean(x²) + eps)

    This is equivalent to T5LayerNorm and avoids the mean-subtraction step
    of standard LayerNorm, providing stable training for deep networks.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"



# Coordinate Embedding with Fourier Features

class CoordinateEmbedding(nn.Module):
    """Embeds continuous physical coordinates into a high-dimensional space.

    Uses random Fourier features (Rahimi & Recht, 2007) to map low-frequency
    coordinate inputs into a richer representation that helps the network
    capture high-frequency variations — critical for PDE solutions.

    Mapping:  φ(x) = [sin(w₁ᵀx), cos(w₁ᵀx), ..., sin(wₘᵀx), cos(wₘᵀx)]
    where wᵢ are learnable frequency vectors.

    Args:
        input_dim: Dimension of input coordinates (e.g., 2 for x, t).
        hidden_size: Output embedding dimension.
        use_fourier: If True, apply Fourier feature mapping before projection.
        num_frequencies: Number of frequency vectors m.
        scale: Initial scale of the frequency distribution.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        use_fourier: bool = True,
        num_frequencies: int = 16,
        scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.use_fourier = use_fourier

        if use_fourier:
            self.num_frequencies = num_frequencies
            # Learnable frequency vectors, initialized from a scaled Gaussian
            self.freqs = nn.Parameter(
                torch.randn(num_frequencies, input_dim) * scale
            )
            fourier_dim = num_frequencies * 2  # sin + cos
            self.proj = nn.Linear(fourier_dim, hidden_size)
        else:
            self.proj = nn.Linear(input_dim, hidden_size)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, S, input_dim) continuous coordinate values.

        Returns:
            (B, S, hidden_size) embedded representations.
        """
        if self.use_fourier:
            # Dot-product with frequency vectors: (B, S, num_freq)
            proj = torch.matmul(coords, self.freqs.T)
            sin_feat = torch.sin(proj)
            cos_feat = torch.cos(proj)
            features = torch.cat([sin_feat, cos_feat], dim=-1)  # (B, S, 2*num_freq)
            return self.proj(features)
        else:
            return self.proj(coords)



# Continuous Rotary Position Embedding (RoPE)

class ContinuousRoPE(nn.Module):
    """Rotary Position Embedding adapted for continuous physical coordinates.

    Standard RoPE uses discrete integer position indices. This implementation
    generalizes it to continuous coordinates (e.g., spatial position x and
    time t), enabling the model to encode physical proximity directly.

    The head dimension is partitioned equally among coordinate dimensions.
    For each coordinate dimension d, rotary embeddings are computed as:

        freq_d = coord_d * inv_freq
        cos_d = cos(freq_d),  sin_d = sin(freq_d)

    and concatenated to form the full (cos, sin) vectors of size head_dim.

    Args:
        head_dim: Dimension per attention head.
        theta: Base frequency for the inverse frequency computation.
        num_coord_dims: Number of coordinate dimensions (e.g., 2 for x, t).
        coord_scale: Scaling factor applied to coordinates before frequency
            multiplication.
    """

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
        # Inverse frequencies: 1 / theta^(2i/d) for i = 0, 2, 4, ...
        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, dim_per_coord, 2, dtype=torch.float) / dim_per_coord)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coords: (B, S, num_coord_dims) continuous coordinate values.

        Returns:
            cos: (B, S, head_dim) cosine components of the rotary embedding.
            sin: (B, S, head_dim) sine components.
        """
        B, S, _ = coords.shape
        dim_per_coord = self.head_dim // self.num_coord_dims

        cos_list: List[torch.Tensor] = []
        sin_list: List[torch.Tensor] = []

        for d in range(self.num_coord_dims):
            # Scale coordinate value
            coord_d = coords[:, :, d : d + 1] * self.coord_scale  # (B, S, 1)
            # Multiply by inverse frequencies: (B, S, dim_per_coord // 2)
            freqs_d = coord_d * self.inv_freq.unsqueeze(0)
            # Duplicate to fill dim_per_coord
            freqs_d = torch.cat([freqs_d, freqs_d], dim=-1)  # (B, S, dim_per_coord)
            cos_list.append(torch.cos(freqs_d))
            sin_list.append(torch.sin(freqs_d))

        cos = torch.cat(cos_list, dim=-1)  # (B, S, head_dim)
        sin = torch.cat(sin_list, dim=-1)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates the second half of the last dimension to the front (negated)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_continuous_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies continuous RoPE to query and key tensors.

    Args:
        q: (B, num_heads, S, head_dim)
        k: (B, num_kv_heads, S, head_dim)
        cos: (B, S, head_dim)
        sin: (B, S, head_dim)

    Returns:
        Rotated (q, k) with the same shapes.
    """
    cos = cos.unsqueeze(1)  # (B, 1, S, head_dim) — broadcast over heads
    sin = sin.unsqueeze(1)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed



# Grouped-Query Attention

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key/value heads to match the number of query heads (GQA).

    (B, num_kv_heads, S, D) → (B, num_kv_heads * n_rep, S, D)
    """
    B, num_kv_heads, S, D = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        B, num_kv_heads, n_rep, S, D
    )
    return hidden_states.reshape(B, num_kv_heads * n_rep, S, D)


class Attention(nn.Module):
    """Multi-head attention with Grouped-Query Attention (GQA) from .

    Supports bidirectional attention (default for PINN) or causal masking.
    Rotary position embeddings are applied to queries and keys.

    Args:
        config: PINNConfig instance.
    """

    def __init__(self, config: PINNConfig) -> None:
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

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, S, hidden_size)
            position_embeddings: Optional (cos, sin) tuple from ContinuousRoPE.
            attention_mask: Optional additive mask of shape (S, S) or (B, 1, S, S).

        Returns:
            (B, S, hidden_size) attention output.
        """
        B, S, _ = hidden_states.shape

        # Project and reshape to (B, H, S, head_dim)
        query_states = (
            self.q_proj(hidden_states)
            .view(B, S, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(B, S, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(B, S, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_continuous_rope(
                query_states, key_states, cos, sin
            )

        # Expand KV heads for GQA
        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states = repeat_kv(value_states, self.num_kv_groups)

        # Scaled dot-product attention
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(S, S, device=hidden_states.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_weights = attn_weights.masked_fill(
                causal_mask, float("-inf")
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, S, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output



# SwiGLU MLP

class MLP(nn.Module):
    """SwiGLU (Swish-Gated Linear Unit) MLP from .

    Computes:  down_proj(act_fn(gate_proj(x)) * up_proj(x))

    This gated activation provides a richer nonlinearity than standard
    feed-forward layers, improving the model's ability to represent
    complex PDE solutions.

    Args:
        config: PINNConfig instance.
    """

    _ACT_MAP = {
        "silu": F.silu,
        "gelu": F.gelu,
        "tanh": torch.tanh,
        "relu": F.relu,
    }

    def __init__(self, config: PINNConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = self._ACT_MAP.get(config.hidden_act, F.silu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))



#  Decoder Layer

class DecoderLayer(nn.Module):
    """A single  decoder layer with pre-normalization.

    Structure (pre-norm):
        h = h + Attn(RMSNorm(h))        # self-attention with residual
        h = h + MLP(RMSNorm(h))         # SwiGLU MLP with residual

    The pre-norm design ensures stable gradient flow through deep stacks,
    which is critical for PINNs that require second-order derivatives.

    Args:
        config: PINNConfig instance.
    """

    def __init__(self, config: PINNConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention sub-layer
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # MLP sub-layer
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states



# Main Model: PINN

class PINN(nn.Module):
    """Advanced Physics-Informed Neural Network based on the  architecture.

    This model replaces the standard fully-connected PINN backbone with a
    -style transformer, leveraging:

    1. **Fourier feature coordinate embedding** — maps continuous (x, t)
       coordinates into a rich high-dimensional space to overcome the
       spectral bias of neural networks.
    2. **Continuous RoPE** — encodes physical coordinate proximity directly
       into attention via rotary embeddings, enabling position-aware
       information sharing between collocation points.
    3. **Grouped-Query Attention** — efficient multi-head attention with
       reduced KV-cache memory, allowing collocation points to exchange
       information for globally consistent PDE solutions.
    4. **SwiGLU MLP** — gated activation function with higher expressive
       power than standard ReLU/Tanh feed-forward layers.
    5. **RMSNorm + Pre-norm residuals** — stable training through deep
       stacks, essential when second-order derivatives are needed for
       physics-informed loss computation.

    The model predicts three coupled fields:
    - **E** (electric field)
    - **H** (magnetic field)
    - **T** (temperature)

    governed by the 1D coupled electro-thermal PDEs:
    - Faraday's law:    ∂E/∂x + μ ∂H/∂t = 0
    - Ampere's law:     ∂H/∂x + σE − ε ∂E/∂t = 0
    - Heat equation:    ρcₚ ∂T/∂t − k ∂²T/∂x² − σE² = 0

    Args:
        config: PINNConfig instance. If None, uses defaults.

    Note:
        Collocation points are processed as a sequence, with self-attention
        enabling inter-point communication. For large numbers of points
        (>max_seq_len), chunked processing is used automatically, though
        inter-chunk attention is lost.

    Example:
        >>> config = PINNConfig(num_hidden_layers=6, hidden_size=256)
        >>> model = PINN(config)
        >>> x = torch.rand(1000, 1)
        >>> t = torch.rand(1000, 1)
        >>> E, H, T = model(x, t)
        >>> residuals = model.compute_pde_residuals(x, t)
    """

    def __init__(self, config: Optional[PINNConfig] = None) -> None:
        super().__init__()

        if config is None:
            config = PINNConfig()
        self.config = config

        #  Input embedding 
        self.coord_embedding = CoordinateEmbedding(
            input_dim=config.input_dim,
            hidden_size=config.hidden_size,
            use_fourier=config.use_fourier_features,
            num_frequencies=config.fourier_num_frequencies,
            scale=config.fourier_scale,
        )

        #  Continuous RoPE 
        self.rotary_emb = ContinuousRoPE(
            head_dim=config.head_dim,
            theta=config.rope_theta,
            num_coord_dims=config.input_dim,
            coord_scale=config.rope_coord_scale,
        )

        #  Decoder layers 
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        #  Final norm and output head 
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_head = nn.Linear(
            config.hidden_size, config.output_dim, bias=True
        )

        #  Weight initialization 
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier normal for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _forward_impl(self, coords: torch.Tensor) -> torch.Tensor:
        """Internal forward pass for a single chunk.

        Args:
            coords: (B, S, input_dim) coordinate tensor.

        Returns:
            (B, S, output_dim) raw output tensor.
        """
        # Embed coordinates
        hidden_states = self.coord_embedding(coords)  # (B, S, hidden_size)

        # Compute continuous rotary embeddings
        position_embeddings = self.rotary_emb(coords)  # (cos, sin)

        # Pass through decoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
            )

        # Final norm and output projection
        hidden_states = self.norm(hidden_states)
        output = self.output_head(hidden_states)  # (B, S, output_dim)
        return output

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the PINN.

        Args:
            x: Spatial coordinates. Supported shapes:
                - (N,) or (N, 1): treated as a single sequence of N points.
                - (B, S) or (B, S, 1): batched sequences.
            t: Temporal coordinates, same shape as x.

        Returns:
            Tuple of (E, H, T), each matching the input's leading shape:
                - If input is (N, 1): outputs are (N, 1).
                - If input is (B, S, 1): outputs are (B, S, 1).

        Note:
            The output layer has no activation function, allowing unbounded
            outputs for the physical fields.
        """
        #  Normalize input shapes 
        # Ensure x and t are at least 2D
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # (N,) -> (N, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Determine if we need to add a batch dimension
        single_batch = False
        if x.dim() == 2:
            if x.shape[-1] == 1:
                # (N, 1) -> (1, N, 1): treat all points as one sequence
                x = x.unsqueeze(0)
                t = t.unsqueeze(0)
                single_batch = True
            else:
                # (B, S) -> (B, S, 1)
                x = x.unsqueeze(-1)
                t = t.unsqueeze(-1)
        # else: already (B, S, 1)

        # Concatenate coordinates: (B, S, input_dim)
        coords = torch.cat([x, t], dim=-1)

        B, S, _ = coords.shape
        max_len = self.config.max_seq_len

        #  Chunked processing for large sequences 
        if S > max_len:
            outputs = []
            for start in range(0, S, max_len):
                end = min(start + max_len, S)
                chunk = coords[:, start:end, :]
                outputs.append(self._forward_impl(chunk))
            output = torch.cat(outputs, dim=1)
        else:
            output = self._forward_impl(coords)

        #  Split into physical fields 
        E = output[..., 0:1]  # (B, S, 1)
        H = output[..., 1:2]
        T = output[..., 2:3]

        #  Restore original shape 
        if single_batch:
            E = E.squeeze(0)  # (N, 1)
            H = H.squeeze(0)
            T = T.squeeze(0)

        return E, H, T

    
    # Physics-Informed Loss Computation
    
    def compute_pde_residuals(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute PDE residuals for the coupled electro-thermal equations.

        The residuals are computed using automatic differentiation:

        1. Faraday's law:    R_E = ∂E/∂x + μ ∂H/∂t
        2. Ampere's law:     R_H = ∂H/∂x + σE − ε ∂E/∂t
        3. Heat equation:    R_T = ρcₚ ∂T/∂t − k ∂²T/∂x² − σE²

        Args:
            x: Spatial coordinates (N, 1). Must have requires_grad=True or
               will be set internally.
            t: Temporal coordinates (N, 1).

        Returns:
            Tuple of (R_E, R_H, R_T), each of shape (N, 1).
            Ideally these should be zero when the PDEs are satisfied.
        """
        # Ensure gradients can be computed
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        if not t.requires_grad:
            t = t.clone().requires_grad_(True)

        # Forward pass
        E, H, T = self.forward(x, t)

        #  First-order derivatives 
        ones = torch.ones_like(E)

        E_x = torch.autograd.grad(
            E, x, grad_outputs=ones, create_graph=True, retain_graph=True
        )[0]
        E_t = torch.autograd.grad(
            E, t, grad_outputs=ones, create_graph=True, retain_graph=True
        )[0]

        H_x = torch.autograd.grad(
            H, x, grad_outputs=torch.ones_like(H), create_graph=True, retain_graph=True
        )[0]
        H_t = torch.autograd.grad(
            H, t, grad_outputs=torch.ones_like(H), create_graph=True, retain_graph=True
        )[0]

        T_x = torch.autograd.grad(
            T, x, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True
        )[0]
        T_t = torch.autograd.grad(
            T, t, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True
        )[0]

        #  Second-order derivative for T 
        T_xx = torch.autograd.grad(
            T_x,
            x,
            grad_outputs=torch.ones_like(T_x),
            create_graph=True,
            retain_graph=True,
        )[0]

        #  PDE residuals 
        cfg = self.config

        # Faraday's law: ∂E/∂x + μ ∂H/∂t = 0
        residual_E = E_x + cfg.mu * H_t

        # Ampere's law: ∂H/∂x + σE − ε ∂E/∂t = 0
        residual_H = H_x + cfg.sigma * E - cfg.epsilon * E_t

        # Heat equation: ρcₚ ∂T/∂t − k ∂²T/∂x² − σE² = 0
        residual_T = (
            cfg.rho * cfg.cp * T_t
            - cfg.k_thermal * T_xx
            - cfg.sigma * E.pow(2)
        )

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
        """Compute the total physics-informed training loss.

        The total loss is a weighted sum of:
        - **PDE residual loss**: enforces the governing equations at
          collocation points (interior of the domain).
        - **Boundary condition loss**: enforces prescribed values of E, H, T
          on the spatial boundary.
        - **Initial condition loss**: enforces the initial temperature
          distribution at t = 0.

        Args:
            x_collocation, t_collocation: Interior collocation points (N_c, 1).
            x_boundary, t_boundary: Boundary points (N_b, 1).
            E_bc, H_bc, T_bc: Boundary condition values (N_b, 1).
            x_initial, t_initial: Initial condition points (N_i, 1).
            T_ic: Initial temperature values (N_i, 1).
            lambda_pde: Weight for PDE residual loss.
            lambda_bc: Weight for boundary condition loss.
            lambda_ic: Weight for initial condition loss.

        Returns:
            Dict with keys 'total', 'pde', 'bc', 'ic', and individual
            residual losses 'res_E', 'res_H', 'res_T'.
        """
        #  PDE residual loss 
        r_E, r_H, r_T = self.compute_pde_residuals(x_collocation, t_collocation)
        loss_pde_E = torch.mean(r_E.pow(2))
        loss_pde_H = torch.mean(r_H.pow(2))
        loss_pde_T = torch.mean(r_T.pow(2))
        loss_pde = loss_pde_E + loss_pde_H + loss_pde_T

        #  Boundary condition loss 
        E_pred_b, H_pred_b, T_pred_b = self.forward(x_boundary, t_boundary)
        loss_bc = (
            torch.mean((E_pred_b - E_bc).pow(2))
            + torch.mean((H_pred_b - H_bc).pow(2))
            + torch.mean((T_pred_b - T_bc).pow(2))
        )

        #  Initial condition loss 
        _, _, T_pred_i = self.forward(x_initial, t_initial)
        loss_ic = torch.mean((T_pred_i - T_ic).pow(2))

        #  Total loss 
        total_loss = (
            lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
        )

        return {
            "total": total_loss,
            "pde": loss_pde,
            "pde_E": loss_pde_E,
            "pde_H": loss_pde_H,
            "pde_T": loss_pde_T,
            "bc": loss_bc,
            "ic": loss_ic,
        }



# Convenience: Lightweight variant (no attention mixing, per-point processing)

class MLPPINN(nn.Module):
    """Lightweight PINN using  components without cross-point attention.

    This variant processes each collocation point independently through
    -style blocks (RMSNorm + SwiGLU MLP + residual connections), but
    skips the self-attention to achieve O(N) complexity instead of O(N²).
    RoPE is still applied within each point's representation for
    coordinate-awareness.

    Use this when the number of collocation points is very large and
    attention memory is a concern.

    Args:
        config: PINNConfig instance.
    """

    def __init__(self, config: Optional[PINNConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = PINNConfig()
        self.config = config

        self.coord_embedding = CoordinateEmbedding(
            input_dim=config.input_dim,
            hidden_size=config.hidden_size,
            use_fourier=config.use_fourier_features,
            num_frequencies=config.fourier_num_frequencies,
            scale=config.fourier_scale,
        )

        # style blocks without attention
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": RMSNorm(
                            config.hidden_size, eps=config.rms_norm_eps
                        ),
                        "norm2": RMSNorm(
                            config.hidden_size, eps=config.rms_norm_eps
                        ),
                        "mlp": MLP(config),
                    }
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.final_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.output_head = nn.Linear(
            config.hidden_size, config.output_dim, bias=True
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Normalize shapes
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if t.dim() == 1:
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

        coords = torch.cat([x, t], dim=-1)  # (B, S, input_dim)
        h = self.coord_embedding(coords)

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

    def compute_pde_residuals(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same PDE residual computation as PINN."""
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



# Training Utility

class PINNTrainer:
    """Training utility for the -based PINN.

    Handles the training loop with Adam/AdamW optimizer, learning rate
    scheduling, and logging of individual loss components.

    Args:
        model: PINN or MLPPINN instance.
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        scheduler_type: 'cosine', 'step', or None.
        device: Device to train on ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_type: Optional[str] = "cosine",
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler_type = scheduler_type
        self.scheduler = None

    def setup_scheduler(self, total_steps: int) -> None:
        if self.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=1e-6
            )
        elif self.scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=total_steps // 5, gamma=0.5
            )

    def train_step(
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
    ) -> Dict[str, float]:
        """Performs a single training step.

        All input tensors are moved to the training device.

        Returns:
            Dict of loss values (as Python floats).
        """
        # Move data to device
        to_dev = lambda v: v.to(self.device)
        x_c, t_c = to_dev(x_collocation), to_dev(t_collocation)
        x_b, t_b = to_dev(x_boundary), to_dev(t_boundary)
        e_b, h_b, tt_b = to_dev(E_bc), to_dev(H_bc), to_dev(T_bc)
        x_i, t_i = to_dev(x_initial), to_dev(t_initial)
        tt_i = to_dev(T_ic)

        self.model.train()
        self.optimizer.zero_grad()

        losses = self.model.compute_loss(
            x_c, t_c, x_b, t_b, e_b, h_b, tt_b, x_i, t_i, tt_i,
            lambda_pde, lambda_bc, lambda_ic,
        )

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {k: v.item() for k, v in losses.items()}

    def train(
        self,
        num_epochs: int,
        collocation_fn,
        boundary_fn,
        initial_fn,
        lambda_pde: float = 1.0,
        lambda_bc: float = 10.0,
        lambda_ic: float = 10.0,
        print_every: int = 100,
    ) -> List[Dict[str, float]]:
        """Full training loop.

        Args:
            num_epochs: Number of training epochs.
            collocation_fn: Callable returning (x, t) collocation points.
            boundary_fn: Callable returning (x, t, E, H, T) boundary data.
            initial_fn: Callable returning (x, t, T) initial condition data.
            lambda_pde, lambda_bc, lambda_ic: Loss weights.
            print_every: Print interval (in epochs).

        Returns:
            List of loss dictionaries, one per epoch.
        """
        self.setup_scheduler(num_epochs)
        history: List[Dict[str, float]] = []

        for epoch in range(num_epochs):
            x_c, t_c = collocation_fn()
            x_b, t_b, e_b, h_b, tt_b = boundary_fn()
            x_i, t_i, tt_i = initial_fn()

            losses = self.train_step(
                x_c, t_c, x_b, t_b, e_b, h_b, tt_b, x_i, t_i, tt_i,
                lambda_pde, lambda_bc, lambda_ic,
            )
            history.append(losses)

            if (epoch + 1) % print_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch+1:5d}/{num_epochs} | "
                    f"LR={lr:.2e} | "
                    f"Total={losses['total']:.6e} | "
                    f"PDE={losses['pde']:.6e} | "
                    f"BC={losses['bc']:.6e} | "
                    f"IC={losses['ic']:.6e}"
                )

        return history



# Example Usage

if __name__ == "__main__":
    # Configuration
    config = PINNConfig(
        input_dim=2,
        output_dim=3,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        use_fourier_features=True,
        fourier_num_frequencies=16,
        fourier_scale=10.0,
        rope_theta=100.0,
        is_causal=False,
        max_seq_len=1024,
        sigma=1.0,
        mu=1.0,
        epsilon=1.0,
        k_thermal=1.0,
        rho=1.0,
        cp=1.0,
    )

    # Create model
    model = PINN(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"PINN parameters: {num_params:,}")

    # Test forward pass
    N = 500
    x = torch.rand(N, 1)
    t = torch.rand(N, 1)
    E, H, T = model(x, t)
    print(f"Forward pass shapes: E={E.shape}, H={H.shape}, T={T.shape}")

    # Test PDE residual computation
    x.requires_grad_(True)
    t.requires_grad_(True)
    r_E, r_H, r_T = model.compute_pde_residuals(x, t)
    print(f"PDE residual shapes: R_E={r_E.shape}, R_H={r_H.shape}, R_T={r_T.shape}")

    # Test loss computation
    loss_dict = model.compute_loss(
        x_collocation=torch.rand(500, 1),
        t_collocation=torch.rand(500, 1),
        x_boundary=torch.rand(50, 1),
        t_boundary=torch.rand(50, 1),
        E_bc=torch.zeros(50, 1),
        H_bc=torch.zeros(50, 1),
        T_bc=torch.ones(50, 1),
        x_initial=torch.rand(50, 1),
        t_initial=torch.zeros(50, 1),
        T_ic=torch.ones(50, 1),
    )
    print(f"Loss components: {loss_dict}")

    # Quick training demo
    print("\n Quick Training Demo ")
    trainer = PINNTrainer(model, lr=1e-3, device="cpu")

    def sample_collocation():
        return torch.rand(256, 1), torch.rand(256, 1)

    def sample_boundary():
        x = torch.cat([torch.zeros(32, 1), torch.ones(32, 1)])
        t = torch.rand(64, 1)
        return x, t, torch.zeros(64, 1), torch.zeros(64, 1), torch.ones(64, 1)

    def sample_initial():
        x = torch.rand(64, 1)
        t = torch.zeros(64, 1)
        return x, t, torch.ones(64, 1)

    history = trainer.train(
        num_epochs=500,
        collocation_fn=sample_collocation,
        boundary_fn=sample_boundary,
        initial_fn=sample_initial,
        print_every=100,
    )

    print("\nTraining complete.")

    # Also test lightweight variant
    print("\nMLPPINN (Lightweight Variant)")
    model_lite = MLPPINN(config)
    num_params_lite = sum(p.numel() for p in model_lite.parameters())
    print(f"MLPPINN parameters: {num_params_lite:,}")

    E2, H2, T2 = model_lite(x.detach(), t.detach())
    print(f"Forward pass shapes: E={E2.shape}, H={H2.shape}, T={T2.shape}")