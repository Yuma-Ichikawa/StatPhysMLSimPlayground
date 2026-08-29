"""
Sequence Models for In-Context Learning and Time Series.

Includes:
- Linear Self-Attention (LSA): Theoretical minimal Transformer for ICL
- State Space Model (SSM): Alternative to attention for sequence modeling
- Linear RNN: Minimal recurrent model

References:
- Mahankali et al. (2024): One Step of GD is Provably Optimal ICL with LSA
- Lu et al. (2024): Asymptotic theory of ICL by linear attention
- Sushma et al. (2024): SSM can learn in-context by gradient descent

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


def _get_device(model: nn.Module) -> torch.device:
    """Get device from model parameters."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


class LinearSelfAttention(BaseModel):
    """
    Linear Self-Attention (LSA) for In-Context Learning.

    Removes softmax from standard attention, making it amenable to
    theoretical analysis. Key result: LSA implements 1-step gradient
    descent for linear regression in-context.

    Architecture:
        q_i = W_Q h_i
        k_j = W_K h_j
        v_j = W_V h_j

    Projection matrices carry the fan-in normalization at initialization;
    their forward pass is intentionally unscaled.

        Attention: A_i = Σ_j (q_i^T k_j) v_j  (no softmax!)

        Output: o_i = W_O A_i

    For ICL linear regression, the optimal LSA computes:
        θ_1 = θ_0 + (η/ℓ) Σ_t (y_t - θ_0^T x_t / √d) x_t / √d

    Args:
        d: Input/embedding dimension
        d_model: Model dimension (default: same as d)
        init_scale: Initialization scale
        use_output_proj: Whether to use output projection
        normalize: Whether to normalize by sequence length

    """

    def __init__(
        self,
        d: int,
        d_model: int | None = None,
        init_scale: float = 1.0,
        use_output_proj: bool = True,
        normalize: bool = True,
        device: str = "cpu",
    ):
        super().__init__(d=d)
        self.d_model = d_model or d
        self.init_scale = init_scale
        self.use_output_proj = use_output_proj
        self.normalize = normalize
        self._device = device

        # QKV projections
        std = init_scale / np.sqrt(d)
        self.W_q = nn.Parameter(torch.randn(self.d_model, d, device=device) * std)
        self.W_k = nn.Parameter(torch.randn(self.d_model, d, device=device) * std)
        self.W_v = nn.Parameter(torch.randn(self.d_model, d, device=device) * std)
        self.context_label_embedding = nn.Parameter(torch.randn(d, device=device) * std)
        self.query_marker_embedding = nn.Parameter(torch.randn(d, device=device) * std)

        if use_output_proj:
            output_std = init_scale / np.sqrt(self.d_model)
            self.W_o = nn.Parameter(torch.randn(d, self.d_model, device=device) * output_std)
        else:
            self.register_parameter("W_o", None)

    def _attention_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tokenwise outputs and unnormalized linear-attention scores."""
        _, seq_len, _ = x.shape

        # Fan-in normalization is part of the parameter initialization.  The
        # previous additional 1/sqrt(d) factors caused signal loss at width.
        Q = x @ self.W_q.T  # (batch, seq_len, d_model)
        K = x @ self.W_k.T  # (batch, seq_len, d_model)
        V = x @ self.W_v.T  # (batch, seq_len, d_model)

        # Linear attention: A_i = Σ_j (q_i^T k_j) v_j
        # (batch, seq_len, d_model) @ (batch, d_model, seq_len) @ (batch, seq_len, d_model)
        # = (batch, seq_len, d_model)

        # Efficient computation: (Q @ K^T) @ V
        attn_scores = Q @ K.transpose(-2, -1)  # (batch, seq_len, seq_len)

        if self.normalize:
            attn_scores = attn_scores / seq_len

        A = attn_scores @ V  # (batch, seq_len, d_model)

        # Output projection
        if self.use_output_proj:
            output = A @ self.W_o.T  # (batch, seq_len, d)
        else:
            output = A

        return output, attn_scores

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """Forward pass through linear self-attention with mean pooling."""
        output, attn_scores = self._attention_sequence(x)
        if return_attention:
            return output, attn_scores

        return output.mean(dim=1)  # (batch, d) - mean pool for scalar output

    def forward_icl(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for ICL linear regression.

        Constructs input sequence from (context_x, context_y) pairs and query.

        Args:
            context_x: (batch, context_len, d)
            context_y: (batch, context_len)
            query_x: (batch, d)

        Returns:
            prediction: (batch,) predicted y for query

        """
        batch_size = context_x.shape[0]
        context_len = context_x.shape[1]

        if context_y.shape != (batch_size, context_len):
            raise ValueError("context_y must have shape (batch, context_len)")
        if query_x.shape != (batch_size, self.d):
            raise ValueError("query_x must have shape (batch, d)")

        # A label and query marker are embedded into the model's input space;
        # unlike the previous augmented-then-sliced path, no context label is
        # discarded before attention is applied.
        label_embedding = self.context_label_embedding.to(dtype=context_x.dtype)
        query_embedding = self.query_marker_embedding.to(dtype=context_x.dtype)
        context = context_x + context_y.unsqueeze(-1) * label_embedding
        query = query_x + query_embedding
        sequence = torch.cat([context, query.unsqueeze(1)], dim=1)
        output, _ = self._attention_sequence(sequence)
        return output[:, -1, 0]

    def get_weight_vector(self) -> torch.Tensor:
        """Return flattened weights."""
        params = [
            self.W_q.flatten(),
            self.W_k.flatten(),
            self.W_v.flatten(),
            self.context_label_embedding,
            self.query_marker_embedding,
        ]
        if self.W_o is not None:
            params.append(self.W_o.flatten())
        return torch.cat(params)

    def compute_order_params(self, teacher_params: dict | None = None) -> dict[str, float]:
        """Compute order parameters."""
        W = self.get_weight_vector()
        q = (W @ W).item() / (W.numel())
        return {"q": q}

    def __repr__(self) -> str:
        return (
            f"LinearSelfAttention(d={self.d}, d_model={self.d_model}, "
            f"normalize={self.normalize})"
        )


class StateSpaceModel(BaseModel):
    """
    State Space Model (SSM) for sequence modeling.

    Discrete-time linear state space model:
        h_{t+1} = A h_t + B u_t
        y_t = C h_t + D u_t

    Can learn in-context by gradient descent when properly structured.

    Args:
        d: Input dimension
        state_dim: Hidden state dimension
        init_scale: Initialization scale
        diagonal_A: Use diagonal A matrix (more stable)
        dt_min: Minimum discretization step
        dt_max: Maximum discretization step

    """

    def __init__(
        self,
        d: int,
        state_dim: int = 64,
        init_scale: float = 1.0,
        diagonal_A: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__(d=d)
        self.state_dim = state_dim
        self.init_scale = init_scale
        self.diagonal_A = diagonal_A
        self._device = device

        # State transition matrix A
        if diagonal_A:
            # HiPPO-inspired initialization for stable dynamics
            # A_ii = -1/2 gives stable exponential decay
            self.A_diag = nn.Parameter(-0.5 + 0.1 * torch.randn(state_dim, device=device))
        else:
            std = init_scale / np.sqrt(state_dim)
            self.A = nn.Parameter(torch.randn(state_dim, state_dim, device=device) * std)

        # Input matrix B
        std_b = init_scale / np.sqrt(d)
        self.B = nn.Parameter(torch.randn(state_dim, d, device=device) * std_b)

        # Output matrix C
        std_c = init_scale / np.sqrt(state_dim)
        self.C = nn.Parameter(torch.randn(d, state_dim, device=device) * std_c)

        # Direct feedthrough D (optional)
        self.D = nn.Parameter(torch.zeros(d, d, device=device))

        # Discretization step (learnable)
        self.log_dt = nn.Parameter(
            torch.log(torch.tensor([dt_min + (dt_max - dt_min) * 0.5], device=device))
        )

    def _get_A(self) -> torch.Tensor:
        """Get state transition matrix."""
        if self.diagonal_A:
            return torch.diag(self.A_diag)
        else:
            return self.A

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SSM.

        Args:
            u: (batch, seq_len, d) input sequence

        Returns:
            y: (batch, seq_len, d) output sequence or (batch, d) final output

        """
        batch_size, seq_len, d = u.shape
        device = u.device
        dt = torch.exp(self.log_dt)

        # Forward and convolution implementations use the same Euler
        # discretization.  This keeps the parallel method an exact algebraic
        # reformulation of the recurrent one.
        A = self._get_A()
        A_d = torch.eye(self.state_dim, device=device) + A * dt
        B_d = self.B * dt  # B_d ≈ B * dt

        # Initialize state
        h = torch.zeros(batch_size, self.state_dim, device=device)

        outputs = []
        for t in range(seq_len):
            # State update: h_{t+1} = A_d h_t + B_d u_t
            h = h @ A_d.T + u[:, t] @ B_d.T / np.sqrt(d)

            # Output: y_t = C h_t + D u_t
            y_t = h @ self.C.T / np.sqrt(self.state_dim) + u[:, t] @ self.D.T / np.sqrt(d)
            outputs.append(y_t)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, d)

        return outputs[:, -1]  # Return final output

    def forward_parallel(self, u: torch.Tensor) -> torch.Tensor:
        """
        Parallel forward pass using convolution (for efficiency).

        This implements the same computation as forward() but using
        the convolution form of SSM.
        """
        batch_size, seq_len, d = u.shape
        device = u.device
        dt = torch.exp(self.log_dt)

        A = self._get_A()
        A_d = torch.eye(self.state_dim, device=device) + A * dt
        B_d = self.B * dt

        # Build convolution kernel for h_{t+1}=A_d h_t+B_d u_t:
        # K[t] = C A_d^t B_d.
        kernel = torch.zeros(seq_len, d, d, device=device)
        A_power = torch.eye(self.state_dim, device=device)

        for t in range(seq_len):
            kernel[t] = self.C @ A_power @ B_d / np.sqrt(self.state_dim * d)
            A_power = A_power @ A_d

        # Convolve: y = K * u
        # Simple implementation (can be made faster with FFT)
        outputs = torch.zeros(batch_size, seq_len, d, device=device)
        for t in range(seq_len):
            for s in range(t + 1):
                outputs[:, t] += u[:, s] @ kernel[t - s].T

        # Add feedthrough
        outputs = outputs + u @ self.D.T / np.sqrt(d)

        return outputs[:, -1]

    def get_weight_vector(self) -> torch.Tensor:
        params = [self.B.flatten(), self.C.flatten(), self.D.flatten()]
        if self.diagonal_A:
            params.append(self.A_diag)
        else:
            params.append(self.A.flatten())
        return torch.cat(params)

    def compute_order_params(self, teacher_params: dict | None = None) -> dict[str, float]:
        W = self.get_weight_vector()
        q = (W @ W).item() / W.numel()
        return {"q": q}

    def __repr__(self) -> str:
        return (
            f"StateSpaceModel(d={self.d}, state_dim={self.state_dim}, "
            f"diagonal_A={self.diagonal_A})"
        )


class LinearRNN(BaseModel):
    """
    Linear Recurrent Neural Network.

    Minimal RNN with linear dynamics:
        h_{t+1} = W_h h_t + W_x x_t / √d
        y_t = W_y h_t / √state_dim

    Useful for analyzing learning dynamics in sequence models.

    Args:
        d: Input dimension
        state_dim: Hidden state dimension
        output_dim: Output dimension (default: 1)
        init_scale: Initialization scale
        spectral_norm: Normalize W_h to have spectral radius < 1

    """

    def __init__(
        self,
        d: int,
        state_dim: int = 64,
        output_dim: int = 1,
        init_scale: float = 1.0,
        spectral_norm: bool = True,
        device: str = "cpu",
    ):
        super().__init__(d=d)
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.init_scale = init_scale
        self.spectral_norm = spectral_norm

        # Recurrent weight
        std_h = init_scale / np.sqrt(state_dim)
        self.W_h = nn.Parameter(torch.randn(state_dim, state_dim, device=device) * std_h)

        # Input weight
        std_x = init_scale / np.sqrt(d)
        self.W_x = nn.Parameter(torch.randn(state_dim, d, device=device) * std_x)

        # Output weight
        std_y = init_scale / np.sqrt(state_dim)
        self.W_y = nn.Parameter(torch.randn(output_dim, state_dim, device=device) * std_y)

    def _get_W_h(self) -> torch.Tensor:
        """Get recurrent weight, possibly normalized."""
        if self.spectral_norm:
            # Normalize to have spectral radius < 1
            with torch.no_grad():
                u, s, v = torch.svd(self.W_h)
                max_sv = s.max()
            if max_sv > 0.99:
                return self.W_h * (0.99 / max_sv)
        return self.W_h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RNN.

        Args:
            x: (batch, seq_len, d) input sequence

        Returns:
            y: (batch, output_dim) final output

        """
        batch_size, seq_len, d = x.shape
        device = x.device

        W_h = self._get_W_h()

        # Initialize hidden state
        h = torch.zeros(batch_size, self.state_dim, device=device)

        for t in range(seq_len):
            # h_{t+1} = W_h h_t + W_x x_t / √d
            h = h @ W_h.T + x[:, t] @ self.W_x.T / np.sqrt(d)

        # Output: y = W_y h / √state_dim
        y = h @ self.W_y.T / np.sqrt(self.state_dim)

        return y.squeeze(-1) if self.output_dim == 1 else y

    def get_weight_vector(self) -> torch.Tensor:
        return torch.cat([self.W_h.flatten(), self.W_x.flatten(), self.W_y.flatten()])

    def compute_order_params(self, teacher_params: dict | None = None) -> dict[str, float]:
        W = self.get_weight_vector()
        q = (W @ W).item() / W.numel()
        return {"q": q}

    def __repr__(self) -> str:
        return (
            f"LinearRNN(d={self.d}, state_dim={self.state_dim}, " f"output_dim={self.output_dim})"
        )


class ModernHopfieldNetwork(BaseModel):
    """
    Modern Hopfield Network with continuous states.

    Energy function:
        E(x) = -1/β log Σ_μ exp(β x^T ξ^μ) + 1/2 ||x||²

    Update rule (gradient descent on energy):
        x_{new} = Σ_μ softmax(β x^T ξ^μ) ξ^μ

    This is equivalent to one step of attention with patterns as KV.

    Args:
        d: Pattern dimension
        n_patterns: Number of stored patterns (M)
        beta: Inverse temperature (sharpness of retrieval)
        n_iterations: Number of update iterations
        init_scale: Pattern initialization scale

    """

    def __init__(
        self,
        d: int,
        n_patterns: int = 100,
        beta: float = 1.0,
        n_iterations: int = 1,
        init_scale: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__(d=d)
        self.n_patterns = n_patterns
        self.beta = beta
        self.n_iterations = n_iterations
        self.init_scale = init_scale

        # Stored patterns ξ^μ
        std = init_scale / np.sqrt(d)
        self.patterns = nn.Parameter(torch.randn(n_patterns, d, device=device) * std)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Hopfield energy.

        Args:
            x: (batch, d) query states

        Returns:
            E: (batch,) energy values

        """
        # x^T ξ^μ / √d
        similarities = x @ self.patterns.T / np.sqrt(self.d)  # (batch, n_patterns)

        # -1/β log Σ_μ exp(β s_μ)
        lse = torch.logsumexp(self.beta * similarities, dim=-1) / self.beta

        # + 1/2 ||x||² / d
        norm_sq = (x**2).sum(dim=-1) / self.d / 2

        return -lse + norm_sq

    def update(self, x: torch.Tensor) -> torch.Tensor:
        """
        One Hopfield update step.

        Args:
            x: (batch, d) current states

        Returns:
            x_new: (batch, d) updated states

        """
        # Attention weights: softmax(β x^T ξ^μ / √d)
        similarities = x @ self.patterns.T / np.sqrt(self.d)  # (batch, n_patterns)
        attn = F.softmax(self.beta * similarities, dim=-1)  # (batch, n_patterns)

        # Weighted sum of patterns
        x_new = attn @ self.patterns  # (batch, d)

        return x_new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: retrieve pattern from query.

        Args:
            x: (batch, d) or (batch, seq_len, d) query states

        Returns:
            output: (batch, d) or (batch,) retrieved patterns

        """
        if x.dim() == 3:
            # Sequence input: use mean as query
            x = x.mean(dim=1)

        # Iterative retrieval
        for _ in range(self.n_iterations):
            x = self.update(x)

        # Return norm (for scalar output compatibility)
        return x.norm(dim=-1) / np.sqrt(self.d)

    def retrieve(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve pattern and return attention weights.

        Returns:
            retrieved: (batch, d) retrieved pattern
            attention: (batch, n_patterns) attention weights

        """
        similarities = x @ self.patterns.T / np.sqrt(self.d)
        attention = F.softmax(self.beta * similarities, dim=-1)
        retrieved = attention @ self.patterns
        return retrieved, attention

    def get_weight_vector(self) -> torch.Tensor:
        return self.patterns.flatten()

    def compute_order_params(self, teacher_params: dict | None = None) -> dict[str, float]:
        P = self.patterns
        q = (P.flatten() @ P.flatten()).item() / P.numel()
        return {"q": q}

    def capacity(self) -> float:
        """
        Theoretical storage capacity.

        For Modern Hopfield: M ~ d (exponential in classical Hopfield)
        """
        return self.n_patterns / self.d

    def __repr__(self) -> str:
        return (
            f"ModernHopfieldNetwork(d={self.d}, n_patterns={self.n_patterns}, "
            f"beta={self.beta}, n_iterations={self.n_iterations})"
        )
