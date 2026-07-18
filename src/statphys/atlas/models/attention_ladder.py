"""Instrumented M0--M8 attention architecture ladder.

The first two stages are the low-rank student models used to study semantic
and positional specialization.  Later stages add standard transformer
ingredients one at a time while preserving a common ``(B, T, d)`` interface
and a common set of diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, overload
from collections.abc import Mapping

import math
import torch
from torch import Tensor, nn


StageName = Literal["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]
NormName = Literal["pre", "post"]
InitializationStrategy = Literal["random", "semantic", "positional", "mixed"]


def _stage_number(stage: str) -> int:
    normalized = stage.strip().lower()
    if normalized not in {f"m{index}" for index in range(9)}:
        raise ValueError("stage must be one of m0, m1, ..., m8")
    return int(normalized[1:])


def _is_floating_dtype(dtype: torch.dtype) -> bool:
    return isinstance(dtype, torch.dtype) and torch.empty((), dtype=dtype).is_floating_point()


@dataclass(frozen=True)
class AttentionLadderConfig:
    """Configuration shared by all stages of the attention ladder.

    Stage semantics are cumulative:

    ``M0`` tied low-rank Q/K, ``M1`` untied low-rank Q/K, ``M2`` full
    single-head Q/K/V/O, ``M3`` multi-head attention, ``M4`` residual paths
    and pre/post LayerNorm, ``M5`` an MLP, ``M6`` causal masking and RoPE, and
    ``M7`` configurable depth, and ``M8`` a pointwise autoregressive readout.
    """

    stage: str
    d_model: int
    seq_len: int
    signal_rank: int = 1
    n_heads: int = 1
    head_dim: int | None = None
    n_layers: int = 1
    ffn_dim: int | None = None
    norm: str = "pre"
    attention_temperature: float = 1.0
    init_seed: int = 2
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        number = _stage_number(self.stage)
        object.__setattr__(self, "stage", f"m{number}")
        if self.d_model < 1:
            raise ValueError("d_model must be positive")
        if self.seq_len < 1:
            raise ValueError("seq_len must be positive")
        if not 1 <= self.signal_rank <= self.d_model:
            raise ValueError("signal_rank must lie in [1, d_model]")
        if self.n_heads < 1:
            raise ValueError("n_heads must be positive")
        if number <= 2 and self.n_heads != 1:
            raise ValueError(f"{self.stage} is single-head; set n_heads=1")
        if self.n_layers < 1:
            raise ValueError("n_layers must be positive")
        if number < 7 and self.n_layers != 1:
            raise ValueError("configurable depth is introduced at m7; use n_layers=1")
        if self.head_dim is None:
            if number >= 3:
                if self.d_model % self.n_heads:
                    raise ValueError(
                        "d_model must be divisible by n_heads when head_dim is omitted"
                    )
                object.__setattr__(self, "head_dim", self.d_model // self.n_heads)
            else:
                object.__setattr__(self, "head_dim", self.d_model)
        elif self.head_dim < 1:
            raise ValueError("head_dim must be positive")
        if self.ffn_dim is None:
            object.__setattr__(self, "ffn_dim", 4 * self.d_model)
        elif self.ffn_dim < 1:
            raise ValueError("ffn_dim must be positive")
        normalized_norm = self.norm.strip().lower()
        if normalized_norm not in {"pre", "post"}:
            raise ValueError("norm must be 'pre' or 'post'")
        object.__setattr__(self, "norm", normalized_norm)
        if self.attention_temperature <= 0 or not math.isfinite(self.attention_temperature):
            raise ValueError("attention_temperature must be finite and positive")
        if not isinstance(self.init_seed, int) or self.init_seed < 0:
            raise ValueError("init_seed must be a non-negative integer")
        if not _is_floating_dtype(self.dtype):
            raise TypeError(f"dtype must be a floating torch.dtype, got {self.dtype!r}")
        torch.device(self.device)

    @property
    def stage_number(self) -> int:
        """Integer stage index in ``[0, 8]``."""
        return _stage_number(self.stage)

    @property
    def effective_n_layers(self) -> int:
        """Number of blocks instantiated by this stage."""
        return self.n_layers if self.stage_number >= 7 else 1

    @property
    def effective_n_heads(self) -> int:
        """Number of learned attention heads at this stage."""
        return self.n_heads if self.stage_number >= 3 else 1

    @property
    def is_autoregressive(self) -> bool:
        """Whether this stage emits next-token predictions through a readout."""
        return self.stage_number == 8

    @property
    def causal_target_shift(self) -> int:
        """Target offset expected by the autoregressive loss runner."""
        return 1 if self.is_autoregressive else 0


def _uniform_parameter_(parameter: Tensor, generator: torch.Generator) -> None:
    fan_out, fan_in = parameter.shape
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    values = torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu")
    values.uniform_(-bound, bound, generator=generator)
    with torch.no_grad():
        parameter.copy_(values.to(device=parameter.device))


def _normal_parameter_(
    parameter: Tensor,
    generator: torch.Generator,
    *,
    std: float = 1.0,
) -> None:
    values = torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu")
    values.normal_(mean=0.0, std=std, generator=generator)
    with torch.no_grad():
        parameter.copy_(values.to(device=parameter.device))


class _AttentionBlock(nn.Module):
    """One ladder block; feature gates are derived only from the stage."""

    def __init__(self, config: AttentionLadderConfig) -> None:
        super().__init__()
        self.config = config
        self.stage_number = config.stage_number
        self.n_heads = config.effective_n_heads
        self.head_dim = int(config.head_dim)
        self.low_rank = self.stage_number <= 1
        self.tied = self.stage_number == 0
        self.use_residual = self.stage_number >= 4
        self.use_mlp = self.stage_number >= 5
        self.use_rope = self.stage_number >= 6
        self.use_causal_mask = self.stage_number >= 6

        if self.low_rank:
            shape = (1, config.d_model, config.signal_rank)
            self.low_rank_q = nn.Parameter(torch.empty(shape, dtype=config.dtype))
            if self.tied:
                self.register_parameter("low_rank_k", None)
            else:
                self.low_rank_k = nn.Parameter(torch.empty(shape, dtype=config.dtype))
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.o_proj = None
        else:
            projection_width = self.n_heads * self.head_dim
            self.register_parameter("low_rank_q", None)
            self.register_parameter("low_rank_k", None)
            self.q_proj = nn.Linear(config.d_model, projection_width, bias=False)
            self.k_proj = nn.Linear(config.d_model, projection_width, bias=False)
            self.v_proj = nn.Linear(config.d_model, projection_width, bias=False)
            self.o_proj = nn.Linear(projection_width, config.d_model, bias=False)

        if self.use_residual:
            self.norm1 = nn.LayerNorm(config.d_model)
            self.norm2 = nn.LayerNorm(config.d_model) if self.use_mlp else None
        else:
            self.norm1 = None
            self.norm2 = None
        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(config.d_model, int(config.ffn_dim)),
                nn.GELU(),
                nn.Linear(int(config.ffn_dim), config.d_model),
            )
        else:
            self.mlp = None

    @property
    def low_rank_k_effective(self) -> Tensor:
        if self.low_rank_q is None:
            raise RuntimeError("effective low-rank factors only exist for m0 and m1")
        return self.low_rank_q if self.tied else self.low_rank_k

    def reset_parameters(self, generator: torch.Generator) -> None:
        """Reset every stochastic parameter using only ``generator``."""
        if self.low_rank:
            assert self.low_rank_q is not None
            _normal_parameter_(self.low_rank_q, generator)
            if not self.tied:
                assert self.low_rank_k is not None
                _normal_parameter_(self.low_rank_k, generator)
        else:
            assert self.q_proj is not None
            assert self.k_proj is not None
            assert self.v_proj is not None
            assert self.o_proj is not None
            for projection in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
                _uniform_parameter_(projection.weight, generator)
        if self.norm1 is not None:
            nn.init.ones_(self.norm1.weight)
            nn.init.zeros_(self.norm1.bias)
        if self.norm2 is not None:
            nn.init.ones_(self.norm2.weight)
            nn.init.zeros_(self.norm2.bias)
        if self.mlp is not None:
            first = self.mlp[0]
            second = self.mlp[2]
            assert isinstance(first, nn.Linear) and isinstance(second, nn.Linear)
            _uniform_parameter_(first.weight, generator)
            _uniform_parameter_(second.weight, generator)
            nn.init.zeros_(first.bias)
            nn.init.zeros_(second.bias)

    @staticmethod
    def _apply_rope(tensor: Tensor) -> Tensor:
        """Apply rotary embeddings to ``(B, H, T, D_h)`` Q or K."""
        _, _, length, width = tensor.shape
        rotary_width = width - (width % 2)
        if rotary_width == 0:
            return tensor
        rotary = tensor[..., :rotary_width]
        frequency_dtype = (
            torch.float32 if tensor.dtype in {torch.float16, torch.bfloat16} else tensor.dtype
        )
        positions = torch.arange(length, device=tensor.device, dtype=frequency_dtype)
        indices = torch.arange(0, rotary_width, 2, device=tensor.device, dtype=frequency_dtype)
        inverse_frequency = 1.0 / (10000.0 ** (indices / rotary_width))
        angles = torch.outer(positions, inverse_frequency)
        cosine = angles.cos().to(dtype=tensor.dtype)[None, None, :, :]
        sine = angles.sin().to(dtype=tensor.dtype)[None, None, :, :]
        even, odd = rotary[..., 0::2], rotary[..., 1::2]
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine),
            dim=-1,
        ).flatten(start_dim=-2)
        if rotary_width == width:
            return rotated
        return torch.cat((rotated, tensor[..., rotary_width:]), dim=-1)

    def _attention(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, width = inputs.shape
        if self.low_rank:
            assert self.low_rank_q is not None
            q = torch.einsum("btd,hdr->bhtr", inputs, self.low_rank_q)
            k = torch.einsum("btd,hdr->bhtr", inputs, self.low_rank_k_effective)
            logits = torch.einsum("bhtr,bhsr->bhts", q, k)
            logits = logits / (width * self.config.attention_temperature)
            attention = torch.softmax(logits, dim=-1)
            output = torch.einsum("bhts,bsd->bhtd", attention, inputs).squeeze(1)
            return output, attention

        assert self.q_proj is not None
        assert self.k_proj is not None
        assert self.v_proj is not None
        assert self.o_proj is not None
        q = self.q_proj(inputs).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(inputs).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(inputs).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            q = self._apply_rope(q)
            k = self._apply_rope(k)
        logits = torch.matmul(q, k.transpose(-2, -1))
        logits = logits / (math.sqrt(self.head_dim) * self.config.attention_temperature)
        if self.use_causal_mask:
            causal = torch.ones(
                (length, length),
                device=inputs.device,
                dtype=torch.bool,
            ).triu(diagonal=1)
            logits = logits.masked_fill(causal, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.o_proj(context), attention

    def forward(
        self,
        inputs: Tensor,
        *,
        ablate_attention: bool,
        ablate_mlp: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return output, attention map, raw attention branch and raw MLP branch."""
        if not self.use_residual:
            attention_output, attention = self._attention(inputs)
            applied = torch.zeros_like(attention_output) if ablate_attention else attention_output
            return applied, attention, attention_output, torch.zeros_like(applied)

        assert self.norm1 is not None
        if self.config.norm == "pre":
            attention_output, attention = self._attention(self.norm1(inputs))
            applied_attention = (
                torch.zeros_like(attention_output) if ablate_attention else attention_output
            )
            hidden = inputs + applied_attention
            if self.mlp is None:
                return hidden, attention, attention_output, torch.zeros_like(hidden)
            assert self.norm2 is not None
            mlp_output = self.mlp(self.norm2(hidden))
            applied_mlp = torch.zeros_like(mlp_output) if ablate_mlp else mlp_output
            return hidden + applied_mlp, attention, attention_output, mlp_output

        attention_output, attention = self._attention(inputs)
        applied_attention = (
            torch.zeros_like(attention_output) if ablate_attention else attention_output
        )
        hidden = self.norm1(inputs + applied_attention)
        if self.mlp is None:
            return hidden, attention, attention_output, torch.zeros_like(hidden)
        assert self.norm2 is not None
        mlp_output = self.mlp(hidden)
        applied_mlp = torch.zeros_like(mlp_output) if ablate_mlp else mlp_output
        return self.norm2(hidden + applied_mlp), attention, attention_output, mlp_output

    def effective_qk(self) -> Tensor:
        """Return unscaled per-head effective QK matrices ``(H, d, d)``."""
        if self.low_rank:
            assert self.low_rank_q is not None
            return torch.einsum(
                "hdr,her->hde",
                self.low_rank_q,
                self.low_rank_k_effective,
            )
        assert self.q_proj is not None and self.k_proj is not None
        q_weight = self.q_proj.weight.view(self.n_heads, self.head_dim, -1)
        k_weight = self.k_proj.weight.view(self.n_heads, self.head_dim, -1)
        return torch.einsum("hkd,hke->hde", q_weight, k_weight)

    def effective_ov(self) -> Tensor:
        """Return per-head effective OV matrices ``(H, d, d)``."""
        if self.low_rank:
            identity = torch.eye(
                self.config.d_model,
                device=self.low_rank_q.device,
                dtype=self.low_rank_q.dtype,
            )
            return identity.unsqueeze(0)
        assert self.v_proj is not None and self.o_proj is not None
        value_weight = self.v_proj.weight.view(self.n_heads, self.head_dim, -1)
        output_weight = self.o_proj.weight.view(-1, self.n_heads, self.head_dim).permute(1, 0, 2)
        return torch.einsum("hdk,hke->hde", output_weight, value_weight)


class InstrumentedAttentionModel(nn.Module):
    """A stage-configurable transformer with phase-analysis diagnostics.

    ``forward(x)`` returns a tensor of shape ``(B,T,d)``.  With
    ``return_diagnostics=True`` it instead returns ``(output, diagnostics)``;
    attention maps have shape ``(B, layers, heads, T, T)`` and representation
    trajectories have shape ``(B, layers + 1, T, d)``.
    """

    def __init__(self, config: AttentionLadderConfig) -> None:
        super().__init__()
        self.config = config

        # nn.Linear constructors initialize eagerly.  Preserve the caller's
        # global RNG state, then overwrite every stochastic parameter below.
        cpu_rng_state = torch.random.get_rng_state()
        try:
            self.blocks = nn.ModuleList(
                [_AttentionBlock(config) for _ in range(config.effective_n_layers)]
            )
            self.autoregressive_readout = (
                nn.Linear(config.d_model, config.d_model, bias=False)
                if config.is_autoregressive
                else None
            )
        finally:
            torch.random.set_rng_state(cpu_rng_state)
        self.reset_parameters(config.init_seed)
        self.to(device=torch.device(config.device), dtype=config.dtype)

    def reset_parameters(self, seed: int | None = None) -> None:
        """Deterministically reset parameters without touching global RNG state."""
        actual_seed = self.config.init_seed if seed is None else seed
        if not isinstance(actual_seed, int) or actual_seed < 0:
            raise ValueError("seed must be a non-negative integer")
        generator = torch.Generator(device="cpu").manual_seed(actual_seed)
        for block in self.blocks:
            block.reset_parameters(generator)
        if self.autoregressive_readout is not None:
            _uniform_parameter_(self.autoregressive_readout.weight, generator)

    def _validate_inputs(self, inputs: Tensor) -> None:
        if not isinstance(inputs, Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, sequence, d_model)")
        if inputs.shape[-1] != self.config.d_model:
            raise ValueError(
                f"expected final dimension {self.config.d_model}, got {inputs.shape[-1]}"
            )
        if inputs.shape[1] > self.config.seq_len:
            raise ValueError(
                f"sequence length {inputs.shape[1]} exceeds configured {self.config.seq_len}"
            )
        if not inputs.is_floating_point():
            raise TypeError("inputs must use a floating dtype")
        parameter = next(self.parameters())
        if inputs.device != parameter.device:
            raise ValueError(
                f"inputs are on {inputs.device}, but model parameters are on {parameter.device}"
            )

    @overload
    def forward(
        self,
        inputs: Tensor,
        *,
        return_diagnostics: Literal[False] = False,
        ablate_attention: bool = False,
        ablate_mlp: bool = False,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        inputs: Tensor,
        *,
        return_diagnostics: Literal[True],
        ablate_attention: bool = False,
        ablate_mlp: bool = False,
    ) -> tuple[Tensor, Mapping[str, Any]]: ...

    def forward(
        self,
        inputs: Tensor,
        *,
        return_diagnostics: bool = False,
        ablate_attention: bool = False,
        ablate_mlp: bool = False,
    ) -> Tensor | tuple[Tensor, Mapping[str, Any]]:
        """Run the model, optionally zeroing complete attention/MLP branches."""
        self._validate_inputs(inputs)
        hidden = inputs
        representations = [hidden]
        attention_maps = []
        attention_outputs = []
        mlp_outputs = []
        for block in self.blocks:
            hidden, attention, attention_output, mlp_output = block(
                hidden,
                ablate_attention=ablate_attention,
                ablate_mlp=ablate_mlp,
            )
            representations.append(hidden)
            attention_maps.append(attention)
            attention_outputs.append(attention_output)
            mlp_outputs.append(mlp_output)
        pre_readout = hidden
        if self.autoregressive_readout is not None:
            hidden = self.autoregressive_readout(hidden)
        if not return_diagnostics:
            return hidden
        diagnostics: dict[str, Any] = {
            "attention_maps": torch.stack(attention_maps, dim=1),
            "representations": torch.stack(representations, dim=1),
            "attention_outputs": torch.stack(attention_outputs, dim=1),
            "mlp_outputs": torch.stack(mlp_outputs, dim=1),
            "pre_readout": pre_readout,
            "is_autoregressive": self.config.is_autoregressive,
            "effective_qk": self.effective_qk_matrices(),
            "effective_ov": self.effective_ov_matrices(),
            "ablate_attention": ablate_attention,
            "ablate_mlp": ablate_mlp,
        }
        return hidden, diagnostics

    def shifted_autoregressive_pairs(
        self,
        predictions: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Align M8 predictions at ``t`` with next-token targets at ``t+1``."""
        if not self.config.is_autoregressive:
            raise RuntimeError("shifted autoregressive pairs are available only for m8")
        if predictions.ndim != 3 or targets.ndim != 3:
            raise ValueError("predictions and targets must have shape (B, T, d)")
        if predictions.shape != targets.shape:
            raise ValueError("predictions and targets must have identical shapes")
        if predictions.shape[1] < 2:
            raise ValueError("autoregressive loss requires a sequence of length at least two")
        return predictions[:, :-1], targets[:, 1:]

    def effective_qk_matrices(self, *, detach: bool = False) -> Tensor:
        """Return ``(layers, heads, d, d)`` unscaled effective QK matrices."""
        matrices = torch.stack([block.effective_qk() for block in self.blocks], dim=0)
        return matrices.detach() if detach else matrices

    def effective_ov_matrices(self, *, detach: bool = False) -> Tensor:
        """Return ``(layers, heads, d, d)`` effective OV matrices."""
        matrices = torch.stack([block.effective_ov() for block in self.blocks], dim=0)
        return matrices.detach() if detach else matrices

    def _coerce_semantic_directions(self, directions: Tensor | Any) -> Tensor:
        value = torch.as_tensor(
            directions,
            device=self.blocks[0].low_rank_q.device,
            dtype=self.blocks[0].low_rank_q.dtype,
        )
        width, rank = self.config.d_model, self.config.signal_rank
        if value.ndim == 1:
            if value.numel() != width:
                raise ValueError(f"semantic direction must have length {width}")
            value = value.unsqueeze(0)
        if value.ndim != 2:
            raise ValueError("semantic_directions must be one- or two-dimensional")
        if value.shape[-1] == width:
            rows = value
        elif value.shape[0] == width:
            rows = value.transpose(0, 1)
        else:
            raise ValueError(
                f"semantic_directions must have one dimension equal to d_model={width}"
            )
        if rows.shape[0] < rank:
            raise ValueError(f"at least {rank} semantic directions are required")
        result = rows[:rank].transpose(0, 1).contiguous()
        if not bool(torch.isfinite(result).all()):
            raise ValueError("semantic_directions must be finite")
        return result

    def _coerce_positional_directions(self, directions: Tensor | Any) -> Tensor:
        value = torch.as_tensor(
            directions,
            device=self.blocks[0].low_rank_q.device,
            dtype=self.blocks[0].low_rank_q.dtype,
        )
        width, rank = self.config.d_model, self.config.signal_rank
        if value.ndim == 1:
            if value.numel() != width:
                raise ValueError(f"positional direction must have length {width}")
            value = value.unsqueeze(0)
        if value.ndim != 2:
            raise ValueError("positional_directions must be one- or two-dimensional")
        if value.shape[-1] == width:
            rows = value
        elif value.shape[0] == width:
            rows = value.transpose(0, 1)
        else:
            raise ValueError(
                f"positional_directions must have one dimension equal to d_model={width}"
            )
        centered = rows - rows.mean(dim=0, keepdim=True) if rows.shape[0] > 1 else rows
        analysis = (
            centered.float() if centered.dtype in {torch.float16, torch.bfloat16} else centered
        )
        _, singular_values, vh = torch.linalg.svd(analysis, full_matrices=False)
        tolerance = torch.finfo(analysis.dtype).eps * max(analysis.shape) * singular_values.max()
        available = int((singular_values > tolerance).sum().item())
        if available < rank:
            raise ValueError(
                f"positional encodings span only {available} directions; {rank} required"
            )
        mean_norm = rows.to(dtype=analysis.dtype).norm(dim=-1).mean()
        result = (vh[:rank].transpose(0, 1).contiguous() * mean_norm).to(dtype=rows.dtype)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("positional_directions must be finite")
        return result

    def initialize_from_directions(
        self,
        semantic_directions: Tensor | Any | None = None,
        positional_directions: Tensor | Any | None = None,
        strategy: InitializationStrategy = "random",
        noise_scale: float = 0.0,
    ) -> InstrumentedAttentionModel:
        """Initialize M0/M1 factors in a controlled semantic/positional basin.

        ``semantic`` copies planted semantic vectors exactly when
        ``noise_scale=0``.  In the exact paper bridge the teacher still sees
        raw tokens while the student sees position-shifted inputs.
        ``positional`` uses the leading centered positional directions, and
        ``mixed`` averages both factor sets.  Full projection stages M2--M8
        support only deterministic random reset.
        """
        normalized = strategy.strip().lower()
        if normalized not in {"random", "semantic", "positional", "mixed"}:
            raise ValueError("strategy must be random, semantic, positional, or mixed")
        if noise_scale < 0 or not math.isfinite(noise_scale):
            raise ValueError("noise_scale must be finite and non-negative")
        if self.config.stage_number >= 2:
            if normalized != "random":
                raise ValueError("directional initialization is defined only for m0 and m1")
            self.reset_parameters(self.config.init_seed)
            return self

        generator = torch.Generator(device="cpu").manual_seed(self.config.init_seed)
        semantic: Tensor | None = None
        positional: Tensor | None = None
        if normalized in {"semantic", "mixed"}:
            if semantic_directions is None:
                raise ValueError(
                    f"semantic_directions are required for {normalized} initialization"
                )
            semantic = self._coerce_semantic_directions(semantic_directions)
        if normalized in {"positional", "mixed"}:
            if positional_directions is None:
                raise ValueError(
                    f"positional_directions are required for {normalized} initialization"
                )
            positional = self._coerce_positional_directions(positional_directions)

        block = self.blocks[0]
        assert block.low_rank_q is not None
        target_shape = block.low_rank_q.shape
        if normalized == "random":
            base = torch.empty(target_shape, dtype=block.low_rank_q.dtype, device="cpu")
            base.normal_(generator=generator)
            base = base.to(device=block.low_rank_q.device)
        elif normalized == "semantic":
            assert semantic is not None
            base = semantic.unsqueeze(0)
        elif normalized == "positional":
            assert positional is not None
            base = positional.unsqueeze(0)
        else:
            assert semantic is not None and positional is not None
            base = ((semantic + positional) / math.sqrt(2.0)).unsqueeze(0)

        def noise() -> Tensor:
            if noise_scale == 0:
                return torch.zeros_like(base)
            value = torch.empty(base.shape, dtype=base.dtype, device="cpu")
            value.normal_(mean=0.0, std=noise_scale, generator=generator)
            return value.to(device=base.device)

        with torch.no_grad():
            block.low_rank_q.copy_(base + noise())
            if not block.tied:
                assert block.low_rank_k is not None
                block.low_rank_k.copy_(base + noise())
        return self


def build_attention_ladder(
    config: AttentionLadderConfig | None = None,
    **config_kwargs: Any,
) -> InstrumentedAttentionModel:
    """Build an instrumented ladder model from a config or config keywords."""
    if config is not None and config_kwargs:
        raise TypeError("pass either config or configuration keywords, not both")
    if config is None:
        config = AttentionLadderConfig(**config_kwargs)
    if not isinstance(config, AttentionLadderConfig):
        raise TypeError("config must be an AttentionLadderConfig")
    return InstrumentedAttentionModel(config)


__all__ = [
    "AttentionLadderConfig",
    "InitializationStrategy",
    "InstrumentedAttentionModel",
    "NormName",
    "StageName",
    "build_attention_ladder",
]
