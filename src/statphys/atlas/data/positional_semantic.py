"""Synthetic positional--semantic teacher data for attention experiments.

The module implements the finite-dimensional teacher used in the numerical
experiments accompanying the attention phase-diagram paper.  It also exposes
five distribution shifts with the same API so that changes in the learned
order parameters can be compared without changing training code.

The exact :func:`PositionalSemanticDataConfig.exact_paper_bridge` setting is
the ``L=2, r=1`` experiment of arXiv:2402.03902: token standard deviation
``sigma=0.5``, antipodal positional vectors, and
``A=[[0.6, 0.4], [0.4, 0.6]]``.  Its target is

``((1 - omega) * A_semantic(raw) + omega * A_positional) @ raw``.

Only the student input receives the positional shift.  This distinction is
essential for faithfully reproducing the reference implementation.

No global random-number state is used.  Teacher, quenched data-distribution,
and model-initialization seeds are intentionally separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor


EnsembleName = Literal["d0", "d1", "d2", "d3", "d4", "d5"]

_ENSEMBLE_ALIASES: Mapping[str, EnsembleName] = {
    "d0": "d0",
    "gaussian": "d0",
    "isotropic_gaussian": "d0",
    "d1": "d1",
    "elliptical": "d1",
    "elliptical_gaussian": "d1",
    "d2": "d2",
    "student_t": "d2",
    "student-t": "d2",
    "heavy_tail": "d2",
    "heavy-tailed": "d2",
    "d3": "d3",
    "codebook": "d3",
    "discrete_codebook": "d3",
    "d4": "d4",
    "hmm": "d4",
    "markov": "d4",
    "latent_markov": "d4",
    "d5": "d5",
    "grammar": "d5",
    "pcfg": "d5",
    "probabilistic_grammar": "d5",
    "context_free": "d5",
}


def _canonical_ensemble(name: str) -> EnsembleName:
    try:
        return _ENSEMBLE_ALIASES[name.strip().lower()]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(_ENSEMBLE_ALIASES.values())))
        raise ValueError(f"unknown ensemble {name!r}; expected one of {allowed}") from exc


def _validate_floating_dtype(dtype: torch.dtype) -> None:
    if not isinstance(dtype, torch.dtype) or not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError(f"dtype must be a floating torch.dtype, got {dtype!r}")


@dataclass(frozen=True)
class PositionalSemanticDataConfig:
    """Configuration for a positional--semantic synthetic teacher.

    Parameters
    ----------
    d_model:
        Token dimension ``d``.
    seq_len:
        Sequence length ``L``.
    signal_rank:
        Number of planted semantic directions ``r``.
    ensemble:
        One of D0--D5 (descriptive aliases such as ``"student_t"`` are also
        accepted).
    sigma:
        Standard deviation of the centred token innovation.  D1--D5 are
        normalized to unit average variance before this scale is applied.
    omega:
        Positional weight in the convex teacher mixture.
    attention_temperature:
        Multiplicative denominator applied to semantic logits.  The exact
        paper bridge uses one.
    teacher_seed, data_seed, init_seed:
        Independent seeds for planted directions/positions, quenched
        distribution parameters and model initialization, respectively.
    positional_matrix:
        Optional row-stochastic positional attention matrix.  If omitted,
        diagonal mass 0.6 and total off-diagonal mass 0.4 are used.

    """

    d_model: int
    seq_len: int = 2
    signal_rank: int = 1
    ensemble: str = "d0"
    sigma: float = 0.5
    omega: float = 0.3
    attention_temperature: float = 1.0
    teacher_seed: int = 0
    data_seed: int = 1
    init_seed: int = 2
    positional_matrix: Tensor | np.ndarray | None = None
    positional_scale: float = 1.0
    elliptical_condition_number: float = 8.0
    student_t_df: float = 5.0
    codebook_size: int = 16
    hmm_states: int = 4
    hmm_persistence: float = 0.9
    hmm_emission_noise: float = 0.25
    grammar_nonterminals: int = 4
    grammar_rule_concentration: float = 1.0
    grammar_emission_noise: float = 0.25
    teacher_input_source: str = "raw"
    target_value_source: str = "raw"
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError("d_model must be positive")
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least two")
        if not 1 <= self.signal_rank <= self.d_model:
            raise ValueError("signal_rank must lie in [1, d_model]")
        object.__setattr__(self, "ensemble", _canonical_ensemble(self.ensemble))
        if self.sigma <= 0 or not np.isfinite(self.sigma):
            raise ValueError("sigma must be finite and positive")
        if not 0.0 <= self.omega <= 1.0:
            raise ValueError("omega must lie in [0, 1]")
        if self.attention_temperature <= 0 or not np.isfinite(self.attention_temperature):
            raise ValueError("attention_temperature must be finite and positive")
        if self.positional_scale < 0 or not np.isfinite(self.positional_scale):
            raise ValueError("positional_scale must be finite and non-negative")
        if self.elliptical_condition_number < 1:
            raise ValueError("elliptical_condition_number must be at least one")
        if self.student_t_df <= 2:
            raise ValueError("student_t_df must exceed two for finite variance")
        if self.codebook_size < 2:
            raise ValueError("codebook_size must be at least two")
        if self.hmm_states < 2:
            raise ValueError("hmm_states must be at least two")
        if not 0 <= self.hmm_persistence <= 1:
            raise ValueError("hmm_persistence must lie in [0, 1]")
        if not 0 <= self.hmm_emission_noise < 1:
            raise ValueError("hmm_emission_noise must lie in [0, 1)")
        if self.grammar_nonterminals < 2:
            raise ValueError("grammar_nonterminals must be at least two")
        if self.grammar_rule_concentration <= 0:
            raise ValueError("grammar_rule_concentration must be positive")
        if not 0 <= self.grammar_emission_noise < 1:
            raise ValueError("grammar_emission_noise must lie in [0, 1)")
        teacher_source = self.teacher_input_source.strip().lower()
        value_source = self.target_value_source.strip().lower()
        if teacher_source not in {"raw", "shifted"}:
            raise ValueError("teacher_input_source must be 'raw' or 'shifted'")
        if value_source not in {"raw", "shifted"}:
            raise ValueError("target_value_source must be 'raw' or 'shifted'")
        object.__setattr__(self, "teacher_input_source", teacher_source)
        object.__setattr__(self, "target_value_source", value_source)
        for name in ("teacher_seed", "data_seed", "init_seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        _validate_floating_dtype(self.dtype)
        # Resolve eagerly so invalid device specifications fail at config time.
        torch.device(self.device)

        if self.positional_matrix is not None:
            matrix = torch.as_tensor(self.positional_matrix, dtype=torch.float64)
            expected = (self.seq_len, self.seq_len)
            if tuple(matrix.shape) != expected:
                raise ValueError(f"positional_matrix must have shape {expected}")
            if not bool(torch.isfinite(matrix).all()) or bool((matrix < 0).any()):
                raise ValueError("positional_matrix must be finite and non-negative")
            if not torch.allclose(matrix.sum(dim=-1), torch.ones(self.seq_len, dtype=matrix.dtype)):
                raise ValueError("rows of positional_matrix must sum to one")

    @classmethod
    def exact_paper_bridge(
        cls,
        d_model: int,
        *,
        omega: float = 0.3,
        ensemble: str = "d0",
        teacher_seed: int = 0,
        data_seed: int = 1,
        init_seed: int = 2,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        **overrides: Any,
    ) -> PositionalSemanticDataConfig:
        """Build the faithful ``L=2, r=1`` bridge to arXiv:2402.03902.

        ``overrides`` is intentionally supported for distribution-shift
        parameters (for example ``student_t_df``), but structural paper
        parameters cannot be supplied twice.
        """
        structural = {
            "seq_len",
            "signal_rank",
            "sigma",
            "positional_matrix",
            "teacher_input_source",
            "target_value_source",
        }
        duplicate = structural.intersection(overrides)
        if duplicate:
            names = ", ".join(sorted(duplicate))
            raise ValueError(f"exact paper parameters cannot be overridden: {names}")
        return cls(
            d_model=d_model,
            seq_len=2,
            signal_rank=1,
            ensemble=ensemble,
            sigma=0.5,
            omega=omega,
            positional_matrix=np.array([[0.6, 0.4], [0.4, 0.6]], dtype=np.float64),
            teacher_seed=teacher_seed,
            data_seed=data_seed,
            init_seed=init_seed,
            teacher_input_source="raw",
            target_value_source="raw",
            device=device,
            dtype=dtype,
            **overrides,
        )


@dataclass(frozen=True)
class PositionalSemanticBatch:
    """A fully instrumented batch sampled from a synthetic teacher.

    All attention tensors retain the batch dimension.  Specifically,
    ``semantic_component_attentions`` has shape ``(B, r, L, L)`` and the
    other attention fields have shape ``(B, L, L)``.
    """

    inputs: Tensor
    targets: Tensor
    raw_tokens: Tensor
    positional_attention: Tensor
    semantic_attention: Tensor
    semantic_component_attentions: Tensor
    mixed_attention: Tensor
    hidden_states: Tensor | None = None
    nonterminal_states: Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> PositionalSemanticBatch:
        """Return a copy moved to ``device`` (and optionally ``dtype``)."""

        def move(value: Tensor) -> Tensor:
            if value.is_floating_point():
                return value.to(device=device, dtype=dtype or value.dtype)
            return value.to(device=device)

        return PositionalSemanticBatch(
            inputs=move(self.inputs),
            targets=move(self.targets),
            raw_tokens=move(self.raw_tokens),
            positional_attention=move(self.positional_attention),
            semantic_attention=move(self.semantic_attention),
            semantic_component_attentions=move(self.semantic_component_attentions),
            mixed_attention=move(self.mixed_attention),
            hidden_states=None if self.hidden_states is None else move(self.hidden_states),
            nonterminal_states=(
                None if self.nonterminal_states is None else move(self.nonterminal_states)
            ),
            metadata=dict(self.metadata),
        )


class PositionalSemanticDataset:
    """Quenched positional--semantic teacher and distribution sampler.

    The object owns planted semantic vectors, positional encodings and any
    distribution-specific quenched parameters.  :meth:`sample` is stateless:
    the same explicit seed always produces the same batch and never changes
    NumPy or PyTorch global RNG state.
    """

    def __init__(self, config: PositionalSemanticDataConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype

        teacher_rng = np.random.default_rng(config.teacher_seed)
        teacher = teacher_rng.standard_normal((config.signal_rank, config.d_model))
        self.teacher_vectors = self._tensor(teacher)
        self.semantic_directions = self.teacher_vectors
        self.positional_encoding = self._build_positional_encoding()
        self.positional_encodings = self.positional_encoding
        self.positional_matrix = self._build_positional_matrix()

        data_rng = np.random.default_rng(config.data_seed)
        self.elliptical_covariance: Tensor | None = None
        self.codebook: Tensor | None = None
        self.hmm_transition: Tensor | None = None
        self.hmm_emissions: Tensor | None = None
        self.grammar_rule_probabilities: Tensor | None = None
        self.grammar_emissions: Tensor | None = None
        self._elliptical_cholesky: np.ndarray | None = None
        self._codebook_np: np.ndarray | None = None
        self._hmm_transition_np: np.ndarray | None = None
        self._hmm_emissions_np: np.ndarray | None = None
        self._grammar_rules_np: np.ndarray | None = None
        self._grammar_emissions_np: np.ndarray | None = None
        self._build_ensemble_state(data_rng)

    @classmethod
    def exact_paper_bridge(cls, d_model: int, **kwargs: Any) -> PositionalSemanticDataset:
        """Construct a dataset directly from the exact paper configuration."""
        return cls(PositionalSemanticDataConfig.exact_paper_bridge(d_model, **kwargs))

    def _tensor(self, value: np.ndarray | Tensor) -> Tensor:
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def _build_positional_matrix(self) -> Tensor:
        if self.config.positional_matrix is not None:
            return torch.as_tensor(
                self.config.positional_matrix,
                dtype=self.dtype,
                device=self.device,
            ).clone()
        length = self.config.seq_len
        off_diagonal = 0.4 / (length - 1)
        matrix = torch.full(
            (length, length),
            off_diagonal,
            dtype=self.dtype,
            device=self.device,
        )
        matrix.fill_diagonal_(0.6)
        return matrix

    def _build_positional_encoding(self) -> Tensor:
        length, width = self.config.seq_len, self.config.d_model
        if length == 2:
            encoding = np.ones((2, width), dtype=np.float64)
            encoding[1] *= -1.0
        else:
            # A deterministic real Fourier frame, centered across positions.
            positions = np.arange(length, dtype=np.float64)[:, None]
            frequencies = np.arange(1, width + 1, dtype=np.float64)[None, :]
            encoding = np.cos(2.0 * np.pi * positions * frequencies / length)
            encoding -= encoding.mean(axis=0, keepdims=True)
            norms = np.linalg.norm(encoding, axis=1, keepdims=True)
            encoding /= np.maximum(norms, np.finfo(np.float64).eps)
            encoding *= np.sqrt(width)
        return self._tensor(encoding * self.config.positional_scale / np.sqrt(width))

    def _build_ensemble_state(self, rng: np.random.Generator) -> None:
        cfg = self.config
        if cfg.ensemble == "d1":
            basis, _ = np.linalg.qr(rng.standard_normal((cfg.d_model, cfg.d_model)))
            eigenvalues = np.geomspace(
                1.0 / cfg.elliptical_condition_number,
                1.0,
                cfg.d_model,
            )
            eigenvalues /= eigenvalues.mean()
            covariance = (basis * eigenvalues[None, :]) @ basis.T
            covariance = 0.5 * (covariance + covariance.T)
            self._elliptical_cholesky = np.linalg.cholesky(covariance)
            self.elliptical_covariance = self._tensor(covariance)
        elif cfg.ensemble == "d3":
            codebook = rng.standard_normal((cfg.codebook_size, cfg.d_model))
            codebook -= codebook.mean(axis=0, keepdims=True)
            rms = np.sqrt(np.mean(codebook**2))
            codebook /= max(rms, np.finfo(np.float64).eps)
            self._codebook_np = codebook
            self.codebook = self._tensor(codebook)
        elif cfg.ensemble == "d4":
            states = cfg.hmm_states
            transition = np.full((states, states), (1.0 - cfg.hmm_persistence) / states)
            transition[np.diag_indices(states)] += cfg.hmm_persistence
            emissions = rng.standard_normal((states, cfg.d_model))
            emissions -= emissions.mean(axis=0, keepdims=True)
            rms = np.sqrt(np.mean(emissions**2))
            signal_std = np.sqrt(1.0 - cfg.hmm_emission_noise**2)
            emissions *= signal_std / max(rms, np.finfo(np.float64).eps)
            self._hmm_transition_np = transition
            self._hmm_emissions_np = emissions
            self.hmm_transition = self._tensor(transition)
            self.hmm_emissions = self._tensor(emissions)
        elif cfg.ensemble == "d5":
            states = cfg.grammar_nonterminals
            rules = rng.gamma(
                shape=cfg.grammar_rule_concentration,
                scale=1.0,
                size=(states, states, states),
            )
            rules /= rules.sum(axis=(1, 2), keepdims=True)
            emissions = rng.standard_normal((states, cfg.d_model))
            emissions -= emissions.mean(axis=0, keepdims=True)
            rms = np.sqrt(np.mean(emissions**2))
            signal_std = np.sqrt(1.0 - cfg.grammar_emission_noise**2)
            emissions *= signal_std / max(rms, np.finfo(np.float64).eps)
            self._grammar_rules_np = rules
            self._grammar_emissions_np = emissions
            self.grammar_rule_probabilities = self._tensor(rules)
            self.grammar_emissions = self._tensor(emissions)

    def initialization_generator(self) -> torch.Generator:
        """Return a fresh CPU generator seeded only by ``init_seed``."""
        return torch.Generator(device="cpu").manual_seed(self.config.init_seed)

    def teacher_state(self) -> dict[str, Tensor]:
        """Return detached copies of the teacher quantities needed by runners."""
        state = {
            "semantic_directions": self.teacher_vectors.detach().clone(),
            "positional_encoding": self.positional_encoding.detach().clone(),
            "positional_matrix": self.positional_matrix.detach().clone(),
        }
        if self.elliptical_covariance is not None:
            state["elliptical_covariance"] = self.elliptical_covariance.detach().clone()
        if self.codebook is not None:
            state["codebook"] = self.codebook.detach().clone()
        if self.hmm_transition is not None:
            state["hmm_transition"] = self.hmm_transition.detach().clone()
        if self.hmm_emissions is not None:
            state["hmm_emissions"] = self.hmm_emissions.detach().clone()
        if self.grammar_rule_probabilities is not None:
            state["grammar_rule_probabilities"] = self.grammar_rule_probabilities.detach().clone()
        if self.grammar_emissions is not None:
            state["grammar_emissions"] = self.grammar_emissions.detach().clone()
        return state

    def _sample_innovations(
        self,
        n_samples: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        cfg = self.config
        shape = (n_samples, cfg.seq_len, cfg.d_model)
        hidden: np.ndarray | None = None
        if cfg.ensemble == "d0":
            innovations = rng.standard_normal(shape)
        elif cfg.ensemble == "d1":
            assert self._elliptical_cholesky is not None
            innovations = rng.standard_normal(shape) @ self._elliptical_cholesky.T
        elif cfg.ensemble == "d2":
            # Standard Student-t has variance nu/(nu-2); correct it exactly.
            innovations = rng.standard_t(cfg.student_t_df, size=shape)
            innovations *= np.sqrt((cfg.student_t_df - 2.0) / cfg.student_t_df)
        elif cfg.ensemble == "d3":
            assert self._codebook_np is not None
            indices = rng.integers(0, cfg.codebook_size, size=(n_samples, cfg.seq_len))
            innovations = self._codebook_np[indices]
        elif cfg.ensemble == "d4":
            assert self._hmm_transition_np is not None
            assert self._hmm_emissions_np is not None
            hidden = np.empty((n_samples, cfg.seq_len), dtype=np.int64)
            hidden[:, 0] = rng.integers(0, cfg.hmm_states, size=n_samples)
            for position in range(1, cfg.seq_len):
                transition_rows = self._hmm_transition_np[hidden[:, position - 1]]
                cumulative = np.cumsum(transition_rows, axis=-1)
                uniforms = rng.random(n_samples)
                hidden[:, position] = (uniforms[:, None] > cumulative).sum(axis=-1)
            innovations = self._hmm_emissions_np[hidden]
            innovations = innovations + cfg.hmm_emission_noise * rng.standard_normal(shape)
        else:
            assert self._grammar_rules_np is not None
            assert self._grammar_emissions_np is not None
            states = cfg.grammar_nonterminals
            hidden = np.empty((n_samples, cfg.seq_len), dtype=np.int64)
            for batch_index in range(n_samples):
                root = int(rng.integers(0, states))
                stack = [(0, cfg.seq_len, root)]
                while stack:
                    left, right, parent = stack.pop()
                    if right - left == 1:
                        hidden[batch_index, left] = parent
                        continue
                    split = int(rng.integers(left + 1, right))
                    pair = int(
                        rng.choice(states * states, p=self._grammar_rules_np[parent].reshape(-1))
                    )
                    left_child, right_child = divmod(pair, states)
                    stack.append((split, right, right_child))
                    stack.append((left, split, left_child))
            innovations = self._grammar_emissions_np[hidden]
            innovations = innovations + cfg.grammar_emission_noise * rng.standard_normal(shape)
        return cfg.sigma * innovations, hidden

    def sample(self, n_samples: int, seed: int | None = None) -> PositionalSemanticBatch:
        """Sample a deterministic, fully instrumented teacher batch.

        Parameters
        ----------
        n_samples:
            Batch size ``B``.
        seed:
            Sampling seed.  ``None`` uses ``data_seed``.  Quenched ensemble
            parameters remain tied to ``data_seed`` in either case.

        """
        if not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        sample_seed = self.config.data_seed if seed is None else seed
        if not isinstance(sample_seed, int) or sample_seed < 0:
            raise ValueError("seed must be a non-negative integer")
        rng = np.random.default_rng(sample_seed)
        raw_np, hidden_np = self._sample_innovations(n_samples, rng)
        raw_tokens = self._tensor(raw_np)
        inputs = raw_tokens + self.positional_encoding.unsqueeze(0)

        teacher_tokens = raw_tokens if self.config.teacher_input_source == "raw" else inputs
        value_tokens = raw_tokens if self.config.target_value_source == "raw" else inputs
        projections = torch.einsum("bld,rd->brl", teacher_tokens, self.teacher_vectors)
        component_logits = torch.einsum("brl,brm->brlm", projections, projections)
        denominator = self.config.d_model * self.config.attention_temperature
        component_attentions = torch.softmax(component_logits / denominator, dim=-1)
        semantic_logits = component_logits.sum(dim=1) / denominator
        semantic_attention = torch.softmax(semantic_logits, dim=-1)
        positional_attention = self.positional_matrix.unsqueeze(0).expand(n_samples, -1, -1)
        mixed_attention = (
            1.0 - self.config.omega
        ) * semantic_attention + self.config.omega * positional_attention
        targets = torch.bmm(mixed_attention, value_tokens)
        hidden_states = None
        if hidden_np is not None:
            hidden_states = torch.as_tensor(hidden_np, dtype=torch.long, device=self.device)
        nonterminal_states = hidden_states if self.config.ensemble == "d5" else None

        metadata: dict[str, Any] = {
            "ensemble": self.config.ensemble,
            "teacher_seed": self.config.teacher_seed,
            "data_seed": self.config.data_seed,
            "init_seed": self.config.init_seed,
            "sample_seed": sample_seed,
            "omega": self.config.omega,
            "sigma": self.config.sigma,
            "teacher_input_source": self.config.teacher_input_source,
            "target_value_source": self.config.target_value_source,
        }
        return PositionalSemanticBatch(
            inputs=inputs,
            targets=targets,
            raw_tokens=raw_tokens,
            positional_attention=positional_attention,
            semantic_attention=semantic_attention,
            semantic_component_attentions=component_attentions,
            mixed_attention=mixed_attention,
            hidden_states=hidden_states,
            nonterminal_states=nonterminal_states,
            metadata=metadata,
        )


__all__ = [
    "EnsembleName",
    "PositionalSemanticBatch",
    "PositionalSemanticDataConfig",
    "PositionalSemanticDataset",
]
