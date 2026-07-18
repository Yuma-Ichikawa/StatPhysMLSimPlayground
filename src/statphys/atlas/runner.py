"""One-run execution engine for the Transformer phase atlas."""

from __future__ import annotations

import math
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .artifacts import RunArtifactStore
from .bridge_training import bridge_loss, train_bridge
from .schema import DataStage, InitStrategy, Precision, RunSpec


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _stage_name(run: RunSpec) -> str:
    return run.phase.architecture.value.split("_", 1)[0]


def _ensemble_name(stage: DataStage) -> str:
    return stage.value.split("_", 1)[0]


def _omega(run: RunSpec) -> float:
    # Schema 1.0 called this the semantic mixture.  New manifests use the
    # unambiguous positional_mixture; retain an explicit compatibility bridge.
    if hasattr(run.phase, "positional_mixture"):
        return float(run.phase.positional_mixture)
    return 1.0 - float(run.phase.semantic_mixture)


def _torch_dtype(precision: Precision) -> Any:
    import torch

    if precision == Precision.FLOAT64:
        return torch.float64
    if precision == Precision.BFLOAT16:
        # Generate teacher data in float32; autocast controls training math.
        return torch.float32
    return torch.float32


def build_dataset(run: RunSpec) -> Any:
    """Construct a quenched teacher without consulting global RNG state."""

    from .data import PositionalSemanticDataConfig, PositionalSemanticDataset

    scale = run.phase.scaling
    common = dict(
        d_model=scale.d_model,
        omega=_omega(run),
        ensemble=_ensemble_name(run.phase.data),
        teacher_seed=run.seeds.seed("teacher"),
        data_seed=run.seeds.seed("data"),
        init_seed=run.seeds.seed("initialization"),
        device="cpu",
        dtype=_torch_dtype(run.training.precision),
        attention_temperature=run.phase.temperature,
        elliptical_condition_number=run.phase.covariance_condition,
        student_t_df=run.phase.tail_degrees_freedom,
    )
    exact = scale.sequence_length == 2 and scale.teacher_rank == 1
    if exact:
        config = PositionalSemanticDataConfig.exact_paper_bridge(**common)
    else:
        config = PositionalSemanticDataConfig(
            seq_len=scale.sequence_length,
            signal_rank=scale.teacher_rank,
            sigma=run.phase.input_noise,
            **common,
        )
    return PositionalSemanticDataset(config)


def build_model(run: RunSpec, dataset: Any) -> Any:
    from .models import build_attention_ladder

    scale = run.phase.scaling
    stage = _stage_name(run)
    model = build_attention_ladder(
        stage=stage,
        d_model=scale.d_model,
        seq_len=scale.sequence_length,
        signal_rank=scale.teacher_rank,
        n_heads=scale.n_heads,
        head_dim=scale.head_dim,
        n_layers=scale.n_layers,
        ffn_dim=scale.ff_dim,
        attention_temperature=run.phase.temperature,
        init_seed=run.seeds.seed("initialization"),
        device="cpu",
        dtype=_torch_dtype(run.training.precision),
    )
    strategy = run.initialization.value
    if strategy == InitStrategy.INTERPOLATED.value:
        strategy = "mixed"
    model.initialize_from_directions(
        dataset.teacher_vectors,
        dataset.positional_encoding,
        strategy=strategy,
        noise_scale=0.0,
    )
    return model


def _teacher_templates(batch: Any) -> tuple[Any, Any]:
    import torch

    positional = torch.bmm(batch.positional_attention, batch.raw_tokens)
    semantic = torch.bmm(batch.semantic_attention, batch.raw_tokens)
    return positional, semantic


def _functional_probe(model: Any, heldout: Any) -> dict[str, float]:
    import torch

    from .observables import centered_functional_overlap, two_template_decomposition

    device = next(model.parameters()).device
    batch = heldout.to(device)
    positional, semantic = _teacher_templates(batch)
    with torch.no_grad():
        prediction, diagnostics = model(batch.inputs, return_diagnostics=True)
    decomposition = two_template_decomposition(prediction, positional, semantic)
    attention = diagnostics["attention_maps"].mean(dim=(0, 1, 2))
    semantic_map = batch.semantic_attention.mean(dim=0)
    positional_map = batch.positional_attention.mean(dim=0)
    sample_loss = float(bridge_loss(prediction, batch.targets) / batch.targets.shape[0])
    return {
        "test_loss_per_sample": sample_loss,
        "functional_m_pos": float(decomposition["m_pos"]),
        "functional_m_sem": float(decomposition["m_sem"]),
        "functional_r2": float(decomposition["r2"]),
        "functional_residual_fraction": float(decomposition["residual_fraction"]),
        "attention_positional_overlap": centered_functional_overlap(
            attention, positional_map, zero_variance="zero"
        ),
        "attention_semantic_overlap": centered_functional_overlap(
            attention, semantic_map, zero_variance="zero"
        ),
    }


def _weight_metrics(model: Any, dataset: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from .observables import (
        head_specialization_metrics,
        latent_overlap_matrix,
        matrix_spectrum,
        permutation_invariant_head_spectrum,
    )

    qk = model.effective_qk_matrices(detach=True).float().cpu().numpy()
    flattened = qk.reshape(-1, qk.shape[-2] * qk.shape[-1])
    semantic_vectors = dataset.teacher_vectors.detach().float().cpu().numpy()
    semantic_operators = np.einsum("rd,re->rde", semantic_vectors, semantic_vectors)
    positions = dataset.positional_encoding.detach().float().cpu().numpy()
    positions = positions - positions.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(positions, full_matrices=False)
    positional_vector = vh[0]
    positional_operator = np.outer(positional_vector, positional_vector)
    templates = np.concatenate((positional_operator[None], semantic_operators), axis=0)
    overlap = latent_overlap_matrix(flattened, templates.reshape(templates.shape[0], -1), absolute=True)
    specialization = head_specialization_metrics(overlap)
    invariant = permutation_invariant_head_spectrum(overlap)

    spectra: list[np.ndarray] = []
    spectral_norms: list[float] = []
    outlier_ratios: list[float] = []
    if getattr(model.config, "stage_number", 9) <= 1:
        # Exact low-rank singular values via thin QR factors; avoids O(d^3)
        # work for the d=1000 calibration runs.
        for block in model.blocks:
            q = block.low_rank_q.detach().float().cpu().numpy()[0]
            k = block.low_rank_k_effective.detach().float().cpu().numpy()[0]
            _, rq = np.linalg.qr(q, mode="reduced")
            _, rk = np.linalg.qr(k, mode="reduced")
            singular = np.linalg.svd(rq @ rk.T, compute_uv=False)
            spectra.append(singular)
            spectral_norms.append(float(singular[0]))
            outlier_ratios.append(
                float(singular[0] / singular[1]) if singular.size > 1 and singular[1] > 0 else math.inf
            )
    else:
        for matrix in qk.reshape(-1, qk.shape[-2], qk.shape[-1]):
            spectrum = matrix_spectrum(matrix, top_k=min(16, matrix.shape[0]))
            spectra.append(np.asarray(spectrum["top_singular_values"]))
            spectral_norms.append(float(spectrum["spectral_norm"]))
            outlier_ratios.append(float(spectrum["outlier_ratio"]))
    padded_spectra = np.full((len(spectra), max(map(len, spectra))), np.nan)
    for index, values in enumerate(spectra):
        padded_spectra[index, : len(values)] = values
    summary = {
        "weight_m_pos_max": float(overlap[:, 0].max(initial=0.0)),
        "weight_m_sem_max": float(overlap[:, 1:].max(initial=0.0)),
        "weight_m_pos_mean": float(overlap[:, 0].mean()),
        "weight_m_sem_mean": float(overlap[:, 1:].mean()),
        "specialization_strength": float(specialization["specialization_strength"]),
        "specialization_entropy": float(specialization["normalized_specialization_entropy"]),
        "effective_heads": float(specialization["effective_heads"]),
        "dead_head_fraction": float(specialization["dead_head_fraction"]),
        "redundant_head_fraction": float(specialization["redundant_head_fraction"]),
        "head_overlap_effective_rank": float(invariant["effective_rank"]),
        "qk_spectral_norm_max": max(spectral_norms),
        "qk_outlier_ratio_max": max(outlier_ratios),
    }
    return summary, {
        "head_latent_overlap": overlap,
        "qk_top_singular_values": padded_spectra,
    }


def _final_diagnostics(model: Any, heldout: Any, dataset: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch

    from .analysis import classify_phase
    from .observables import (
        attention_geometry,
        intervention_loss_deltas,
        position_correlation,
        representation_statistics,
    )

    device = next(model.parameters()).device
    batch = heldout.to(device)
    positional, semantic = _teacher_templates(batch)
    with torch.no_grad():
        prediction, diagnostics = model(batch.inputs, return_diagnostics=True)
        no_attention = model(batch.inputs, ablate_attention=True)
        no_mlp = model(batch.inputs, ablate_mlp=True)
    functional = _functional_probe(model, heldout)
    phase = classify_phase(functional["functional_m_pos"], functional["functional_m_sem"])
    attention = diagnostics["attention_maps"].detach().float().cpu().numpy()
    geometry = attention_geometry(attention)
    representations = diagnostics["representations"][:, -1].detach().float().cpu().numpy()
    representation = representation_statistics(representations)
    correlations = position_correlation(representations)
    baseline = float(bridge_loss(prediction, batch.targets) / batch.targets.shape[0])
    interventions = intervention_loss_deltas(
        baseline,
        {
            "remove_attention": float(bridge_loss(no_attention, batch.targets) / batch.targets.shape[0]),
            "remove_mlp": float(bridge_loss(no_mlp, batch.targets) / batch.targets.shape[0]),
        },
    )
    weights, weight_arrays = _weight_metrics(model, dataset)
    summary: dict[str, Any] = {
        **functional,
        **weights,
        "phase_label": phase["label"],
        "attention_entropy": geometry["entropy"],
        "attention_effective_support": geometry["effective_support"],
        "attention_span": geometry["span"],
        "attention_sink_mass": geometry["sink_mass"],
        "attention_diagonal_mass": geometry["diagonal_mass"],
        "attention_previous_token_mass": geometry["previous_token_mass"],
        "representation_participation_ratio": representation["participation_ratio"],
        "representation_effective_rank": representation["effective_rank"],
        "representation_anisotropy": representation["anisotropy"],
        "representation_top_fraction": representation["top_explained_fraction"],
        "correlation_length_integral": float(
            0.5 + np.nansum(np.maximum(correlations[1:], 0.0))
        ),
        "remove_attention_delta": interventions["interventions"]["remove_attention"]["delta"],
        "remove_mlp_delta": interventions["interventions"]["remove_mlp"]["delta"],
    }
    arrays = {
        **weight_arrays,
        "position_correlation": correlations,
        "attention_per_map_entropy": np.asarray(geometry["per_map"]["entropy"]),
        "representation_covariance_eigenvalues": np.asarray(
            representation["covariance_eigenvalues"]
        ),
    }
    if getattr(run_observation_policy := heldout, "metadata", None) is not None:
        # Deliberately no-op: retained for forward-compatible batch metadata.
        del run_observation_policy
    return _jsonable(summary), arrays


def run_experiment(
    run: RunSpec,
    output_root: str | Path,
    *,
    device: str = "auto",
    repo: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute and transactionally register one immutable run specification."""

    store = RunArtifactStore(output_root)
    directory = store.begin(run, repo=repo, overwrite=overwrite)
    try:
        dataset = build_dataset(run)
        train_batch = dataset.sample(run.phase.scaling.n_train, seed=run.seeds.seed("data"))
        heldout = dataset.sample(
            run.observables.heldout_size,
            seed=(run.seeds.seed("data") + 2_147_483_647) % (2**32),
        )
        model = build_model(run, dataset)
        probe = lambda current, _step: _functional_probe(current, heldout)
        result = train_bridge(
            model,
            train_batch.inputs,
            train_batch.targets,
            run.training,
            seed=run.seeds.seed("minibatch"),
            device=device,
            probe=probe,
            checkpoint_dir=directory / "checkpoints",
        )
        diagnostics, arrays = _final_diagnostics(model, heldout, dataset)
        summary = {
            "run_id": run.run_id,
            "objective_normalization": "sum_squared_error_over_2d",
            "n_train": run.phase.scaling.n_train,
            "alpha": run.phase.scaling.alpha,
            "omega_positional": _omega(run),
            "architecture": run.phase.architecture.value,
            "data": run.phase.data.value,
            "initialization": run.initialization.value,
            "replica": run.replica,
            **result.summary(),
            "train_loss_per_sample": result.final_loss / run.phase.scaling.n_train,
            **diagnostics,
        }
        trajectory_arrays = {name: np.asarray(values) for name, values in result.history.items()}
        store.save_arrays(run.run_id, "trajectories", **trajectory_arrays)
        store.save_arrays(run.run_id, "diagnostics", **arrays)
        teacher = {
            name: value.detach().float().cpu().numpy()
            for name, value in dataset.teacher_state().items()
        }
        store.save_arrays(run.run_id, "teacher", **teacher)
        store.save_summary(run.run_id, summary)
        store.set_status(run.run_id, "completed")
        return summary
    except BaseException as exc:
        store.set_status(
            run.run_id,
            "failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback="".join(traceback.format_exception(exc))[-12_000:],
        )
        raise

