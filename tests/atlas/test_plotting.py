"""Headless output tests for paper-oriented atlas figures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")

from statphys.atlas.plotting import (  # noqa: E402
    generate_paper_figures,
    plot_finite_size_collapse,
    plot_fluctuation_diagnostics,
    plot_order_parameters,
    plot_phase_map,
    plot_spectral_specialization,
)


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for size in (4, 9):
        for control in (-0.5, 0.0, 0.5):
            for teacher in (11, 12):
                scaled_observable = size**-0.5 * (1.0 + control * size**0.5)
                rows.append(
                    {
                        "run_id": f"{size}-{control}-{teacher}",
                        "eligible_for_claims": True,
                        "eligible_for_transition": True,
                        "teacher_seed": teacher,
                        "data_seed": teacher + 100,
                        "optimizer_seed": teacher + 200,
                        "spec.phase.semantic_mixture": control,
                        "spec.phase.temperature": 1.0,
                        "spec.phase.scaling.d_model": size,
                        "phase_label": "positional" if control < 0 else "semantic",
                        "summary.functional_m_pos": 0.5 - control,
                        "summary.functional_m_sem": 0.5 + control,
                        "mean_m_pos": 0.5 - control,
                        "mean_m_sem": 0.5 + control,
                        "susceptibility_competition": size * (1.0 - abs(control) / 2.0),
                        "binder_competition_raw": 0.4 + 0.1 * control,
                        "binder_competition_centered": 0.3 + 0.08 * control,
                        "summary.scaling_observable": scaled_observable,
                        "diagnostics.qk_singular_values_mean": [2.0 + control, 0.5],
                        "summary.qk_spectral_norm_max": 2.0 + control,
                        "summary.qk_outlier_ratio_max": 4.0 + control,
                        "summary.specialization_strength": 0.2 + control**2,
                        "summary.specialization_entropy": 0.6 - 0.1 * control,
                        "summary.effective_heads": 1.0 + size / 10.0,
                    }
                )
    return rows


def _assert_pdf_png(output: object) -> None:
    paths = output.paths  # type: ignore[attr-defined]
    assert set(paths) == {"pdf", "png"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def test_all_standard_figures_render_pdf_and_png_headlessly(tmp_path: Path) -> None:
    rows = _rows()
    outputs = [
        plot_phase_map(rows, tmp_path),
        plot_order_parameters(rows, tmp_path),
        plot_fluctuation_diagnostics(rows, tmp_path),
        plot_finite_size_collapse(
            rows,
            tmp_path,
            critical_control=0.0,
            observable_exponent=0.5,
            inverse_nu=0.5,
            observable_column="summary.scaling_observable",
            n_grid=32,
        ),
        plot_spectral_specialization(rows, tmp_path, top_k=2),
    ]
    for output in outputs:
        _assert_pdf_png(output)
    assert matplotlib.get_backend().lower() == "agg"
    assert outputs[1].metadata["hierarchy"] == "optimizer→data→teacher"
    assert outputs[3].metadata["score"] == pytest.approx(0.0, abs=1e-24)
    assert outputs[4].metadata["n_spectral_series"] > 0
    assert outputs[4].metadata["n_specialization_series"] > 0


def test_phase_disagreement_and_missing_cells_are_explicit(tmp_path: Path) -> None:
    rows = _rows()
    duplicate = dict(rows[0])
    duplicate["phase_label"] = "semantic"
    output = plot_phase_map([rows[0], duplicate], tmp_path, stem="disagreement")
    _assert_pdf_png(output)
    assert output.metadata["labels"] == ["unresolved"]

    missing = dict(rows[1])
    missing["eligible_for_transition"] = False
    missing["eligible_for_claims"] = False
    missing_output = plot_phase_map([missing], tmp_path, stem="missing")
    assert missing_output.metadata["n_missing"] == 1


def test_fss_plot_rejects_missing_grid_instead_of_extrapolating(tmp_path: Path) -> None:
    incomplete = [
        row
        for row in _rows()
        if not (
            row["spec.phase.scaling.d_model"] == 9
            and row["spec.phase.semantic_mixture"] == 0.5
        )
    ]
    with pytest.raises(ValueError, match="complete grid"):
        plot_finite_size_collapse(
            incomplete,
            tmp_path,
            critical_control=0.0,
            observable_exponent=0.5,
            inverse_nu=0.5,
            observable_column="summary.scaling_observable",
        )


def test_plotters_do_not_write_outside_requested_directory(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "paper_figures"
    output = plot_order_parameters(_rows(), destination, stem="custom_orders")
    _assert_pdf_png(output)
    assert all(path.parent == destination for path in output.paths.values())
    assert {path.name for path in output.paths.values()} == {
        "custom_orders.pdf",
        "custom_orders.png",
    }


def test_generate_paper_figures_loads_cli_bundle_and_reports_outputs(tmp_path: Path) -> None:
    source = tmp_path / "atlas_aggregate.json"
    rows = _rows()
    source.write_text(
        json.dumps({"runs": rows, "ensembles": rows, "claims": [], "metadata": {"test": True}}),
        encoding="utf-8",
    )
    report = generate_paper_figures(source, tmp_path / "figures")
    assert report["n_generated"] == 5
    assert not report["skipped"]
    assert Path(report["manifest"]).is_file()
    for figure in report["generated"].values():
        assert set(figure["paths"]) == {"pdf", "png"}

