"""Release contracts for the manuscript, figures, and follow-up design."""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"


def test_manuscript_figures_have_exact_size_and_complete_coverage() -> None:
    manifest = json.loads((FIGURES / "generated_manifest.json").read_text(encoding="utf-8"))
    assert manifest["figure_size_inches"] == [6.4, 4.8]
    assert len(manifest["figures"]) == 9

    sources = "\n".join(path.read_text(encoding="utf-8") for path in PAPER.rglob("*.tex"))
    referenced = set(re.findall(r"figures/(figure[^}]+\.pdf)", sources))
    assert referenced == set(manifest["figures"])

    for name in manifest["figures"]:
        pdf = FIGURES / name
        png = pdf.with_suffix(".png")
        assert pdf.is_file() and pdf.stat().st_size > 0
        assert png.is_file() and png.stat().st_size > 0
        assert b"/MediaBox [ 0 0 460.8 345.6 ]" in pdf.read_bytes()

        data = png.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", data[16:24])
        assert (width, height) == (1920, 1440)


def test_dense_confirmation_is_a_360_task_prospective_design() -> None:
    config_path = (
        ROOT
        / "experiments"
        / "phase_continuation"
        / "configs"
        / "tensor_reference_dense_confirmation.toml"
    )
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    experiment = config["experiments"][0]
    task_count = (
        len(config["study"]["seeds"])
        * len(experiment["variants"])
        * len(experiment["controls"])
        * len(experiment["sizes"])
    )
    assert task_count == 360
    assert experiment["parameters"]["prospective_holdout_width"] == max(experiment["sizes"])
    assert len(set(experiment["controls"])) == len(experiment["controls"])


def test_public_manuscript_artifacts_contain_no_machine_local_coordinates() -> None:
    files = [PAPER / "README.md", ROOT / "scripts" / "generate_paper_figures.py"]
    files.extend(PAPER.rglob("*.tex"))
    files.append(
        ROOT
        / "experiments"
        / "phase_continuation"
        / "configs"
        / "tensor_reference_dense_confirmation.toml"
    )
    forbidden = ("/mnt/", "/home/", "fsas-2025", "localhost", "127.0.0.1")
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path
