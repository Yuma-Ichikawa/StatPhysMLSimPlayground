from __future__ import annotations

import hashlib
import json

import pytest

from statphys.phase_tensor.data import build_token_dataset


def _write_corpus(root, name: str, payload: bytes) -> None:
    directory = root / "byte_corpora"
    directory.mkdir(parents=True)
    (directory / f"{name}.bin").write_bytes(payload)
    (directory / f"{name}.json").write_text(
        json.dumps({"name": name, "sha256": hashlib.sha256(payload).hexdigest()})
    )


def test_natural_corpus_splits_are_byte_disjoint_and_hashed(tmp_path):
    payload = bytes(range(256)) * 64
    _write_corpus(tmp_path, "tinystories", payload)

    train = build_token_dataset(
        "tinystories", count=8, length=32, seed=11, data_root=tmp_path, corpus_split="train"
    )
    test = build_token_dataset(
        "tinystories", count=8, length=32, seed=13, data_root=tmp_path, corpus_split="test"
    )
    ood = build_token_dataset(
        "tinystories", count=8, length=32, seed=17, data_root=tmp_path, corpus_split="ood"
    )

    ranges = [
        (item.metadata["corpus_byte_start"], item.metadata["corpus_byte_stop"])
        for item in (train, test, ood)
    ]
    assert ranges[0][1] <= ranges[1][0]
    assert ranges[1][1] <= ranges[2][0]
    assert len({item.metadata["corpus_partition_sha256"] for item in (train, test, ood)}) == 3


def test_unknown_corpus_split_is_rejected(tmp_path):
    _write_corpus(tmp_path, "tinystories", bytes(range(256)) * 8)
    with pytest.raises(ValueError, match="unknown corpus split"):
        build_token_dataset(
            "tinystories",
            count=2,
            length=16,
            seed=11,
            data_root=tmp_path,
            corpus_split="calibration",
        )
