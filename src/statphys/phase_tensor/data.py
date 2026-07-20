"""Synthetic-to-natural byte datasets with content-addressed corpus caches."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

VOCABULARY = 258
BOS = 1


@dataclass(frozen=True)
class CorpusSpec:
    repository: str
    subset: str | None
    split: str
    field: str
    revision: str = "main"


CORPUS_SPECS = {
    "tinystories": CorpusSpec(
        "roneneldan/TinyStories", None, "train", "text", "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
    ),
    "simplestories": CorpusSpec(
        "SimpleStories/SimpleStories", None, "train", "story", "e63b8adc3b1a1bdc7cac5b500d150b71346b0628"
    ),
    "fineweb_edu": CorpusSpec(
        "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    ),
    # Parquet conversion of the official v1_6 sample; the upstream repository
    # still depends on dataset scripts removed in datasets 5.
    "dolma": CorpusSpec(
        "devingulliver/dolma-v1_6-sample", None, "train", "text", "6b03e4bbe0d9633c408555a7a8667b944033b46c"
    ),
}


@dataclass(frozen=True)
class TokenDataset:
    inputs: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    metadata: dict[str, Any]

    @property
    def size(self) -> int:
        return int(self.inputs.shape[0])


def token_data_summary(dataset: TokenDataset) -> dict[str, float]:
    """Compute an O(1) corpus summary vector on the registered training sample."""
    tokens = dataset.inputs.detach().cpu().numpy().astype(np.int64, copy=False)
    counts = np.bincount(tokens.reshape(-1), minlength=VOCABULARY).astype(np.float64)
    mass = counts / max(counts.sum(), 1.0)
    nonzero = mass[mass > 0.0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    effective_vocabulary = math.exp(entropy) / VOCABULARY
    ranks = np.arange(1, nonzero.size + 1, dtype=np.float64)
    ordered = np.sort(nonzero)[::-1]
    zipf_exponent = float(-np.polyfit(np.log(ranks), np.log(ordered), deg=1)[0]) if ordered.size >= 3 else 0.0
    pairs = tokens[:, :-1].reshape(-1) * VOCABULARY + tokens[:, 1:].reshape(-1)
    pair_mass = np.bincount(pairs, minlength=VOCABULARY * VOCABULARY).reshape(VOCABULARY, VOCABULARY)
    pair_mass = pair_mass / max(pair_mass.sum(), 1.0)
    marginal_left = pair_mass.sum(axis=1, keepdims=True)
    marginal_right = pair_mass.sum(axis=0, keepdims=True)
    valid = pair_mass > 0.0
    mutual_information = float(np.sum(pair_mass[valid] * np.log(pair_mass[valid] / (marginal_left * marginal_right)[valid])))
    sample = tokens[: min(tokens.shape[0], 1024)]
    duplicate_fraction = 1.0 - np.unique(sample, axis=0).shape[0] / max(sample.shape[0], 1)
    positive_counts = counts[counts > 0.0]
    tail_ratio = float(counts.max() / max(float(positive_counts.mean()) if positive_counts.size else 1.0, 1.0))
    return {
        "data_effective_vocabulary_fraction": float(effective_vocabulary),
        "data_zipf_exponent": zipf_exponent,
        "data_entropy_rate_normalized": float(entropy / math.log(VOCABULARY)),
        "data_adjacent_mi_normalized": float(mutual_information / math.log(VOCABULARY)),
        "data_duplicate_fraction": float(duplicate_fraction),
        "data_frequency_tail_ratio": float(tail_ratio / (1.0 + tail_ratio)),
    }


def _corpus_path(root: str | Path, name: str) -> Path:
    return Path(root) / "byte_corpora" / f"{name}.bin"


def _dolma_records() -> Any:
    """Stream official Dolma JSONL shards without the retired HF dataset script."""
    import gzip
    import requests

    index_url = (
        "https://huggingface.co/datasets/allenai/dolma/resolve/main/"
        "urls/v1_6-sample.txt"
    )
    index_response = requests.get(index_url, timeout=120)
    index_response.raise_for_status()
    shard_urls = [line.strip() for line in index_response.text.splitlines() if line.strip()]
    if not shard_urls:
        raise RuntimeError("the official Dolma shard index was empty")
    for shard_url in shard_urls:
        with requests.get(shard_url, stream=True, timeout=120) as response:
            response.raise_for_status()
            response.raw.decode_content = False
            with gzip.GzipFile(fileobj=response.raw) as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if isinstance(record, dict):
                        yield record


def prepare_corpus(name: str, root: str | Path, max_bytes: int = 64 * 1024 * 1024) -> Path:
    """Stream and atomically cache a bounded, version-recorded corpus prefix."""
    if name not in CORPUS_SPECS:
        raise ValueError(f"unknown corpus: {name}")
    target = _corpus_path(root, name)
    metadata_path = target.with_suffix(".json")
    if target.exists() and metadata_path.exists() and target.stat().st_size >= max_bytes:
        return target
    spec = CORPUS_SPECS[name]
    if spec.repository == "allenai/dolma":
        dataset = _dolma_records()
    else:
        from datasets import load_dataset

        kwargs: dict[str, Any] = {
            "path": spec.repository,
            "split": spec.split,
            "streaming": True,
            "revision": spec.revision,
            "cache_dir": str(Path(root) / "huggingface"),
        }
        if spec.subset is not None:
            kwargs["name"] = spec.subset
        dataset = load_dataset(**kwargs)
    payload = bytearray()
    documents = 0
    for record in dataset:
        text = record.get(spec.field)
        if not isinstance(text, str) or not text.strip():
            continue
        payload.extend(text.encode("utf-8", errors="replace"))
        payload.extend(b"\n\n")
        documents += 1
        if len(payload) >= max_bytes:
            break
    if not payload:
        raise RuntimeError(f"no text was streamed for corpus {name}")
    payload = payload[:max_bytes]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)
    digest = sha256(payload).hexdigest()
    metadata = {
        "name": name,
        "repository": spec.repository,
        "subset": spec.subset,
        "split": spec.split,
        "field": spec.field,
        "revision": spec.revision,
        "bytes": len(payload),
        "documents": documents,
        "sha256": digest,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return target


def _encode_bytes(values: np.ndarray) -> np.ndarray:
    return values.astype(np.int64) + 2


def _natural_windows(
    corpus: bytes,
    count: int,
    length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(corpus) <= length + 1:
        raise ValueError("cached corpus is shorter than one requested sequence")
    starts = rng.integers(0, len(corpus) - length - 1, size=count)
    sequences = np.stack(
        [np.frombuffer(corpus[start : start + length + 1], dtype=np.uint8) for start in starts]
    )
    encoded = _encode_bytes(sequences)
    return encoded[:, :-1], encoded[:, 1:], np.ones((count, length), dtype=np.float32)


def _retrieval_sequences(
    count: int,
    length: int,
    rng: np.random.Generator,
    noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if length < 16:
        raise ValueError("retrieval sequences require sequence_length >= 16")
    sequences = rng.integers(2 + ord("a"), 2 + ord("z") + 1, size=(count, length + 1))
    pairs = max(2, min(6, (length - 4) // 3))
    for row in range(count):
        keys = rng.choice(np.arange(2 + ord("a"), 2 + ord("h") + 1), size=pairs, replace=False)
        values = rng.choice(np.arange(2 + ord("A"), 2 + ord("Z") + 1), size=pairs, replace=False)
        for index, (key, value) in enumerate(zip(keys, values)):
            offset = 3 * index
            sequences[row, offset : offset + 3] = (key, 2 + ord("="), value)
        selected = int(rng.integers(0, pairs))
        sequences[row, -4:] = (2 + ord("?"), keys[selected], 2 + ord("="), values[selected])
        if rng.random() < noise:
            sequences[row, -1] = int(rng.choice(values))
    mask = np.zeros((count, length), dtype=np.float32)
    mask[:, -1] = 1.0
    return sequences[:, :-1], sequences[:, 1:], mask


def _counting_sequences(
    count: int,
    length: int,
    rng: np.random.Generator,
    noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences = rng.integers(2 + ord("a"), 2 + ord("d") + 1, size=(count, length + 1))
    for row in range(count):
        query = int(rng.integers(2 + ord("a"), 2 + ord("d") + 1))
        answer = int(np.count_nonzero(sequences[row, :-4] == query) % 10)
        if rng.random() < noise:
            answer = int(rng.integers(0, 10))
        sequences[row, -4:] = (2 + ord("?"), query, 2 + ord("="), 2 + ord("0") + answer)
    mask = np.zeros((count, length), dtype=np.float32)
    mask[:, -1] = 1.0
    return sequences[:, :-1], sequences[:, 1:], mask


def _hmm_sequences(
    count: int,
    length: int,
    rng: np.random.Generator,
    noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences = np.empty((count, length + 1), dtype=np.int64)
    for row in range(count):
        state = int(rng.integers(0, 4))
        for position in range(length + 1):
            if rng.random() < 0.15:
                state = (state + int(rng.choice((-1, 1)))) % 4
            emitted = state if rng.random() >= noise else int(rng.integers(0, 4))
            sequences[row, position] = 2 + ord("a") + emitted
    return sequences[:, :-1], sequences[:, 1:], np.ones((count, length), dtype=np.float32)


def _injected_sequences(
    corpus: bytes,
    count: int,
    length: int,
    rng: np.random.Generator,
    noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if length < 16:
        raise ValueError("natural injection requires sequence_length >= 16")
    prefix_length = length - 7
    starts = rng.integers(0, len(corpus) - prefix_length - 1, size=count)
    sequences = np.empty((count, length + 1), dtype=np.int64)
    for row, start in enumerate(starts):
        prefix = _encode_bytes(np.frombuffer(corpus[start : start + prefix_length], dtype=np.uint8))
        key = int(rng.integers(2 + ord("a"), 2 + ord("h") + 1))
        value = int(rng.integers(2 + ord("A"), 2 + ord("Z") + 1))
        answer = value if rng.random() >= noise else int(rng.integers(2 + ord("A"), 2 + ord("Z") + 1))
        trailer = np.asarray(
            [2 + ord("|"), key, 2 + ord("="), value, 2 + ord("?"), key, 2 + ord("="), answer],
            dtype=np.int64,
        )
        sequences[row] = np.concatenate((prefix, trailer))
    mask = np.zeros((count, length), dtype=np.float32)
    mask[:, -1] = 1.0
    return sequences[:, :-1], sequences[:, 1:], mask


def build_token_dataset(
    kind: str,
    *,
    count: int,
    length: int,
    seed: int,
    noise: float = 0.0,
    data_root: str | Path | None = None,
) -> TokenDataset:
    rng = np.random.default_rng(seed)
    normalized = kind.lower()
    metadata: dict[str, Any] = {"kind": normalized, "seed": seed, "noise": noise}
    if normalized == "synthetic_retrieval":
        inputs, targets, mask = _retrieval_sequences(count, length, rng, noise)
    elif normalized == "synthetic_counting":
        inputs, targets, mask = _counting_sequences(count, length, rng, noise)
    elif normalized == "synthetic_hmm":
        inputs, targets, mask = _hmm_sequences(count, length, rng, noise)
    else:
        corpus_name = "fineweb_edu" if normalized == "natural_injected" else normalized
        root = data_root or os.environ.get("STATPHYS_DATA_ROOT")
        if root is None:
            raise RuntimeError("set STATPHYS_DATA_ROOT for semi-natural and natural datasets")
        corpus_path = _corpus_path(root, corpus_name)
        if not corpus_path.exists():
            raise FileNotFoundError(f"prepare the registered corpus first: {corpus_path}")
        corpus = corpus_path.read_bytes()
        metadata_path = corpus_path.with_suffix(".json")
        if metadata_path.exists():
            metadata.update(json.loads(metadata_path.read_text()))
        if normalized == "natural_injected":
            inputs, targets, mask = _injected_sequences(corpus, count, length, rng, noise)
        else:
            inputs, targets, mask = _natural_windows(corpus, count, length, rng)
    digest = sha256(inputs.tobytes() + targets.tobytes() + mask.tobytes()).hexdigest()
    metadata.update({"examples": count, "sequence_length": length, "sample_sha256": digest})
    return TokenDataset(
        torch.as_tensor(inputs, dtype=torch.long),
        torch.as_tensor(targets, dtype=torch.long),
        torch.as_tensor(mask, dtype=torch.float32),
        metadata,
    )
