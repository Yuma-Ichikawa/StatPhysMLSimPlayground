"""Trainable phase-continuation tensor for realistic Transformer settings."""

from .data import CORPUS_SPECS, prepare_corpus
from .runner import run_phase_tensor

__all__ = ["CORPUS_SPECS", "prepare_corpus", "run_phase_tensor"]
