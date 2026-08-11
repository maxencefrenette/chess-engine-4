"""Dense recipe with Muon on hidden matrices and fused AdamW elsewhere."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from chess_engine_4.training.config import load_training_config

_DENSE_CONFIG = Path(__file__).resolve().parents[2] / "configs/dense.py"


def config(*, d_model: int, training_ratio: float = 0.055, history_length: int = 8):
    baseline = load_training_config(
        _DENSE_CONFIG,
        d_model=d_model,
        training_ratio=training_ratio,
        history_length=history_length,
    )
    return replace(
        baseline,
        optimizer=replace(baseline.optimizer, algorithm="muon"),
    )
