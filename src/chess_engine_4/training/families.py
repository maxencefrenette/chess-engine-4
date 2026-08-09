"""Canonical model-family artifact registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FamilySpec:
    family: str
    config: Path
    best_runs: Path
    throughput: Path
    anchor_ratio: float


FAMILY_SPECS = (
    FamilySpec(
        family="dense",
        config=Path("configs/dense.py"),
        best_runs=Path("experiments/best-runs-dense.toml"),
        throughput=Path("experiments/throughput-dense.toml"),
        anchor_ratio=0.2,
    ),
    FamilySpec(
        family="moe64a2",
        config=Path("configs/moe64a2.py"),
        best_runs=Path("experiments/best-runs-moe64a2.toml"),
        throughput=Path("experiments/throughput-moe64a2.toml"),
        anchor_ratio=0.05,
    ),
)
FAMILIES = {spec.family: spec for spec in FAMILY_SPECS}
