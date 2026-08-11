from __future__ import annotations

from pathlib import Path

from chess_engine_4.training.compare_run import compare_run_data
from chess_engine_4.training.wandb_metrics import (
    LOSS_MEAN_KEY,
    LOSS_SPIKE_COUNT_KEY,
    POLICY_TOP1_KEY,
)


def test_compare_run_reports_eg_flops() -> None:
    comparison = compare_run_data(
        wandb_url="https://wandb.ai/e/p/runs/candidate",
        config={
            "flops_per_sample": 10,
            "batch_size": 100,
            "steps": 97_956,
            "parameter_count": 979_488,
            "d_model": 64,
            "training_ratio": 0.2,
        },
        summary={
            LOSS_MEAN_KEY: 3.5,
            POLICY_TOP1_KEY: 0.4,
            LOSS_SPIKE_COUNT_KEY: 1,
        },
        best_runs_path=Path("experiments/best-runs-dense.toml"),
    )

    assert comparison.flops == 97_956_000
    assert comparison.training_ratio == 0.2
    assert comparison.eg_flops > 0
    assert comparison.improves_width_default
    assert comparison.beats_trend
    assert comparison.incumbent_eg_flops is not None
    assert comparison.eg_flops > comparison.incumbent_eg_flops
