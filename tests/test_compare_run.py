from __future__ import annotations

from pathlib import Path

import pytest

from chess_engine_4.training.compare_run import compare_run_data
from chess_engine_4.training.wandb_metrics import (
    LOSS_MEAN_KEY,
    LOSS_SPIKE_COUNT_KEY,
    POLICY_TOP1_KEY,
)


def test_compare_run_reports_eg_flops(tmp_path: Path) -> None:
    best_runs = tmp_path / "best.toml"
    best_runs.write_text(
        """
[runs.low]
model_kind = "dense"
run_name = "low"
wandb_url = "https://wandb.ai/e/p/runs/low"
flops = 1e4
d_model = 64
depth = 2
batch_size = 10
lr = 1e-3
params = 100
samples_seen = 1000
loss = 4.0
policy_top1 = 0.2

[runs.high]
model_kind = "dense"
run_name = "high"
wandb_url = "https://wandb.ai/e/p/runs/high"
flops = 1e6
d_model = 128
depth = 2
batch_size = 10
lr = 1e-3
params = 200
samples_seen = 2000
loss = 3.0
policy_top1 = 0.3
""".strip(),
        encoding="utf-8",
    )
    comparison = compare_run_data(
        wandb_url="https://wandb.ai/e/p/runs/candidate",
        config={"flops_per_sample": 10, "batch_size": 5, "steps": 20, "d_model": 64},
        summary={
            LOSS_MEAN_KEY: 2.9,
            POLICY_TOP1_KEY: 0.4,
            LOSS_SPIKE_COUNT_KEY: 0,
        },
        best_runs_path=best_runs,
    )

    assert comparison.flops == 1_000
    assert comparison.eg_flops > 0
    assert comparison.improves_width_default
    assert comparison.beats_trend
    assert comparison.incumbent_eg_flops is not None
    assert comparison.eg_flops > comparison.incumbent_eg_flops


def test_compare_run_rejects_loss_spikes(tmp_path: Path) -> None:
    best_runs = tmp_path / "best.toml"
    best_runs.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="detected 1 loss spike"):
        compare_run_data(
            wandb_url="https://wandb.ai/e/p/runs/candidate",
            config={"flops_per_sample": 10, "batch_size": 5, "steps": 20, "d_model": 64},
            summary={
                LOSS_MEAN_KEY: 2.9,
                POLICY_TOP1_KEY: 0.4,
                LOSS_SPIKE_COUNT_KEY: 1,
            },
            best_runs_path=best_runs,
        )
