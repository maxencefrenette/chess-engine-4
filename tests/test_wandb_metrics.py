from __future__ import annotations

from chess_engine_4.training.wandb_metrics import (
    tail_metrics_from_history,
    tail_values,
    wandb_run_path_from_url,
)


def test_wandb_run_path_from_url() -> None:
    assert (
        wandb_run_path_from_url("https://wandb.ai/maxence-frenette/chess-engine-4/runs/abc123")
        == "maxence-frenette/chess-engine-4/abc123"
    )


def test_tail_values_sort_by_step_and_skip_missing() -> None:
    rows = [
        {"_step": 20, "loss/total": 2.0},
        {"_step": 10, "loss/total": 1.0},
        {"_step": 30},
        {"_step": 40, "loss/total": 4.0},
    ]

    assert tail_values(sorted(rows, key=lambda row: row["_step"]), "loss/total", tail=2) == [
        2.0,
        4.0,
    ]


def test_tail_metrics_from_history() -> None:
    metrics = tail_metrics_from_history(
        "https://wandb.ai/entity/project/runs/runid",
        [
            {"_step": 1, "loss/total": 10.0, "metrics/policy_top1": 0.1},
            {"_step": 3, "loss/total": 30.0, "metrics/policy_top1": 0.3},
            {"_step": 2, "loss/total": 20.0, "metrics/policy_top1": 0.2},
        ],
        tail=2,
    )

    assert metrics.loss == 25.0
    assert metrics.policy_top1 == 0.25
    assert metrics.tail_count == 2
