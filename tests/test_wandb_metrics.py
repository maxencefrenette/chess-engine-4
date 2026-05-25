from __future__ import annotations

from chess_engine_4.training.wandb_metrics import (
    metrics_from_summary,
    wandb_run_path_from_url,
)


def test_wandb_run_path_from_url() -> None:
    assert (
        wandb_run_path_from_url("https://wandb.ai/maxence-frenette/chess-engine-4/runs/abc123")
        == "maxence-frenette/chess-engine-4/abc123"
    )


def test_metrics_from_summary() -> None:
    metrics = metrics_from_summary(
        "https://wandb.ai/entity/project/runs/runid",
        {
            "loss/task[ema=0.99]": 4.2,
            "loss/task2[ema=0.99]": 17.89,
            "metrics/policy_top1[ema=0.99]": 0.3,
            "router/dead_experts": 0.25,
            "router/dead_experts_max": 1.0,
        },
    )

    assert metrics.loss == 4.2
    assert metrics.loss_std == 0.5
    assert metrics.loss_upper_1sd == 4.7
    assert metrics.policy_top1 == 0.3
    assert metrics.router_dead_experts == 0.25
    assert metrics.router_dead_experts_max == 1.0
