"""Compare a completed run with the current training-FLOPs loss curve."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from chess_engine_4.training.scaling_laws import (
    DEFAULT_BEST_RUNS,
    fit_loss_power_law,
    read_best_runs,
)
from chess_engine_4.training.wandb_metrics import metrics_from_summary, wandb_run_path_from_url


@dataclass(frozen=True, slots=True)
class RunComparison:
    wandb_url: str
    d_model: int
    training_ratio: float
    flops: float
    loss: float
    predicted_loss: float
    eg_flops: float
    incumbent_eg_flops: float | None

    @property
    def beats_trend(self) -> bool:
        return self.eg_flops > 1.0

    @property
    def improves_width_default(self) -> bool:
        return self.incumbent_eg_flops is None or self.eg_flops > self.incumbent_eg_flops


def compare_run() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a W&B run with the fitted loss versus training-FLOPs curve."
    )
    parser.add_argument("wandb_url")
    parser.add_argument("--best-runs", type=Path, default=DEFAULT_BEST_RUNS)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    import wandb

    run = wandb.Api(timeout=args.timeout).run(wandb_run_path_from_url(args.wandb_url))
    comparison = compare_run_data(
        wandb_url=args.wandb_url,
        config=run.config,
        summary=run.summary,
        best_runs_path=args.best_runs,
    )
    width_verdict = "PROMOTE" if comparison.improves_width_default else "KEEP INCUMBENT"
    trend_verdict = "BEATS TREND" if comparison.beats_trend else "BELOW TREND"
    print(f"width_verdict: {width_verdict}")
    print(f"trend_verdict: {trend_verdict}")
    print(f"d_model: {comparison.d_model}")
    print(f"training_ratio: {comparison.training_ratio:g}")
    print(f"flops: {comparison.flops:.6e}")
    print(f"loss: {comparison.loss:.6f}")
    print(f"fitted_loss: {comparison.predicted_loss:.6f}")
    print(f"EG_flops: {comparison.eg_flops:.3f}x")
    if comparison.incumbent_eg_flops is None:
        print("incumbent_EG_flops: none")
    else:
        print(f"incumbent_EG_flops: {comparison.incumbent_eg_flops:.3f}x")
        print(
            "width_EG_flops_delta: "
            f"{comparison.eg_flops - comparison.incumbent_eg_flops:+.3f}x"
        )


def compare_run_data(
    *,
    wandb_url: str,
    config: Mapping[str, object],
    summary: Mapping[str, object],
    best_runs_path: Path,
) -> RunComparison:
    flops_per_sample = _positive_int(config, "flops_per_sample")
    batch_size = _positive_int(config, "batch_size")
    steps = _positive_int(config, "steps")
    d_model = _positive_int(config, "d_model")
    training_ratio = _positive_float(config, "training_ratio")
    candidate_flops = flops_per_sample * batch_size * steps
    metrics = metrics_from_summary(wandb_url, summary)
    if metrics.loss_spike_count:
        raise ValueError(
            f"W&B run is invalid: detected {metrics.loss_spike_count} loss spike(s)."
        )
    defaults = read_best_runs(best_runs_path)
    loss_curve = fit_loss_power_law((result.flops, result.loss) for result in defaults)
    same_width = [
        result
        for result in defaults
        if result.d_model == d_model and result.training_ratio == training_ratio
    ]
    incumbent_eg_flops = (
        max(loss_curve.flops_for_loss(result.loss) / result.flops for result in same_width)
        if same_width
        else None
    )
    return RunComparison(
        wandb_url=wandb_url,
        d_model=d_model,
        training_ratio=training_ratio,
        flops=candidate_flops,
        loss=metrics.loss,
        predicted_loss=loss_curve.predict(candidate_flops),
        eg_flops=loss_curve.flops_for_loss(metrics.loss) / candidate_flops,
        incumbent_eg_flops=incumbent_eg_flops,
    )


def _positive_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int | float) or value <= 0 or int(value) != value:
        raise ValueError(f"W&B run config has no positive integer {key!r} value.")
    return int(value)


def _positive_float(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"W&B run config has no positive {key!r} value.")
    return float(value)
