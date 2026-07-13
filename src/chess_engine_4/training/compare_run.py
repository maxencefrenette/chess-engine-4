"""Compare a completed run with the current modified-compute loss curve."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from chess_engine_4.training.flops import modified_compute
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
    flops: float
    modified_compute: float
    loss: float
    loss_upper_1sd: float
    predicted_loss: float
    predicted_loss_upper_1sd: float
    flops_efficiency: float
    modified_compute_efficiency: float
    incumbent_flops_efficiency: float | None

    @property
    def residual(self) -> float:
        return self.loss_upper_1sd - self.predicted_loss_upper_1sd

    @property
    def beats_trend(self) -> bool:
        return self.residual < 0

    @property
    def improves_width_default(self) -> bool:
        return (
            self.incumbent_flops_efficiency is None
            or self.flops_efficiency > self.incumbent_flops_efficiency
        )


def compare_run() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a W&B run with the fitted loss_upper_1sd curve."
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
    print(f"flops: {comparison.flops:.6e}")
    print(f"loss: {comparison.loss:.6f}")
    print(f"fitted_loss: {comparison.predicted_loss:.6f}")
    print(f"loss_flops_compute_efficiency: {comparison.flops_efficiency:.3f}x")
    if comparison.incumbent_flops_efficiency is None:
        print("incumbent_loss_flops_compute_efficiency: none")
    else:
        print(
            "incumbent_loss_flops_compute_efficiency: "
            f"{comparison.incumbent_flops_efficiency:.3f}x"
        )
        print(
            "width_compute_efficiency_delta: "
            f"{comparison.flops_efficiency - comparison.incumbent_flops_efficiency:+.3f}x"
        )
    print(f"modified_compute: {comparison.modified_compute:.6e}")
    print(f"loss_upper_1sd: {comparison.loss_upper_1sd:.6f}")
    print(f"fitted_loss_upper_1sd: {comparison.predicted_loss_upper_1sd:.6f}")
    print(f"residual: {comparison.residual:+.6f}")
    print(
        "loss_upper_modified_compute_efficiency: "
        f"{comparison.modified_compute_efficiency:.3f}x"
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
    candidate_flops = flops_per_sample * batch_size * steps
    candidate_compute = modified_compute(
        flops_per_sample=flops_per_sample,
        batch_size=batch_size,
        steps=steps,
    )
    metrics = metrics_from_summary(wandb_url, summary)
    if metrics.loss_spike_count:
        raise ValueError(
            f"W&B run is invalid: detected {metrics.loss_spike_count} loss spike(s)."
        )
    defaults = read_best_runs(best_runs_path)
    loss_curve = fit_loss_power_law(
        (result.compute / (result.samples_seen / result.batch_size), result.loss)
        for result in defaults
    )
    same_width = [result for result in defaults if result.d_model == d_model]
    incumbent_flops_efficiency = (
        max(
            loss_curve.compute_for_loss(result.loss)
            / (result.compute / (result.samples_seen / result.batch_size))
            for result in same_width
        )
        if same_width
        else None
    )
    upper_curve = fit_loss_power_law(
        (result.compute, result.loss_upper_1sd) for result in defaults
    )
    return RunComparison(
        wandb_url=wandb_url,
        d_model=d_model,
        flops=candidate_flops,
        modified_compute=candidate_compute,
        loss=metrics.loss,
        loss_upper_1sd=metrics.loss_upper_1sd,
        predicted_loss=loss_curve.predict(candidate_flops),
        predicted_loss_upper_1sd=upper_curve.predict(candidate_compute),
        flops_efficiency=(
            loss_curve.compute_for_loss(metrics.loss) / candidate_flops
        ),
        modified_compute_efficiency=(
            upper_curve.compute_for_loss(metrics.loss_upper_1sd) / candidate_compute
        ),
        incumbent_flops_efficiency=incumbent_flops_efficiency,
    )


def _positive_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int | float) or value <= 0 or int(value) != value:
        raise ValueError(f"W&B run config has no positive integer {key!r} value.")
    return int(value)
