"""Optional instrumentation for the production training loop."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class TrainingProfileConfig:
    warmup_steps: int = 50
    profile_steps: int = 200

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if self.profile_steps <= 0:
            raise ValueError("profile_steps must be positive.")

    @property
    def total_steps(self) -> int:
        return self.warmup_steps + self.profile_steps


def summarize_profile(
    *,
    profile: TrainingProfileConfig,
    records: list[dict[str, Any]],
    measured_wall_start: float | None,
    overall_wall_start: float,
    device: torch.device,
    batch_size: int,
    flops_per_sample: int,
    theoretical_tflops: float | None,
) -> dict[str, Any]:
    torch.cuda.synchronize(device)
    measured_wall_end = time.perf_counter()
    measured = records[profile.warmup_steps :]
    if not measured or measured_wall_start is None:
        raise RuntimeError("Training ended before profiling collected any measured steps.")

    if profile.warmup_steps > 0:
        previous_for_gap = records[profile.warmup_steps - 1 : -1]
        current_for_gap = measured
    else:
        previous_for_gap = measured[:-1]
        current_for_gap = measured[1:]

    copy_ms = [record["copy_start"].elapsed_time(record["copy_end"]) for record in measured]
    train_ms = [record["train_start"].elapsed_time(record["train_end"]) for record in measured]
    idle_gap_ms = [
        previous["train_end"].elapsed_time(current["copy_start"])
        for previous, current in zip(previous_for_gap, current_for_gap, strict=True)
    ]
    measured_wall_ms_per_step = (measured_wall_end - measured_wall_start) * 1000.0 / len(measured)
    train_gpu_mean_ms = statistics.fmean(train_ms)
    achieved_train_only_tflops = batch_size * flops_per_sample / (train_gpu_mean_ms / 1000.0) / 1e12
    gpu_work_mean_ms = statistics.fmean(copy_ms) + train_gpu_mean_ms
    gpu_idle_gap_mean_ms = statistics.fmean(idle_gap_ms) if idle_gap_ms else 0.0

    return {
        "warmup_steps": profile.warmup_steps,
        "profile_steps": len(measured),
        "overall_wall_sec": measured_wall_end - overall_wall_start,
        "measured_wall_ms_per_step": measured_wall_ms_per_step,
        "theoretical_tflops": theoretical_tflops,
        "achieved_train_only_tflops": achieved_train_only_tflops,
        "train_only_mfu": (
            achieved_train_only_tflops / theoretical_tflops if theoretical_tflops else None
        ),
        "end_to_end_mfu": (
            batch_size
            * flops_per_sample
            / (measured_wall_ms_per_step / 1000.0)
            / 1e12
            / theoretical_tflops
            if theoretical_tflops and measured_wall_ms_per_step > 0
            else None
        ),
        "gpu_work_mean_ms": gpu_work_mean_ms,
        "gpu_idle_gap_mean_ms": gpu_idle_gap_mean_ms,
        "gpu_idle_fraction_of_step": gpu_idle_gap_mean_ms / measured_wall_ms_per_step,
        "data_fetch_wall": _summarize([record["fetch_wall_ms"] for record in measured]),
        "enqueue_wall": _summarize([record["enqueue_wall_ms"] for record in measured]),
        "h2d_copy_gpu": _summarize(copy_ms),
        "train_gpu": _summarize(train_ms),
        "gpu_idle_gap": _summarize(idle_gap_ms),
        "stream_copy_plus_train": _summarize(
            [record["copy_start"].elapsed_time(record["train_end"]) for record in measured]
        ),
    }


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p90_ms": 0.0, "max_ms": 0.0}
    sorted_values = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p90_ms": sorted_values[int(0.9 * (len(sorted_values) - 1))],
        "max_ms": max(values),
    }
