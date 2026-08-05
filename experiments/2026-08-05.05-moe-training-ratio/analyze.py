"""Reproduce the MoE training-ratio allocation charts and fit."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from chess_engine_4.training.scaling_laws import (
    UndertrainingLossLaw,
    fit_loss_power_law,
    fit_undertraining_loss_law,
)

HERE = Path(__file__).parent
ANCHOR_RATIO = 0.02
TARGET_LOSSES = (3.3, 3.2, 3.1, 3.0, 2.95, 2.9, 2.86)
COLORS = {0.01: "#2563eb", 0.02: "#111827", 0.05: "#16a34a", 0.1: "#dc2626"}


@dataclass(frozen=True)
class Result:
    d_model: int
    training_ratio: float
    flops: float
    loss: float
    runtime_sec: float
    eligible: bool

    @property
    def cost(self) -> float:
        return self.runtime_sec * 6.25 / 3600


def main() -> None:
    results = read_results()
    plot_resource(results, resource="flops", output="loss-vs-flops.svg")
    plot_resource(results, resource="cost", output="loss-vs-realized-cost.svg")
    law = fit_ratio_law(results)

    print(law.baseline.format())
    print(
        "penalty = "
        f"{law.penalty_coefficient:.6f} * "
        f"(C0.02 / 1e15)^-{law.compute_exponent:.6f} * "
        f"((r / 0.02)^-{law.ratio_exponent:.6f} - 1)"
    )
    print(f"RMSE = {law.rmse:.6f}")
    print("\nEmpirical realized-cost interpolation:")
    for target in TARGET_LOSSES:
        candidates = {
            ratio: cost
            for ratio in sorted({result.training_ratio for result in results})
            if (cost := interpolated_cost(results, ratio, target)) is not None
        }
        winner, cost = min(candidates.items(), key=lambda item: item[1])
        print(f"loss={target:g} ratio={winner:g} cost=${cost:.3f}")


def read_results() -> list[Result]:
    with (HERE / "results.toml").open("rb") as handle:
        rows = tomllib.load(handle)["runs"]
    return [
        Result(
            d_model=int(row["d_model"]),
            training_ratio=float(row["training_ratio"]),
            flops=float(row["flops"]),
            loss=float(row["loss"]),
            runtime_sec=float(row["runtime_sec"]),
            eligible=bool(row["eligible"]),
        )
        for row in rows
    ]


def fit_ratio_law(results: list[Result]) -> UndertrainingLossLaw:
    eligible = [result for result in results if result.eligible]
    baselines = [result for result in eligible if result.training_ratio == ANCHOR_RATIO]
    baseline = fit_loss_power_law((result.flops, result.loss) for result in baselines)
    candidates = [result for result in eligible if result.training_ratio != ANCHOR_RATIO]
    return fit_undertraining_loss_law(
        baseline,
        (
            (
                result.flops * ANCHOR_RATIO / result.training_ratio,
                result.training_ratio / ANCHOR_RATIO,
                result.loss,
            )
            for result in candidates
        ),
    )


def interpolated_cost(
    results: list[Result],
    ratio: float,
    target_loss: float,
) -> float | None:
    points = sorted(
        (result for result in results if result.eligible and result.training_ratio == ratio),
        key=lambda result: result.loss,
        reverse=True,
    )
    for weaker, stronger in zip(points, points[1:], strict=False):
        if weaker.loss >= target_loss >= stronger.loss:
            fraction = (weaker.loss - target_loss) / (weaker.loss - stronger.loss)
            return math.exp(
                math.log(weaker.cost) + fraction * (math.log(stronger.cost) - math.log(weaker.cost))
            )
    return None


def plot_resource(results: list[Result], *, resource: str, output: str) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    for ratio in sorted({result.training_ratio for result in results}):
        points = sorted(
            (result for result in results if result.eligible and result.training_ratio == ratio),
            key=lambda result: result.d_model,
        )
        x_values = [getattr(result, resource) for result in points]
        axis.plot(
            x_values,
            [result.loss for result in points],
            color=COLORS[ratio],
            linewidth=1.4,
            marker="o",
            label=f"{ratio:g}x",
        )
        rejected = [
            result for result in results if not result.eligible and result.training_ratio == ratio
        ]
        axis.scatter(
            [getattr(result, resource) for result in rejected],
            [result.loss for result in rejected],
            color=COLORS[ratio],
            marker="x",
            s=65,
            linewidth=1.6,
        )

    axis.set_xscale("log")
    axis.set_xlabel("Training FLOPs" if resource == "flops" else "Realized B200 cost (USD)")
    axis.set_ylabel("Final loss")
    axis.set_title(
        "MoE loss by training FLOPs"
        if resource == "flops"
        else "MoE loss by realized training cost"
    )
    axis.grid(alpha=0.2)
    axis.legend(title="Training ratio", frameon=False, ncols=2)
    figure.tight_layout()
    figure.savefig(HERE / output)
    plt.close(figure)


if __name__ == "__main__":
    main()
