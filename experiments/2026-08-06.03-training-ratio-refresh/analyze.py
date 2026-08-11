"""Reproduce the dense and MoE loss-versus-cost comparison."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
GPU_DOLLARS_PER_SEC = {"B200": 0.001736, "RTX-PRO-6000": 0.000842}
COLORS = {
    0.01: "#2563eb",
    0.02: "#111827",
    0.05: "#16a34a",
    0.1: "#2563eb",
    0.2: "#111827",
    0.5: "#16a34a",
}
TARGETS = {
    "dense": (3.4, 3.3, 3.2, 3.1, 3.05, 3.0, 2.95),
    "moe64a2": (3.4, 3.3, 3.2, 3.1, 3.05, 3.0, 2.95, 2.9),
}


@dataclass(frozen=True)
class Result:
    family: str
    d_model: int
    training_ratio: float
    steps: int
    loss: float
    eligible: bool


def main() -> None:
    results = read_results()
    plot(results)
    for family in ("dense", "moe64a2"):
        print(family)
        for target in TARGETS[family]:
            candidates = {
                ratio: cost
                for ratio in sorted(
                    {result.training_ratio for result in results if result.family == family}
                )
                if (cost := interpolated_cost(results, family, ratio, target)) is not None
            }
            if candidates:
                ratio, cost = min(candidates.items(), key=lambda item: item[1])
                print(f"  loss={target:g} ratio={ratio:g} cost=${cost:.3f}")


def read_results() -> list[Result]:
    with (HERE / "results.toml").open("rb") as handle:
        rows = tomllib.load(handle)["runs"]
    return [
        Result(
            family=str(row["family"]),
            d_model=int(row["d_model"]),
            training_ratio=float(row["training_ratio"]),
            steps=int(row["steps"]),
            loss=float(row["loss"]),
            eligible=bool(row["eligible"]),
        )
        for row in rows
    ]


def steady_state_cost(result: Result) -> float:
    with (HERE.parent / f"throughput-{result.family}.toml").open("rb") as handle:
        throughput = tomllib.load(handle)["models"][f"d{result.d_model}"]
    seconds = result.steps * float(throughput["measured_wall_ms_per_step"]) / 1000.0
    return seconds * GPU_DOLLARS_PER_SEC[str(throughput["gpu"])]


def interpolated_cost(
    results: list[Result],
    family: str,
    ratio: float,
    target_loss: float,
) -> float | None:
    points = sorted(
        (
            result
            for result in results
            if result.family == family
            and result.training_ratio == ratio
            and result.eligible
        ),
        key=lambda result: result.loss,
        reverse=True,
    )
    for weaker, stronger in pairwise(points):
        if weaker.loss >= target_loss >= stronger.loss:
            fraction = (weaker.loss - target_loss) / (weaker.loss - stronger.loss)
            return math.exp(
                math.log(steady_state_cost(weaker))
                + fraction
                * (math.log(steady_state_cost(stronger)) - math.log(steady_state_cost(weaker)))
            )
    return None


def plot(results: list[Result]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    for axis, family, title in zip(
        axes,
        ("dense", "moe64a2"),
        ("Dense", "MoE 64A2"),
        strict=True,
    ):
        family_results = [result for result in results if result.family == family]
        for ratio in sorted({result.training_ratio for result in family_results}):
            eligible = sorted(
                (
                    result
                    for result in family_results
                    if result.training_ratio == ratio and result.eligible
                ),
                key=lambda result: result.d_model,
            )
            color = COLORS[ratio]
            axis.plot(
                [steady_state_cost(result) for result in eligible],
                [result.loss for result in eligible],
                color=color,
                linewidth=1.3,
                marker="o",
                label=f"{ratio:g}x",
            )
            rejected = [
                result
                for result in family_results
                if result.training_ratio == ratio and not result.eligible
            ]
            axis.scatter(
                [steady_state_cost(result) for result in rejected],
                [result.loss for result in rejected],
                color=color,
                marker="x",
                s=65,
                linewidth=1.7,
            )
        axis.set_xscale("log")
        axis.set_xlabel("Steady-state Modal GPU cost (USD)")
        axis.set_ylabel("Final loss")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend(title="Training ratio", frameon=False)
    figure.suptitle("Final loss by training cost", fontsize=15, fontweight="bold")
    figure.tight_layout()
    figure.savefig(HERE / "loss-vs-cost.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()
