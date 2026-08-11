"""Plan acquisition of the canonical L-shaped scaling-law ladder."""

from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path

from chess_engine_4.training.budget_planner import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_DATASET,
    DEFAULT_FOCUS_BUDGET,
    bootstrap_family_fits,
    fit_families,
    read_dataset_samples,
    read_family_evidence,
    suggest_runs,
)
from chess_engine_4.training.families import FAMILIES

DEFAULT_LADDERS = Path("experiments/scaling-ladders.toml")


@dataclass(frozen=True, slots=True)
class LadderSpec:
    family: str
    anchor_width: int
    width_ratio: float
    widths: tuple[int, ...]
    data_ratios: tuple[float, ...]

    @property
    def scaffold(self) -> frozenset[tuple[int, float]]:
        return frozenset(
            {(self.anchor_width, ratio) for ratio in self.data_ratios}
            | {(width, self.width_ratio) for width in self.widths}
        )

    @property
    def grid(self) -> frozenset[tuple[int, float]]:
        return frozenset((width, ratio) for width in self.widths for ratio in self.data_ratios)


def read_ladder(path: Path, family: str) -> LadderSpec:
    with path.open("rb") as handle:
        row = tomllib.load(handle)[family]
    return LadderSpec(
        family=family,
        anchor_width=int(row["anchor_width"]),
        width_ratio=float(row["width_ratio"]),
        widths=tuple(int(value) for value in row["widths"]),
        data_ratios=tuple(float(value) for value in row["data_ratios"]),
    )


def missing_scaffold(
    ladder: LadderSpec, observed: frozenset[tuple[int, float]]
) -> list[tuple[int, float]]:
    return sorted(ladder.scaffold - observed, key=lambda item: (item[1], item[0]))


def ladder_planner() -> None:
    parser = argparse.ArgumentParser(
        description="Complete an L-shaped scaling ladder, then rank cheap interior runs."
    )
    parser.add_argument("--family", choices=tuple(FAMILIES), default="dense")
    parser.add_argument("--ladder", type=Path, default=DEFAULT_LADDERS)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--focus-budget", type=float, default=DEFAULT_FOCUS_BUDGET)
    parser.add_argument("--max-run-cost", type=float, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--assume-samples", type=int, default=None)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("count must be positive")
    if args.focus_budget <= 0:
        parser.error("focus-budget must be positive")
    if args.bootstrap_samples < 20:
        parser.error("bootstrap-samples must be at least 20")
    max_cost = args.focus_budget * 0.1 if args.max_run_cost is None else args.max_run_cost
    if max_cost <= 0:
        parser.error("max-run-cost must be positive")

    ladder = read_ladder(args.ladder, args.family)
    spec = FAMILIES[args.family]
    evidence = read_family_evidence((spec,))
    if not evidence:
        parser.error(f"{args.family} has insufficient scaling evidence")
    observed = evidence[0].observed_coordinates
    missing = missing_scaffold(ladder, observed)
    print(
        f"ladder: {args.family}, d{ladder.anchor_width} data arm, "
        f"{ladder.width_ratio:g}x width arm"
    )
    print(f"scaffold: {len(ladder.scaffold) - len(missing)}/{len(ladder.scaffold)} observed")
    if missing:
        print("decision: complete the scaffold before selecting interior runs")
        for width, ratio in missing:
            print(
                f"  d{width} at {ratio:g}x: uv run train-modal --config {spec.config} "
                f"--d-model {width} --training-ratio {ratio:g}"
            )
        return

    assume_samples = (
        read_dataset_samples(args.dataset)
        if args.assume_samples is None
        else args.assume_samples
    )
    fits = fit_families(evidence)
    ensembles = bootstrap_family_fits(
        evidence, fits, samples=args.bootstrap_samples, seed=2026
    )
    suggestions = suggest_runs(
        focus_budget=args.focus_budget,
        count=args.count,
        max_cost=max_cost,
        assume_samples=assume_samples,
        evidence=evidence,
        fits=fits,
        bootstrap_fits=ensembles,
        allowed_coordinates=ladder.grid,
    )
    print(
        f"decision: rank unobserved interior cells (max ${max_cost:g} each, "
        f"${args.focus_budget:g} final budget)"
    )
    if not suggestions:
        print("  no eligible run")
    for index, suggestion in enumerate(suggestions, start=1):
        print(
            f"  {index}. d{suggestion.d_model} at {suggestion.training_ratio:g}x, "
            f"batch={suggestion.samples // suggestion.steps:,}, steps={suggestion.steps:,}, "
            f"cost=${suggestion.estimated_cost:.3f}, "
            f"net_loss_improvement={suggestion.expected_loss_improvement:+.5f}"
        )
        print(f"     {suggestion.command}")
    if suggestions and suggestions[0].expected_loss_improvement > 0:
        print("recommendation: run candidate 1 before the final training run")
    elif suggestions:
        print("recommendation: skip interior runs and preserve the final-run budget")


if __name__ == "__main__":
    ladder_planner()
