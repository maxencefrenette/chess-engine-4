"""Compare the minimum-safe width arm with the established 0.1x arm."""

from __future__ import annotations

import csv
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from chess_engine_4.training.scaling_laws import SkalingLaw, fit_skaling_law

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
BEST_RUNS = ROOT / "experiments/best-runs-dense.toml"
RESULTS = HERE / "results.csv"
MATCHED_WIDTHS = (128, 256, 512, 1024)


@dataclass(frozen=True, slots=True)
class Point:
    name: str
    width: int
    ratio: float
    params: int
    samples: int
    loss: float
    spikes: int

    def fit_tuple(self) -> tuple[int, int, float]:
        return self.params, self.samples, self.loss


def main() -> None:
    canonical = _canonical_points()
    short = _short_points()
    data_arm = [point for point in canonical if point.width == 64]
    old_width_arm = [
        point
        for point in canonical
        if point.width in MATCHED_WIDTHS and point.ratio == 0.1
    ]
    holdout = [point for point in canonical if point.width > 64 and point.ratio > 0.1]
    short_matched = [point for point in short if point.width in MATCHED_WIDTHS]
    short_full = [point for point in short if point.width > 64]
    short_stable = [point for point in short_full if point.spikes == 0]

    fits = {
        "old_0.1x_arm": _fit_summary(data_arm + old_width_arm, holdout),
        "short_0.055x_matched_arm": _fit_summary(data_arm + short_matched, holdout),
        "short_0.055x_with_d768": _fit_summary(data_arm + short_full, holdout),
        "short_0.055x_stable_only": _fit_summary(data_arm + short_stable, holdout),
    }
    canonical_law = fit_skaling_law([point.fit_tuple() for point in canonical])
    output = {
        "canonical_holdout": [point.name for point in holdout],
        "fits": fits,
        "canonical_fit_prediction_mape_on_short_arm": _mape(canonical_law, short),
        "short_arm_leave_one_width_out": _leave_one_width_out(data_arm, short_full),
        "short_runs": [asdict(point) for point in short],
        "protocol_valid": not any(point.spikes for point in short),
    }
    path = HERE / "results.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(path)


def _fit_summary(train: list[Point], holdout: list[Point]) -> dict[str, object]:
    law = fit_skaling_law([point.fit_tuple() for point in train])
    return {
        "training_rows": [point.name for point in train],
        "heldout_mape": _mape(law, holdout),
        "model_exponent": law.model_exponent,
        "coupling": law.coupling,
        "effective_model_exponent": law.model_exponent * law.coupling,
        "law": law.format(),
    }


def _leave_one_width_out(
    data_arm: list[Point], width_arm: list[Point]
) -> list[dict[str, float | int]]:
    output = []
    for heldout in width_arm:
        train = data_arm + [point for point in width_arm if point != heldout]
        law = fit_skaling_law([point.fit_tuple() for point in train], restarts=32)
        output.append(
            {
                "heldout_width": heldout.width,
                "heldout_ape": _mape(law, [heldout]),
                "model_exponent": law.model_exponent,
                "coupling": law.coupling,
                "effective_model_exponent": law.model_exponent * law.coupling,
            }
        )
    return output


def _mape(law: SkalingLaw, points: list[Point]) -> float:
    return 100 * sum(
        abs(law.predict(point.params, point.samples) - point.loss) / point.loss
        for point in points
    ) / len(points)


def _canonical_points() -> list[Point]:
    with BEST_RUNS.open("rb") as file:
        rows = tomllib.load(file)["scaling_runs"]
    return [
        Point(
            name=name,
            width=int(row["d_model"]),
            ratio=float(row["training_ratio"]),
            params=int(row["params"]),
            samples=int(row["samples_seen"]),
            loss=float(row["loss"]),
            spikes=int(row["loss_spike_count"]),
        )
        for name, row in rows.items()
    ]


def _short_points() -> list[Point]:
    with RESULTS.open(newline="", encoding="utf-8") as file:
        rows = [
            Point(
                name=row["run_name"],
                width=int(row["width"]),
                ratio=float(row["training_ratio"]),
                params=int(row["params"]),
                samples=int(row["samples_seen"]),
                loss=float(row["loss"]),
                spikes=int(row["loss_spike_count"]),
            )
            for row in csv.DictReader(file)
        ]
    selected: dict[int, Point] = {}
    for row in rows:
        if row.spikes == 0:
            selected[row.width] = row
    return list(selected.values())


if __name__ == "__main__":
    main()
