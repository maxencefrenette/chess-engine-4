"""Plot paired holdout errors for d32- and d64-anchored Skaling fits."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent


def main() -> None:
    results = json.loads((HERE / "results.json").read_text())
    result = next(row for row in results if row.get("comparison") == "l_shape_anchors")
    holdout = result["common_holdout"]
    labels = [name.removeprefix("new_runs.").replace("-refresh", "") for name in holdout]
    errors = {
        width: [
            result["anchors"][width]["laws"]["skaling"][
                "absolute_percentage_errors"
            ][name]
            for name in holdout
        ]
        for width in ("32", "64")
    }

    positions = np.arange(len(labels))
    width = 0.38
    fig, axis = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    axis.bar(positions - width / 2, errors["32"], width, label="d32 anchor", color="#e07a1f")
    axis.bar(positions + width / 2, errors["64"], width, label="d64 anchor", color="#2774ae")
    axis.axhline(
        result["anchors"]["32"]["laws"]["skaling"]["common_holdout_mape"],
        color="#e07a1f",
        linestyle="--",
        linewidth=1.3,
    )
    axis.axhline(
        result["anchors"]["64"]["laws"]["skaling"]["common_holdout_mape"],
        color="#2774ae",
        linestyle="--",
        linewidth=1.3,
    )
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set_ylabel("Absolute percentage error")
    axis.set_title("Skaling L-shape anchors on identical current-recipe holdouts")
    d32_mean = result["anchors"]["32"]["laws"]["skaling"]["common_holdout_mape"]
    d64_mean = result["anchors"]["64"]["laws"]["skaling"]["common_holdout_mape"]
    bootstrap = result["paired_cell_bootstrap"]["skaling"]
    lower, upper = bootstrap["percentile_95_interval"]
    axis.text(
        0.99,
        0.97,
        f"Mean: d32 {d32_mean:.3f}%  |  d64 {d64_mean:.3f}%\n"
        "Paired-cell bootstrap difference: "
        f'{bootstrap["mape_difference_first_minus_second"]:+.3f} pp '
        f"[{lower:.3f}, {upper:.3f}]",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.legend()
    fig.savefig(HERE / "dense-anchor-comparison.svg")
    fig.savefig(HERE / "dense-anchor-comparison.png", dpi=180)


if __name__ == "__main__":
    main()
