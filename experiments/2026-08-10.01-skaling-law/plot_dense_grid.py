"""Plot the current-recipe dense observations used by the primary fit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent


def main() -> None:
    results = json.loads((HERE / "results.json").read_text())
    result = next(
        row
        for row in results
        if row["family"] == "dense"
        and row["minimum_width"] == 64
        and row["minimum_ratio"] is None
        and row["current_only"]
    )
    observations = result["observations"]
    l_shape = set(result["l_shape"]["train"])

    fig, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for row in observations:
        is_l_shape = row["name"] in l_shape
        axis.scatter(
            row["width"],
            row["ratio"],
            s=125 if is_l_shape else 90,
            marker="s" if is_l_shape else "o",
            color="#e07a1f" if is_l_shape else "#2774ae",
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        axis.annotate(
            f'{row["loss"]:.3f}',
            (row["width"], row["ratio"]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axis.scatter([], [], s=90, color="#2774ae", label="Additional full-grid observation")
    axis.scatter([], [], s=125, marker="s", color="#e07a1f", label="d64-anchored L-shape")
    axis.axhspan(0.045, 0.055, color="#777777", alpha=0.10, label="0.05x sensitivity band")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks([64, 128, 256, 512, 1024], ["d64", "d128", "d256", "d512", "d1024"])
    axis.set_yticks(
        [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0],
        ["0.05", "0.1", "0.2", "0.3", "0.5", "0.75", "1", "1.5", "2"],
    )
    axis.set_xlabel("Dense model width")
    axis.set_ylabel("Chinchilla training ratio")
    axis.set_title("Current-recipe dense observations used by the primary Skaling fit")
    axis.text(
        0.01,
        0.01,
        "Labels are final validation-loss EMA. The primary fit uses all 21 points;\n"
        "the 17-point sensitivity fit excludes the shaded 0.05x row.",
        transform=axis.transAxes,
        fontsize=9,
        va="bottom",
        color="#444444",
    )
    axis.grid(True, which="both", color="#dddddd", linewidth=0.7, zorder=0)
    axis.legend(loc="upper right", frameon=True)
    fig.savefig(HERE / "dense-observations.svg")
    fig.savefig(HERE / "dense-observations.png", dpi=180)


if __name__ == "__main__":
    main()
