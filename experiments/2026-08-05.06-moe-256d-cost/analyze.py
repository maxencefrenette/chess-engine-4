"""Plot the MoE 256d realized-cost experiment."""

from __future__ import annotations

import tomllib
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent


def main() -> None:
    with (HERE / "results.toml").open("rb") as handle:
        data = tomllib.load(handle)
    price = float(data["b200_dollars_per_hour"])

    figure, axis = plt.subplots(figsize=(9.5, 6.0))
    colors = {256: "#2563eb", 512: "#16a34a"}
    for width in (256, 512):
        rows = [row for row in data["runs"] if row["d_model"] == width]
        for row in rows:
            marker = "s" if row["incumbent"] else ("o" if row["eligible"] else "x")
            axis.scatter(
                row["runtime_sec"] * price / 3600,
                row["loss"],
                color=colors[width],
                marker=marker,
                s=65,
                linewidth=1.7,
            )

    for width, color in colors.items():
        axis.scatter([], [], color=color, label=f"d{width}")
    axis.scatter([], [], color="#374151", marker="s", label="Incumbent")
    axis.scatter([], [], color="#374151", marker="o", label="Eligible 256d")
    axis.scatter([], [], color="#374151", marker="x", label="Rejected 256d")
    axis.set_xlabel("Realized B200 cost (USD)")
    axis.set_ylabel("Final loss")
    axis.set_title("MoE 256d batch cost comparison")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncols=2)
    figure.tight_layout()
    figure.savefig(HERE / "loss-vs-cost.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()
