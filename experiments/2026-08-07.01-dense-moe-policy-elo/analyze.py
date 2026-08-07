"""Plot policy Elo against batch-256 inference throughput."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
COLORS = {"dense": "#2563eb", "moe": "#059669", "lc0": "#d97706"}


def family(name: str) -> str:
    if name.startswith("dense-"):
        return "dense"
    if name.startswith("moe64a2-"):
        return "moe"
    return "lc0"


def main() -> None:
    results = json.loads((HERE / "results.json").read_text())
    throughput = json.loads((HERE / "backendbench.json").read_text())
    nodes_per_sec = {row["name"]: row["nodes_per_sec"] for row in throughput["engines"]}

    figure, axis = plt.subplots(figsize=(11, 7), facecolor="white")
    axis.set_facecolor("white")
    for row in results["ratings"]:
        name = row["name"]
        text_offset = (-6, 5) if name == "dense-d32" else (6, 5)
        if name.startswith("dense-") and name != "dense-d32":
            text_offset = (6, -14)
        axis.errorbar(
            nodes_per_sec[name],
            row["elo"],
            yerr=row["elo_95ci"],
            fmt="o",
            color=COLORS[family(name)],
            capsize=3,
            markersize=6,
            linewidth=1.3,
        )
        axis.annotate(
            name,
            (nodes_per_sec[name], row["elo"]),
            xytext=text_offset,
            textcoords="offset points",
            fontsize=8.2,
            ha="right" if name == "dense-d32" else "left",
        )

    axis.set_xscale("log")
    axis.set_xlim(8_000, 1_000_000)
    axis.set_ylim(-700, 1_050)
    axis.set_xlabel("Nodes per second at batch 256")
    axis.set_ylabel("Policy Elo (95% confidence interval)")
    axis.set_title("Dense and MoE policy strength by inference throughput", fontweight="bold")
    axis.grid(alpha=0.2)
    axis.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color=COLORS["dense"], label="Dense"),
            Line2D([], [], marker="o", linestyle="none", color=COLORS["moe"], label="MoE 64A2"),
            Line2D([], [], marker="o", linestyle="none", color=COLORS["lc0"], label="LCZero"),
        ],
        frameon=False,
        loc="upper right",
    )
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(HERE / "policy-elo.svg", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
