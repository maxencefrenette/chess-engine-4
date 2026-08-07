"""Plot policy Elo ratings from the adaptive tournament."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).parent


def main() -> None:
    results = json.loads((HERE / "results.json").read_text())
    throughput = json.loads((HERE / "backendbench.json").read_text())
    nodes_per_sec = {
        row["name"]: row["nodes_per_sec"]
        for row in throughput["engines"]
    }

    figure, axis = plt.subplots(figsize=(10, 6.4), facecolor="white")
    axis.set_facecolor("white")
    for row in results["ratings"]:
        name = row["name"]
        color = "#2563eb" if name.startswith("dense-") else "#d97706"
        axis.errorbar(
            nodes_per_sec[name],
            row["elo"],
            yerr=row["elo_95ci"],
            fmt="o",
            color=color,
            capsize=3,
            markersize=6,
            linewidth=1.4,
        )
        text_offset = {
            "dense-d32": (-6, 8),
            "dense-d64": (6, -14),
        }.get(name, (6, 5))
        axis.annotate(
            name,
            (nodes_per_sec[name], row["elo"]),
            xytext=text_offset,
            textcoords="offset points",
            fontsize=8.5,
            ha="right" if name == "dense-d32" else "left",
        )

    axis.set_xscale("log")
    axis.set_xlim(8_000, 650_000)
    axis.set_ylim(-750, 1_050)
    axis.set_xlabel("Nodes per second at batch 256")
    axis.set_ylabel("Policy Elo (95% confidence interval)")
    axis.set_title("Policy strength by inference throughput", fontweight="bold")
    axis.grid(alpha=0.2)
    axis.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color="#2563eb", label="Dense"),
            Line2D([], [], marker="o", linestyle="none", color="#d97706", label="LCZero"),
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
