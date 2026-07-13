from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chess_engine_4.training.scaling_laws import fit_loss_power_law

HERE = Path(__file__).parent
COLORS = {"swiglu": "#2563eb", "geglu": "#dc2626"}
LABELS = {"swiglu": "SwiGLU", "geglu": "GEGLU"}


def main() -> None:
    with (HERE / "activation-results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    fig, ax = plt.subplots(figsize=(8, 5))
    all_flops = [float(row["flops"]) for row in rows]
    fit_x = np.logspace(np.log10(min(all_flops)), np.log10(max(all_flops)), 300)

    for activation in ("swiglu", "geglu"):
        series = [row for row in rows if row["activation"] == activation]
        flops = np.array([float(row["flops"]) for row in series])
        losses = np.array([float(row["loss"]) for row in series])
        fit = fit_loss_power_law(zip(flops, losses, strict=True))
        color = COLORS[activation]

        ax.scatter(flops, losses, color=color, s=42, label=LABELS[activation], zorder=3)
        ax.plot(fit_x, [fit.predict(value) for value in fit_x], color=color, linewidth=1.25)

    ax.set_xscale("log")
    ax.set_xlabel("Physical training FLOPs")
    ax.set_ylabel("EMA loss")
    ax.set_title("Loss vs training FLOPs")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "loss-vs-flops.svg", format="svg")


if __name__ == "__main__":
    main()
