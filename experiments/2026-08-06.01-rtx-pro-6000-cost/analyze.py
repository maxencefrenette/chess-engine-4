"""Plot dense training cost on B200 and RTX PRO 6000."""

from __future__ import annotations

import tomllib
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent


def main() -> None:
    with (HERE / "results.toml").open("rb") as handle:
        data = tomllib.load(handle)

    cpu_rate = data["cpu_cores"] * data["cpu_dollars_per_core_second"]
    rates = {
        "B200": data["b200_gpu_dollars_per_second"] + cpu_rate,
        "RTX PRO 6000": data["rtx_pro_6000_gpu_dollars_per_second"] + cpu_rate,
    }
    widths = [row["d_model"] for row in data["models"]]

    figure, axis = plt.subplots(figsize=(9.5, 6.0))
    for label, field, color in (
        ("B200", "b200_ms_per_step", "#2563eb"),
        ("RTX PRO 6000", "rtx_pro_6000_ms_per_step", "#d97706"),
    ):
        costs = [
            row["steps"] * row[field] / 1000 * rates[label]
            for row in data["models"]
        ]
        axis.plot(widths, costs, color=color, marker="o", linewidth=1.6, label=label)

    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(widths, labels=[f"d{width}" for width in widths])
    axis.set_xlabel("Residual width")
    axis.set_ylabel("Estimated Modal cost (USD)")
    axis.set_title("Dense 0.2x Chinchilla training cost")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(HERE / "dense-training-cost.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()

