from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

ratios = [30, 35, 40, 45, 48, 50]
physical = [0.7816, 0.8737, 0.9516, 0.9801, 0.9816, 1.0041]
invalid = {30, 40, 48}

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ratios, physical, marker="o", linewidth=1.5, label="FLOPs efficiency")
for ratio in invalid:
    index = ratios.index(ratio)
    ax.scatter(
        ratio,
        physical[index],
        marker="x",
        color="#c43c35",
        s=60,
        zorder=4,
    )
ax.axhline(1.0, color="#888888", linewidth=1, linestyle="--")
ax.set_xlabel("Training samples per parameter")
ax.set_ylabel("Geometric mean efficiency")
ax.set_title("Dense data-allocation sweep")
ax.grid(alpha=0.2)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("aggregate-efficiency.svg"))
