"""Plot depth efficiency against the pre-experiment FLOPs frontier."""

from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = {
    32: [(1, 0.987), (2, 1.127), (3, 1.149), (4, 1.203), (5, 1.225), (6, 1.280), (8, 1.312)],
    64: [(2, 0.635), (3, 0.718), (4, 0.787), (5, 0.906), (6, 1.015), (7, 1.073), (8, 1.055)],
    128: [
        (2, 0.582),
        (3, 0.812),
        (4, 1.062),
        (5, 1.047),
        (6, 1.108),
        (7, 1.212),
        (8, 1.335),
        (9, 1.270),
        (10, 1.397),
    ],
    256: [(3, 0.942), (4, 1.112), (5, 1.256), (6, 1.239), (7, 1.305), (8, 1.420), (9, 1.423)],
    512: [(4, 0.964), (5, 1.059), (6, 1.172), (7, 1.223), (8, 1.315)],
    1024: [(6, 0.874), (7, 0.894), (8, 0.906)],
}
SELECTED_DEPTHS = {32: 8, 64: 8, 128: 8, 256: 8, 512: 8, 1024: 8}


def main() -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    for width, points in RESULTS.items():
        depths, efficiencies = zip(*points, strict=True)
        (line,) = axis.plot(depths, efficiencies, marker="o", linewidth=1.4, label=f"d{width}")
        if width in SELECTED_DEPTHS:
            selected_depth = SELECTED_DEPTHS[width]
            selected_efficiency = dict(points)[selected_depth]
            axis.scatter(
                [selected_depth],
                [selected_efficiency],
                marker="*",
                s=150,
                color=line.get_color(),
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )

    axis.axhline(1.0, color="#666666", linewidth=1, linestyle="--")
    axis.set_xlabel("Depth")
    axis.set_ylabel("EG_flops versus prior frontier")
    axis.set_title("Dense depth efficiency at 0.2x Chinchilla")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(Path(__file__).with_name("depth-efficiency.svg"))


if __name__ == "__main__":
    main()
