"""Chart the replicated dense history-length sweep."""

from pathlib import Path

import matplotlib.pyplot as plt

# width, FLOPs, two EMA losses, EG_flops from mean loss, two per-seed EG_flops
RESULTS = {
    8: [
        (32, 9.629393928192e12, (4.0838735641, 4.0911009397), 1.0408, (1.0603, 1.0216)),
        (64, 5.958643951616e13, (3.8052408659, 3.7797394671), 0.8976, (0.8289, 0.9728)),
        (128, 4.63100904849408e14, (3.4849338651, 3.4877493993), 1.0211, (1.0329, 1.0095)),
        (256, 4.542506530111488e15, (3.2310701770, 3.2328849377), 1.1103, (1.1212, 1.0996)),
        (512, 5.3939674389479424e16, (3.0316757732, 3.0303059013), 1.1284, (1.1174, 1.1396)),
    ],
    4: [
        (32, 5.189150982144e12, (4.3169698117, 4.3063612065), 0.6542, (0.6387, 0.6702)),
        (64, 3.6833806815232e13, (3.9199162746, 3.9092489922), 0.6972, (0.6762, 0.7190)),
        (128, 3.31961616285696e14, (3.5253521076, 3.5380881900), 0.9932, (1.0438, 0.9454)),
        (256, 3.696465559027712e15, (3.2473988260, 3.2425733972), 1.1880, (1.1581, 1.2187)),
        (512, 4.7983067468464128e16, (3.0424369774, 3.0423263721), 1.0782, (1.0774, 1.0791)),
    ],
    2: [
        (32, 3.47939241984e12, (4.5251119891, 4.5263913341), 0.3893, (0.3903, 0.3883)),
        (64, 2.7498942107648e13, (4.0413409906, 4.0324001828), 0.4749, (0.4638, 0.4864)),
        (128, 2.7455777857536e14, (3.5727695268, 3.5728144250), 0.8776, (0.8777, 0.8774)),
        (256, 3.306108299771904e15, (3.2813559421, 3.2803594930), 0.9169, (0.9123, 0.9215)),
        (512, 4.51354169131008e16, (3.0507948768, 3.0488311838), 1.0327, (1.0186, 1.0469)),
    ],
}

SINGLE_RUN_D1024 = {
    8: (7.2993951137988608e17, 2.8882412565, 0.8793),
    4: (6.85530384353591296e17, 2.8871285280, 0.9563),
    2: (6.63848432461021184e17, 2.8949856612, 0.8512),
}


def _mean(values: tuple[float, float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    output_dir = Path(__file__).parent

    figure, axis = plt.subplots(figsize=(10, 6))
    for history_length, points in RESULTS.items():
        means = [_mean(point[2]) for point in points]
        errors = [abs(point[2][1] - point[2][0]) / 2 for point in points]
        plot = axis.errorbar(
            [point[1] for point in points],
            means,
            yerr=errors,
            fmt="o",
            capsize=3,
            label=f"History {history_length} (two-run mean)",
        )
        d1024 = SINGLE_RUN_D1024[history_length]
        axis.scatter(
            [d1024[0]],
            [d1024[1]],
            marker="x",
            s=60,
            color=plot[0].get_color(),
        )
    axis.set_xscale("log")
    axis.set_xlabel("Training FLOPs")
    axis.set_ylabel("Loss")
    axis.set_title("Dense history-length sweep at 0.2x Chinchilla")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss-vs-flops.svg")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    offsets = {8: 0.94, 4: 1.0, 2: 1.06}
    for history_length, points in RESULTS.items():
        means = [point[3] for point in points]
        lower_errors = [point[3] - min(point[4]) for point in points]
        upper_errors = [max(point[4]) - point[3] for point in points]
        plot = axis.errorbar(
            [point[0] * offsets[history_length] for point in points],
            means,
            yerr=[lower_errors, upper_errors],
            fmt="o",
            capsize=3,
            label=f"History {history_length} (two-run mean)",
        )
        d1024 = SINGLE_RUN_D1024[history_length]
        axis.scatter(
            [1024 * offsets[history_length]],
            [d1024[2]],
            marker="x",
            s=60,
            color=plot[0].get_color(),
        )
    axis.axhline(1.0, color="#666666", linewidth=1, linestyle="--")
    axis.set_xscale("log", base=2)
    widths = [32, 64, 128, 256, 512, 1024]
    axis.set_xticks(widths, labels=[f"d{width}" for width in widths])
    axis.set_xlabel("Residual width")
    axis.set_ylabel("EG_flops")
    axis.set_title("Training-FLOPs efficiency by history length")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "eg-flops.svg")


if __name__ == "__main__":
    main()
