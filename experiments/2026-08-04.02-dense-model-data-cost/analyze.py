"""Reproduce the dense model/data and dollar-allocation analysis."""

from __future__ import annotations

import csv
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, least_squares

from chess_engine_4.training.scaling_laws import (
    UndertrainingLossLaw,
    fit_loss_power_law,
    fit_undertraining_loss_law,
)

HERE = Path(__file__).parent
THROUGHPUT_PATH = HERE.parent / "throughput-dense.toml"
B200_DOLLARS_PER_HOUR = 6.25
RATIO_MIN = 0.005
RATIO_MAX = 2.0
COLORS = plt.get_cmap("tab10")


@dataclass(frozen=True)
class Result:
    d_model: int
    training_ratio: float
    params: int
    samples_seen: int
    batch_size: int
    steps: int
    flops_per_sample: int
    lr: float
    loss: float
    policy_top1: float

    @property
    def flops(self) -> float:
        return self.flops_per_sample * self.samples_seen

    @property
    def one_x_flops(self) -> float:
        return self.flops / self.training_ratio


def main() -> None:
    results = read_results()
    throughput = read_throughput()
    lnd_parameters, lnd_rmse = fit_model_data_law(results)
    undertraining = fit_ratio_law(results)

    plot_model_data_fit(results, lnd_parameters, lnd_rmse)
    plot_budget_curve(results, throughput, undertraining, x_axis="flops")
    plot_budget_curve(results, throughput, undertraining, x_axis="cost")

    print("Model/data fit:")
    print(format_model_data_law(lnd_parameters))
    print(f"RMSE = {lnd_rmse:.6f}")
    print("\nAnchored undertraining fit:")
    print(undertraining.baseline.format())
    print(
        "penalty = "
        f"{undertraining.penalty_coefficient:.6f} * "
        f"(C1 / 1e15)^-{undertraining.compute_exponent:.6f} * "
        f"(r^-{undertraining.ratio_exponent:.6f} - 1)"
    )
    print(f"RMSE = {undertraining.rmse:.6f}")
    print("\nDollar-optimal allocations:")
    for target in (4.0, 3.8, 3.6, 3.4, 3.2, 3.1, 3.0, 2.9, 2.8):
        optimum = optimum_for_loss(target, throughput, undertraining)
        if optimum is not None:
            cost, width, ratio, steps, flops = optimum
            print(
                f"loss={target:.1f} d{width} ratio={ratio:.4f} "
                f"steps={steps:.0f} flops={flops:.3e} cost=${cost:.4f}"
            )


def read_results() -> list[Result]:
    with (HERE / "results.csv").open(newline="", encoding="utf-8") as handle:
        return [
            Result(
                d_model=int(row["d_model"]),
                training_ratio=float(row["training_ratio"]),
                params=int(row["params"]),
                samples_seen=int(row["samples_seen"]),
                batch_size=int(row["batch_size"]),
                steps=int(row["steps"]),
                flops_per_sample=int(row["flops_per_sample"]),
                lr=float(row["lr"]),
                loss=float(row["loss"]),
                policy_top1=float(row["policy_top1"]),
            )
            for row in csv.DictReader(handle)
        ]


def read_throughput() -> dict[int, dict[str, float | int]]:
    with THROUGHPUT_PATH.open("rb") as handle:
        models = tomllib.load(handle)["models"]
    return {int(name.removeprefix("d")): values for name, values in models.items()}


def fit_model_data_law(results: list[Result]) -> tuple[np.ndarray, float]:
    params = np.array([result.params / 1e6 for result in results])
    samples = np.array([result.samples_seen / 1e8 for result in results])
    losses = np.array([result.loss for result in results])

    def residuals(values: np.ndarray) -> np.ndarray:
        floor, model_coefficient, model_exponent, data_coefficient, data_exponent = values
        predictions = (
            floor
            + model_coefficient * params ** (-model_exponent)
            + data_coefficient * samples ** (-data_exponent)
        )
        return predictions - losses

    bounds = ([0.0, 0.0, 0.001, 0.0, 0.001], [3.0, 20.0, 2.0, 20.0, 2.0])
    generator = np.random.default_rng(5)
    best = None
    for _ in range(500):
        initial = [
            generator.uniform(0.1, 2.8),
            10 ** generator.uniform(-2.0, 0.7),
            10 ** generator.uniform(-1.7, 0.0),
            10 ** generator.uniform(-2.0, 0.7),
            10 ** generator.uniform(-1.7, 0.0),
        ]
        candidate = least_squares(residuals, initial, bounds=bounds, max_nfev=100_000)
        if best is None or np.sum(candidate.fun**2) < np.sum(best.fun**2):
            best = candidate
    assert best is not None
    return best.x, float(np.sqrt(np.mean(best.fun**2)))


def predict_model_data(values: np.ndarray, params: np.ndarray, samples: np.ndarray) -> np.ndarray:
    floor, model_coefficient, model_exponent, data_coefficient, data_exponent = values
    return (
        floor
        + model_coefficient * (params / 1e6) ** (-model_exponent)
        + data_coefficient * (samples / 1e8) ** (-data_exponent)
    )


def format_model_data_law(values: np.ndarray) -> str:
    floor, model_coefficient, model_exponent, data_coefficient, data_exponent = values
    return (
        f"L(N,D) = {floor:.4f} + {model_coefficient:.4g} * "
        f"(N/1e6)^-{model_exponent:.4f} + {data_coefficient:.4f} * "
        f"(D/1e8)^-{data_exponent:.4f}"
    )


def fit_ratio_law(results: list[Result]) -> UndertrainingLossLaw:
    baselines = [result for result in results if result.training_ratio == 1.0]
    baseline = fit_loss_power_law((result.flops, result.loss) for result in baselines)
    undertrained = [result for result in results if result.training_ratio < 1.0]
    return fit_undertraining_loss_law(
        baseline,
        ((result.one_x_flops, result.training_ratio, result.loss) for result in undertrained),
    )


def plot_model_data_fit(results: list[Result], values: np.ndarray, rmse: float) -> None:
    observed = np.array([result.loss for result in results])
    predicted = predict_model_data(
        values,
        np.array([result.params for result in results]),
        np.array([result.samples_seen for result in results]),
    )
    figure, axis = plt.subplots(figsize=(8.5, 6.0))
    for index, width in enumerate(sorted({result.d_model for result in results})):
        mask = np.array([result.d_model == width for result in results])
        axis.scatter(predicted[mask], observed[mask], s=45, color=COLORS(index), label=f"d{width}")
    low = min(observed.min(), predicted.min()) - 0.05
    high = max(observed.max(), predicted.max()) + 0.05
    axis.plot([low, high], [low, high], color="#222222", linewidth=1.2, linestyle="--")
    axis.set(xlabel="Fitted loss", ylabel="Observed loss", xlim=(low, high), ylim=(low, high))
    axis.set_title(f"Conventional L(N,D) fit (RMSE {rmse:.3f})")
    axis.grid(alpha=0.2)
    axis.legend(ncols=2, frameon=False)
    figure.tight_layout()
    figure.savefig(HERE / "model-data-fit.svg")
    plt.close(figure)


def plot_budget_curve(
    results: list[Result],
    throughput: dict[int, dict[str, float | int]],
    law: UndertrainingLossLaw,
    *,
    x_axis: str,
) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    all_curves = []
    ratios = np.geomspace(RATIO_MIN, RATIO_MAX, 400)
    widths = sorted(throughput)
    for index, width in enumerate(widths):
        row = throughput[width]
        one_x_flops = float(row["flops_per_sample"]) * float(row["samples_1x"])
        losses = np.array([law.predict(one_x_flops, ratio) for ratio in ratios])
        if x_axis == "flops":
            x_values = one_x_flops * ratios
        else:
            one_x_cost = steady_state_cost(row, float(row["steps_1x"]))
            x_values = one_x_cost * ratios
        visible = losses <= 5.15
        x_values = x_values[visible]
        losses = losses[visible]
        color = COLORS(index % 10)
        axis.plot(x_values, losses, color=color, linewidth=1.3, label=f"d{width}")
        observed = [result for result in results if result.d_model == width]
        observed_x = [
            result.flops if x_axis == "flops" else steady_state_cost(row, result.steps)
            for result in observed
        ]
        axis.scatter(
            observed_x,
            [result.loss for result in observed],
            color=color,
            edgecolor="white",
            linewidth=0.6,
            s=42,
            zorder=3,
        )
        all_curves.append((x_values, losses))

    x_low = min(curve[0][0] for curve in all_curves)
    x_high = max(curve[0][-1] for curve in all_curves)
    envelope_x = np.geomspace(x_low, x_high, 600)
    envelope_y = []
    for budget in envelope_x:
        candidates = []
        for x_values, losses in all_curves:
            if x_values[0] <= budget <= x_values[-1]:
                candidates.append(float(np.interp(math.log(budget), np.log(x_values), losses)))
        envelope_y.append(min(candidates) if candidates else math.nan)
    axis.plot(
        envelope_x,
        envelope_y,
        color="#111111",
        linewidth=2.0,
        linestyle="--",
        label="Predicted lower envelope",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Training FLOPs" if x_axis == "flops" else "Steady-state B200 cost (USD)")
    axis.set_ylabel("Final loss")
    axis.set_ylim(2.65, 5.15)
    axis.set_title(
        "Final loss by training FLOPs"
        if x_axis == "flops"
        else "Final loss by steady-state B200 cost"
    )
    axis.grid(alpha=0.2)
    axis.legend(ncols=1, frameon=False, loc="upper right")
    figure.tight_layout()
    figure.savefig(HERE / f"loss-vs-{x_axis}.svg")
    plt.close(figure)


def steady_state_cost(throughput: dict[str, float | int], steps: float) -> float:
    seconds = steps * float(throughput["measured_wall_ms_per_step"]) / 1000.0
    return seconds * B200_DOLLARS_PER_HOUR / 3600.0


def optimum_for_loss(
    target_loss: float,
    throughput: dict[int, dict[str, float | int]],
    law: UndertrainingLossLaw,
) -> tuple[float, int, float, float, float] | None:
    candidates = []
    for width, row in throughput.items():
        one_x_flops = float(row["flops_per_sample"]) * float(row["samples_1x"])
        if (
            not law.predict(one_x_flops, RATIO_MAX)
            <= target_loss
            <= law.predict(one_x_flops, RATIO_MIN)
        ):
            continue
        ratio = brentq(
            lambda value, flops=one_x_flops: law.predict(flops, value) - target_loss,
            RATIO_MIN,
            RATIO_MAX,
        )
        steps = float(row["steps_1x"]) * ratio
        candidates.append((steady_state_cost(row, steps), width, ratio, steps, one_x_flops * ratio))
    return min(candidates) if candidates else None


if __name__ == "__main__":
    main()
