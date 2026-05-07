"""Scaling-law report generation."""

from __future__ import annotations

import argparse
import html
import math
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from chess_engine_4.data.leela import INPUT_PLANE_COUNT, POLICY_SIZE

DEFAULT_BEST_RUNS = Path("experiments/best-runs.toml")
DEFAULT_OUTPUT_ROOT = Path("reports/scaling-laws")
CHARTS = [
    ("Loss fit", "loss.svg"),
    ("Policy top-1", "policy_top1.svg"),
    ("Model size fit", "model_size.svg"),
    ("Data samples fit", "data_samples.svg"),
    ("Batch size fit", "batch_size.svg"),
    ("Learning rate fit", "learning_rate.svg"),
    ("Runtime fit", "runtime.svg"),
]


@dataclass(frozen=True, slots=True)
class SweepResult:
    budget: str
    source_experiment: str
    flops: float
    run_name: str
    batch_size: int
    lr: float
    d_model: int
    depth: int
    params: int
    non_embedding_params: int
    samples_seen: int
    loss_ema: float
    policy_top1: float
    runtime_sec: float
    wandb_url: str


@dataclass(frozen=True, slots=True)
class PowerLaw:
    intercept: float
    slope: float

    def predict(self, x: float) -> float:
        return 10 ** (self.intercept + self.slope * math.log10(x))

    def format(self, variable: str, input_variable: str = "C") -> str:
        return f"{variable} = {10**self.intercept:.4g} * {input_variable}^{self.slope:.4f}"


@dataclass(frozen=True, slots=True)
class LossPowerLaw:
    floor: float
    coefficient: float
    exponent: float
    rmse: float

    def predict(self, x: float) -> float:
        return self.floor + self.coefficient * x ** (-self.exponent)

    def format(self) -> str:
        return f"L(C) = {self.floor:.4f} + {self.coefficient:.4g} * C^-{self.exponent:.4f}"


@dataclass(frozen=True, slots=True)
class ScalingLaws:
    loss: LossPowerLaw
    policy_top1: float
    d_model: PowerLaw
    depth: PowerLaw
    non_embedding_params: PowerLaw
    samples: PowerLaw
    batch_size: PowerLaw
    lr: PowerLaw
    runtime_sec: PowerLaw


@dataclass(frozen=True, slots=True)
class HparamSuggestion:
    flops_target: float
    d_model: int
    depth: int
    batch_size: int
    lr: float
    target_non_embedding_params: int
    actual_non_embedding_params: int
    total_params: int
    samples_seen: int
    runtime_sec: float


@dataclass(frozen=True, slots=True)
class Series:
    label: str
    points: list[tuple[float, float]]
    color: str
    line: bool = True
    markers: bool = True
    dashed: bool = False


def scaling_laws() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-flops", type=float, default=1e16)
    parser.add_argument("--best-runs", type=Path, default=DEFAULT_BEST_RUNS)
    parser.add_argument("--gpu", default="t4")
    parser.add_argument("--config", default="configs/1e15.toml")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-output", action="store_true")
    parser.add_argument("--write-config", type=Path, default=None)
    args = parser.parse_args()

    best_results = read_best_runs(args.best_runs)
    laws = fit_scaling_laws(best_results)
    suggestion = extrapolate(laws, args.target_flops)
    report = format_report(
        best_results=best_results,
        laws=laws,
        suggestion=suggestion,
        config=args.config,
        gpu=args.gpu,
    )
    print(report)

    if not args.no_output:
        output_dir = report_output_dir(args.output_root, args.target_flops)
        write_report_artifacts(
            output_dir=output_dir,
            best_results=best_results,
            laws=laws,
            suggestion=suggestion,
            config=args.config,
            gpu=args.gpu,
        )
        print(f"\nwrote report to {output_dir / 'README.md'}")

    if args.write_config is not None:
        args.write_config.write_text(format_config(suggestion), encoding="utf-8")
        print(f"\nwrote {args.write_config}")


def read_best_runs(path: Path) -> list[SweepResult]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    raw_runs = data.get("runs", {})
    results = [
        SweepResult(
            budget=budget,
            source_experiment=str(row["source_experiment"]),
            flops=float(row["flops"]),
            run_name=str(row["run_name"]),
            batch_size=int(row["batch_size"]),
            lr=float(row["lr"]),
            d_model=int(row["d_model"]),
            depth=int(row["depth"]),
            params=int(row["params"]),
            non_embedding_params=int(row["non_embedding_params"]),
            samples_seen=int(row["samples_seen"]),
            loss_ema=float(row["loss_ema"]),
            policy_top1=float(row["policy_top1"]),
            runtime_sec=float(row["runtime_sec"]),
            wandb_url=str(row["wandb_url"]),
        )
        for budget, row in raw_runs.items()
    ]
    if not results:
        raise ValueError(f"No rows found in {path}.")
    return sorted(results, key=lambda result: result.flops)


def best_results_by_budget(results: Iterable[SweepResult]) -> list[SweepResult]:
    best: dict[str, SweepResult] = {}
    for result in results:
        previous = best.get(result.budget)
        if previous is None or result.loss_ema < previous.loss_ema:
            best[result.budget] = result
    return sorted(best.values(), key=lambda result: result.flops)


def fit_scaling_laws(best_results: list[SweepResult]) -> ScalingLaws:
    if len(best_results) < 2:
        raise ValueError("At least two best-run points are required for extrapolation.")
    return ScalingLaws(
        loss=fit_loss_power_law((r.flops, r.loss_ema) for r in best_results),
        policy_top1=sum(r.policy_top1 for r in best_results) / len(best_results),
        d_model=fit_power_law((r.flops, r.d_model) for r in best_results),
        depth=fit_power_law((r.flops, r.depth) for r in best_results),
        non_embedding_params=fit_power_law(
            (r.flops, r.non_embedding_params) for r in best_results
        ),
        samples=fit_power_law((r.flops, r.samples_seen) for r in best_results),
        batch_size=fit_power_law((r.flops, r.batch_size) for r in best_results),
        lr=fit_power_law((r.flops, r.lr) for r in best_results),
        runtime_sec=fit_power_law((r.flops, r.runtime_sec) for r in best_results),
    )


def extrapolate(laws: ScalingLaws, flops_target: float) -> HparamSuggestion:
    if flops_target <= 0:
        raise ValueError("target FLOPs must be positive.")

    target_non_embedding_params = round(laws.non_embedding_params.predict(flops_target))
    d_model, depth, actual_non_embedding_params = closest_architecture(
        target_non_embedding_params=target_non_embedding_params,
        target_d_model=laws.d_model.predict(flops_target),
        target_depth=laws.depth.predict(flops_target),
    )
    return HparamSuggestion(
        flops_target=flops_target,
        d_model=d_model,
        depth=depth,
        batch_size=round_to_batch_ladder(laws.batch_size.predict(flops_target)),
        lr=round_to_lr_ladder(laws.lr.predict(flops_target)),
        target_non_embedding_params=target_non_embedding_params,
        actual_non_embedding_params=actual_non_embedding_params,
        total_params=parameter_count(d_model=d_model, depth=depth),
        samples_seen=round_to_int(laws.samples.predict(flops_target)),
        runtime_sec=laws.runtime_sec.predict(flops_target),
    )


def fit_power_law(points: Iterable[tuple[float, float]]) -> PowerLaw:
    logs = [(math.log10(x), math.log10(y)) for x, y in points]
    mean_x = sum(x for x, _ in logs) / len(logs)
    mean_y = sum(y for _, y in logs) / len(logs)
    variance_x = sum((x - mean_x) ** 2 for x, _ in logs)
    if variance_x == 0:
        raise ValueError("Cannot fit a power law with identical x values.")
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in logs)
    slope = covariance / variance_x
    intercept = mean_y - slope * mean_x
    return PowerLaw(intercept=intercept, slope=slope)


def fit_loss_power_law(points: Iterable[tuple[float, float]]) -> LossPowerLaw:
    values = list(points)
    losses = [loss for _, loss in values]
    min_loss = min(losses)
    span = max(losses) - min_loss
    lower_floor = max(0.0, min_loss - max(2.0, span * 20))
    upper_floor = min_loss - 1e-6

    best: LossPowerLaw | None = None
    for index in range(1000):
        floor = lower_floor + (upper_floor - lower_floor) * index / 999
        shifted = [(flops, loss - floor) for flops, loss in values]
        if any(loss <= 0 for _, loss in shifted):
            continue
        law = fit_power_law(shifted)
        coefficient = 10**law.intercept
        exponent = -law.slope
        errors = [floor + coefficient * flops ** (-exponent) - loss for flops, loss in values]
        rmse = math.sqrt(sum(error**2 for error in errors) / len(errors))
        candidate = LossPowerLaw(
            floor=floor,
            coefficient=coefficient,
            exponent=exponent,
            rmse=rmse,
        )
        if best is None or candidate.rmse < best.rmse:
            best = candidate

    if best is None:
        raise ValueError("Could not fit loss scaling law.")
    return best


def closest_architecture(
    *,
    target_non_embedding_params: int,
    target_d_model: float | None = None,
    target_depth: float | None = None,
) -> tuple[int, int, int]:
    candidates: list[tuple[float, int, int, int]] = []
    for d_model in range(64, 2049, 64):
        for depth in range(2, 25):
            params = non_embedding_parameter_count(d_model=d_model, depth=depth)
            distance = abs(math.log(params / target_non_embedding_params))
            if target_d_model is not None:
                distance += 0.5 * abs(math.log(d_model / target_d_model))
            if target_depth is not None:
                distance += 0.5 * abs(math.log(depth / target_depth))
            candidates.append((distance, d_model, depth, params))
    _, d_model, depth, params = min(candidates)
    return d_model, depth, params


def non_embedding_parameter_count(*, d_model: int, depth: int, mlp_ratio: float = 4.0) -> int:
    hidden_dim = int(d_model * mlp_ratio)
    return depth * (3 * d_model * hidden_dim + d_model)


def parameter_count(*, d_model: int, depth: int, mlp_ratio: float = 4.0) -> int:
    input_dim = INPUT_PLANE_COUNT * 8 * 8
    hidden_dim = int(d_model * mlp_ratio)
    block_params = depth * (3 * d_model * hidden_dim + d_model)
    input_params = input_dim * d_model + d_model
    final_norm_params = d_model
    policy_params = d_model * POLICY_SIZE + POLICY_SIZE
    wdl_params = d_model * 3 + 3
    moves_left_params = d_model + 1
    return (
        input_params
        + block_params
        + final_norm_params
        + policy_params
        + wdl_params
        + moves_left_params
    )


def round_to_batch_ladder(value: float) -> int:
    ladder = sorted(
        {round(multiplier * 2**power) for power in range(5, 15) for multiplier in (1, 1.5)}
    )
    return min(ladder, key=lambda candidate: abs(math.log(candidate / value)))


def round_to_lr_ladder(value: float) -> float:
    ladder = sorted(
        mantissa * 10**exponent
        for exponent in range(-6, -1)
        for mantissa in (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)
    )
    return min(ladder, key=lambda candidate: abs(math.log(candidate / value)))


def round_to_int(value: float) -> int:
    return int(round(value))


def report_output_dir(output_root: Path, target_flops: float) -> Path:
    return output_root / f"{target_flops:.0e}".replace("+", "")


def write_report_artifacts(
    *,
    output_dir: Path,
    best_results: list[SweepResult],
    laws: ScalingLaws,
    suggestion: HparamSuggestion,
    config: str,
    gpu: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_chart in output_dir.glob("*.svg"):
        stale_chart.unlink()
    write_scaling_charts(
        output_dir=output_dir,
        best_results=best_results,
        laws=laws,
        suggestion=suggestion,
    )
    report = format_report(
        best_results=best_results,
        laws=laws,
        suggestion=suggestion,
        config=config,
        gpu=gpu,
        chart_links=CHARTS,
    )
    (output_dir / "README.md").write_text(report + "\n", encoding="utf-8")


def write_scaling_charts(
    *,
    output_dir: Path,
    best_results: list[SweepResult],
    laws: ScalingLaws,
    suggestion: HparamSuggestion,
) -> None:
    min_flops = min(result.flops for result in best_results)
    max_flops = max(suggestion.flops_target, max(result.flops for result in best_results))
    curve_flops = logspace(min_flops, max_flops, 96)

    write_svg_chart(
        output_dir / "loss.svg",
        title="Loss fit",
        x_label="Training FLOPs",
        y_label="Loss EMA",
        x_log=True,
        y_log=True,
        series=[
            Series(
                "observed best",
                [(result.flops, result.loss_ema) for result in best_results],
                "#1f77b4",
                line=False,
            ),
            Series(
                "fit",
                [(flops, laws.loss.predict(flops)) for flops in curve_flops],
                "#d62728",
                markers=False,
                dashed=True,
            ),
            Series(
                "target",
                [(suggestion.flops_target, laws.loss.predict(suggestion.flops_target))],
                "#2ca02c",
            ),
        ],
    )

    write_svg_chart(
        output_dir / "policy_top1.svg",
        title="Policy top-1",
        x_label="Training FLOPs",
        y_label="Top-1 accuracy",
        x_log=True,
        y_log=False,
        series=[
            Series(
                "observed best",
                [(result.flops, result.policy_top1) for result in best_results],
                "#1f77b4",
                line=False,
            ),
        ],
    )

    write_svg_chart(
        output_dir / "model_size.svg",
        title="Model size fit",
        x_label="Training FLOPs",
        y_label="Non-embedding parameters",
        x_log=True,
        y_log=True,
        series=[
            Series(
                "N observed",
                [(result.flops, result.non_embedding_params) for result in best_results],
                "#9467bd",
                line=False,
            ),
            Series(
                "N fit",
                [(flops, laws.non_embedding_params.predict(flops)) for flops in curve_flops],
                "#9467bd",
                markers=False,
                dashed=True,
            ),
        ],
    )

    write_svg_chart(
        output_dir / "data_samples.svg",
        title="Data samples fit",
        x_label="Training FLOPs",
        y_label="Samples seen",
        x_log=True,
        y_log=True,
        series=[
            Series(
                "D observed",
                [(result.flops, result.samples_seen) for result in best_results],
                "#ff7f0e",
                line=False,
            ),
            Series(
                "D fit",
                [(flops, laws.samples.predict(flops)) for flops in curve_flops],
                "#ff7f0e",
                markers=False,
                dashed=True,
            ),
        ],
    )

    write_svg_chart(
        output_dir / "batch_size.svg",
        title="Batch size fit",
        x_label="Training FLOPs",
        y_label="Batch size",
        x_log=True,
        y_log=True,
        series=[
            Series(
                "batch observed",
                [(result.flops, result.batch_size) for result in best_results],
                "#1f77b4",
                line=False,
            ),
            Series(
                "batch fit",
                [(flops, laws.batch_size.predict(flops)) for flops in curve_flops],
                "#1f77b4",
                markers=False,
                dashed=True,
            ),
            Series("target", [(suggestion.flops_target, suggestion.batch_size)], "#2ca02c"),
        ],
    )

    write_svg_chart(
        output_dir / "learning_rate.svg",
        title="Learning rate fit",
        x_label="Training FLOPs",
        y_label="Learning rate",
        x_log=True,
        y_log=True,
        series=[
            Series(
                "lr observed",
                [(result.flops, result.lr) for result in best_results],
                "#d62728",
                line=False,
            ),
            Series(
                "lr fit",
                [(flops, laws.lr.predict(flops)) for flops in curve_flops],
                "#d62728",
                markers=False,
                dashed=True,
            ),
            Series("target", [(suggestion.flops_target, suggestion.lr)], "#2ca02c"),
        ],
    )

    write_svg_chart(
        output_dir / "runtime.svg",
        title="Runtime fit",
        x_label="Training FLOPs",
        y_label="Runtime seconds",
        x_log=True,
        y_log=True,
        series=[
            Series(
                "runtime observed",
                [(result.flops, result.runtime_sec) for result in best_results],
                "#2ca02c",
                line=False,
            ),
            Series(
                "runtime fit",
                [(flops, laws.runtime_sec.predict(flops)) for flops in curve_flops],
                "#2ca02c",
                markers=False,
                dashed=True,
            ),
        ],
    )


def logspace(start: float, stop: float, count: int) -> list[float]:
    log_start = math.log10(start)
    log_stop = math.log10(stop)
    return [
        10 ** (log_start + (log_stop - log_start) * index / (count - 1))
        for index in range(count)
    ]


def write_svg_chart(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_log: bool,
    y_log: bool,
    series: list[Series],
) -> None:
    width = 920
    height = 560
    left = 92
    right = 28
    top = 58
    bottom = 82
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_points = [point for item in series for point in item.points]
    x_min, x_max = padded_domain([x for x, _ in all_points], log_scale=x_log)
    y_min, y_max = padded_domain([y for _, y in all_points], log_scale=y_log)

    def scale_x(value: float) -> float:
        return left + normalize(value, x_min, x_max, log_scale=x_log) * plot_width

    def scale_y(value: float) -> float:
        return top + (1 - normalize(value, y_min, y_max, log_scale=y_log)) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>"
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#1f2933}"
        ".title{font-size:22px;font-weight:700}"
        ".label{font-size:13px;font-weight:600}"
        ".tick{font-size:11px;fill:#52606d}"
        ".grid{stroke:#e4e7eb;stroke-width:1}"
        ".axis{stroke:#9aa5b1;stroke-width:1.2}"
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{left}" y="34">{html.escape(title)}</text>',
    ]

    for tick in ticks(x_min, x_max, log_scale=x_log):
        x = scale_x(tick)
        parts.append(
            f'<line class="grid" x1="{x:.2f}" y1="{top}" '
            f'x2="{x:.2f}" y2="{top + plot_height}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 22}" '
            f'text-anchor="middle">{html.escape(format_axis_number(tick))}</text>'
        )
    for tick in ticks(y_min, y_max, log_scale=y_log):
        y = scale_y(tick)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" '
            f'x2="{left + plot_width}" y2="{y:.2f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">'
            f"{html.escape(format_axis_number(tick))}</text>"
        )

    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{top + plot_height}"/>',
            f'<text class="label" x="{left + plot_width / 2}" y="{height - 24}" '
            f'text-anchor="middle">{html.escape(x_label)}</text>',
            f'<text class="label" x="20" y="{top + plot_height / 2}" text-anchor="middle" '
            f'transform="rotate(-90 20 {top + plot_height / 2})">{html.escape(y_label)}</text>',
        ]
    )

    legend_x = left + plot_width - 210
    legend_y = top + 10
    for index, item in enumerate(series):
        y = legend_y + index * 22
        dash = ' stroke-dasharray="7 5"' if item.dashed else ""
        parts.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" '
            f'stroke="{item.color}" stroke-width="3"{dash}/>'
        )
        parts.append(
            f'<text class="tick" x="{legend_x + 36}" y="{y + 4}">'
            f"{html.escape(item.label)}</text>"
        )

    for item in series:
        points = [(scale_x(x), scale_y(y)) for x, y in item.points]
        if item.line and len(points) > 1:
            dash = ' stroke-dasharray="7 5"' if item.dashed else ""
            path_points = " ".join(
                ("M" if index == 0 else "L") + f"{x:.2f},{y:.2f}"
                for index, (x, y) in enumerate(points)
            )
            parts.append(
                f'<path d="{path_points}" fill="none" stroke="{item.color}" '
                f'stroke-width="1.4"{dash}/>'
            )
        if item.markers:
            for x, y in points:
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.2" fill="{item.color}"/>'
                )

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def normalize(value: float, minimum: float, maximum: float, *, log_scale: bool) -> float:
    if log_scale:
        value = math.log10(value)
        minimum = math.log10(minimum)
        maximum = math.log10(maximum)
    return (value - minimum) / (maximum - minimum)


def padded_domain(values: list[float], *, log_scale: bool) -> tuple[float, float]:
    minimum = min(values)
    maximum = max(values)
    if minimum <= 0:
        raise ValueError("Chart values must be positive.")
    if minimum == maximum:
        return minimum * 0.9, maximum * 1.1
    if log_scale:
        log_min = math.log10(minimum)
        log_max = math.log10(maximum)
        padding = (log_max - log_min) * 0.08
        return 10 ** (log_min - padding), 10 ** (log_max + padding)
    padding = (maximum - minimum) * 0.12
    return minimum - padding, maximum + padding


def ticks(minimum: float, maximum: float, *, log_scale: bool) -> list[float]:
    if log_scale:
        start = math.floor(math.log10(minimum))
        stop = math.ceil(math.log10(maximum))
        return [10**exponent for exponent in range(start, stop + 1)]
    step = nice_step((maximum - minimum) / 5)
    start = math.ceil(minimum / step) * step
    values: list[float] = []
    value = start
    while value <= maximum:
        values.append(value)
        value += step
    return values


def nice_step(raw_step: float) -> float:
    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / 10**exponent
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * 10**exponent


def format_axis_number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 10_000 or abs(value) < 0.01:
        return f"{value:.0e}".replace("+", "")
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.2g}"
    return f"{value:.1g}"


def format_report(
    *,
    best_results: list[SweepResult],
    laws: ScalingLaws,
    suggestion: HparamSuggestion,
    config: str,
    gpu: str,
    chart_links: list[tuple[str, str]] | None = None,
) -> str:
    lines = [
        "# Hyperparameter Scaling Report",
        "",
        "This is a repo-local extrapolation from the current best W&B runs, not a claim that "
        "the true scaling law is identified. With only three FLOPs budgets, the fitted "
        "curves are useful for choosing the next run and fragile as forecasts.",
        "",
        "## Best Observed EMA Points",
        "",
        "| Budget | FLOPs | Model | Batch | LR | Samples | Loss EMA | Policy Top-1 | W&B |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in best_results:
        lines.append(
            f"| `{result.budget}` | {result.flops:.1e} | d{result.d_model}x{result.depth} | "
            f"{result.batch_size} | {result.lr:g} | {result.samples_seen:,} | "
            f"{result.loss_ema:.4f} | {result.policy_top1:.4f} | {result.wandb_url} |"
        )

    command = (
        f"uv run train-modal --config {config} --flops-target {suggestion.flops_target:.0e} "
        f"--d-model {suggestion.d_model} --depth {suggestion.depth} "
        f"--batch-size {suggestion.batch_size} --lr {suggestion.lr:g} --gpu {gpu}"
    )

    lines.extend(
        [
            "",
            "## Loss Fit",
            "",
            "Fitted form:",
            "",
            f"```text\n{laws.loss.format()}\n```",
            "",
            f"- RMSE on the observed EMA points: `{laws.loss.rmse:.4f}`",
            f"- Predicted loss EMA at `{suggestion.flops_target:.0e}` FLOPs: "
            f"`{laws.loss.predict(suggestion.flops_target):.4f}`",
            "",
        ]
    )
    if chart_links:
        lines.extend(["## Charts", ""])
        for title, filename in chart_links:
            lines.append(f"![{title}]({filename})")
            lines.append("")

    lines.extend(
        [
            "## Model And Data Fits",
            "",
            "These fit model size and data size as power laws of training compute.",
            "",
            f"```text\n{laws.d_model.format('d_model')}\n```",
            "",
            f"```text\n{laws.depth.format('depth')}\n```",
            "",
            f"```text\n{laws.non_embedding_params.format('N_non_embedding')}\n```",
            "",
            f"```text\n{laws.samples.format('D_samples')}\n```",
            "",
            f"- Width exponent: `{laws.d_model.slope:.4f}`",
            f"- Depth exponent: `{laws.depth.slope:.4f}`",
            f"- Model-size exponent: `{laws.non_embedding_params.slope:.4f}`",
            f"- Data-size exponent: `{laws.samples.slope:.4f}`",
            f"- Predicted sample/parameter ratio at target: "
            f"`{suggestion.samples_seen / suggestion.actual_non_embedding_params:.4f}`",
            "",
            "## Training Hyperparameter Fits",
            "",
            f"```text\n{laws.batch_size.format('batch_size')}\n```",
            "",
            f"```text\n{laws.lr.format('lr')}\n```",
            "",
            f"```text\n{laws.runtime_sec.format('runtime_sec')}\n```",
            "",
            "## Extrapolated Target",
            "",
            f"- FLOPs target: `{suggestion.flops_target:.0e}`",
            f"- Model: `d{suggestion.d_model}x{suggestion.depth}`",
            f"- Batch size: `{suggestion.batch_size}`",
            f"- LR: `{suggestion.lr:g}`",
            f"- Target non-embedding params: `{suggestion.target_non_embedding_params:,}`",
            f"- Actual non-embedding params: `{suggestion.actual_non_embedding_params:,}`",
            f"- Total params: `{suggestion.total_params:,}`",
            f"- Estimated samples: `{suggestion.samples_seen:,}`",
            f"- Estimated runtime from prior runs: `{suggestion.runtime_sec / 3600:.2f}h`",
            "",
            "## Launch Command",
            "",
            f"```bash\n{command}\n```",
        ]
    )
    return "\n".join(lines)


def format_config(suggestion: HparamSuggestion) -> str:
    name = f"{suggestion.flops_target:.0e}".replace("+", "")
    return "\n".join(
        [
            "[run]",
            f'name = "{name}"',
            "seed = 1",
            f"flops_target = {suggestion.flops_target:.0e}",
            "log_every = 10",
            'device = "auto"',
            "",
            "[data]",
            f"batch_size = {suggestion.batch_size}",
            "",
            "[model]",
            f"d_model = {suggestion.d_model}",
            f"depth = {suggestion.depth}",
            "mlp_ratio = 4.0",
            "rms_norm_eps = 1e-6",
            "",
            "[optimizer]",
            f"lr = {suggestion.lr:g}",
            "weight_decay = 0.01",
            "",
            "[loss]",
            "policy = 1.0",
            "value = 1.0",
            "moves_left = 1.0",
            "",
        ]
    )
