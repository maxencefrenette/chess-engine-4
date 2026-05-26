"""Scaling-law report generation."""

from __future__ import annotations

import argparse
import math
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from chess_engine_4.model import (
    mlp_moe_parameter_count,
    mlp_parameter_count,
    transformer64_parameter_count,
)
from chess_engine_4.training.config import load_training_config

DEFAULT_BEST_RUNS = Path("experiments/best-runs-mlp.toml")
DEFAULT_OUTPUT_ROOT = Path("reports/scaling-laws/mlp")
CHARTS = [
    ("Loss fit", "loss.svg"),
    ("Policy top-1", "policy_top1.svg"),
    ("Model size fit", "model_size.svg"),
    ("Datapoints per parameter", "datapoints_per_parameter.svg"),
    ("Data samples fit", "data_samples.svg"),
    ("Batch size fit", "batch_size.svg"),
    ("Learning rate fit", "learning_rate.svg"),
]


@dataclass(frozen=True, slots=True)
class SweepResult:
    budget: str
    source_experiment: str
    model_kind: str
    compute: float
    run_name: str
    batch_size: int
    lr: float
    d_model: int
    depth: int
    num_heads: int | None
    params: int
    samples_seen: int
    loss: float
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
class LinearLaw:
    intercept: float
    slope: float

    def predict(self, x: float) -> float:
        return self.intercept + self.slope * math.log10(x)

    def format(self, variable: str, input_variable: str = "log10(C)") -> str:
        return f"{variable} = {self.intercept:.4g} + {self.slope:.4g} * {input_variable}"


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
    model_kind: str
    num_heads: PowerLaw | None
    loss: LossPowerLaw
    policy_top1: LinearLaw
    d_model: PowerLaw
    depth: PowerLaw
    params: PowerLaw
    datapoints_per_parameter: LinearLaw
    samples: PowerLaw
    batch_size: PowerLaw
    lr: PowerLaw


@dataclass(frozen=True, slots=True)
class HparamSuggestion:
    model_kind: str
    compute_budget: float
    d_model: int
    depth: int
    num_heads: int | None
    batch_size: int
    lr: float
    target_params: int
    actual_params: int
    samples_seen: int


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
    parser.add_argument("--target-compute-budget", type=float, default=1e16)
    parser.add_argument("--best-runs", type=Path, default=DEFAULT_BEST_RUNS)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--config", default="configs/mlp/1e19.toml")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-output", action="store_true")
    parser.add_argument("--write-config", type=Path, default=None)
    args = parser.parse_args()

    best_results = read_best_runs(args.best_runs)
    laws = fit_scaling_laws(best_results)
    suggestion = extrapolate(laws, args.target_compute_budget)
    gpu = args.gpu or load_training_config(args.config).infra.gpu_type
    report = format_report(
        best_results=best_results,
        laws=laws,
        suggestion=suggestion,
        config=args.config,
        gpu=gpu,
    )
    print(report)

    if not args.no_output:
        output_dir = report_output_dir(args.output_root, args.target_compute_budget)
        write_report_artifacts(
            output_dir=output_dir,
            best_results=best_results,
            laws=laws,
            suggestion=suggestion,
            config=args.config,
            gpu=gpu,
        )
        print(f"\nwrote report to {output_dir / 'README.md'}")

    if args.write_config is not None:
        args.write_config.write_text(format_config(suggestion, gpu=gpu), encoding="utf-8")
        print(f"\nwrote {args.write_config}")


def read_best_runs(path: Path) -> list[SweepResult]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    raw_runs = data.get("runs", {})
    results = [
        SweepResult(
            budget=budget,
            source_experiment=str(row["source_experiment"]),
            model_kind=str(row.get("model_kind", "mlp")),
            compute=float(row["compute"]),
            run_name=str(row["run_name"]),
            batch_size=int(row["batch_size"]),
            lr=float(row["lr"]),
            d_model=int(row["d_model"]),
            depth=int(row["depth"]),
            num_heads=int(row["num_heads"]) if "num_heads" in row else None,
            params=int(row["params"]),
            samples_seen=int(row["samples_seen"]),
            loss=float(row["loss"]),
            policy_top1=float(row["policy_top1"]),
            runtime_sec=float(row["runtime_sec"]),
            wandb_url=str(row["wandb_url"]),
        )
        for budget, row in raw_runs.items()
    ]
    if not results:
        raise ValueError(f"No rows found in {path}.")
    return sorted(results, key=lambda result: result.compute)


def best_results_by_budget(results: Iterable[SweepResult]) -> list[SweepResult]:
    best: dict[str, SweepResult] = {}
    for result in results:
        previous = best.get(result.budget)
        if previous is None or result.loss < previous.loss:
            best[result.budget] = result
    return sorted(best.values(), key=lambda result: result.compute)


def fit_scaling_laws(best_results: list[SweepResult]) -> ScalingLaws:
    if len(best_results) < 2:
        raise ValueError("At least two best-run points are required for extrapolation.")
    model_kinds = {result.model_kind for result in best_results}
    if len(model_kinds) != 1:
        raise ValueError(
            f"Cannot fit one report across multiple model kinds: {sorted(model_kinds)}."
        )
    num_heads = [result.num_heads for result in best_results]
    return ScalingLaws(
        model_kind=next(iter(model_kinds)),
        num_heads=(
            fit_power_law((r.compute, r.num_heads) for r in best_results if r.num_heads)
            if all(value is not None for value in num_heads)
            else None
        ),
        loss=fit_loss_power_law((r.compute, r.loss) for r in best_results),
        policy_top1=fit_linear_law((r.compute, r.policy_top1) for r in best_results),
        d_model=fit_power_law((r.compute, r.d_model) for r in best_results),
        depth=fit_power_law((r.compute, r.depth) for r in best_results),
        params=fit_power_law((r.compute, r.params) for r in best_results),
        datapoints_per_parameter=fit_linear_law(
            (r.compute, r.samples_seen / r.params) for r in best_results
        ),
        samples=fit_power_law((r.compute, r.samples_seen) for r in best_results),
        batch_size=fit_power_law((r.compute, r.batch_size) for r in best_results),
        lr=fit_power_law((r.compute, r.lr) for r in best_results),
    )


def extrapolate(laws: ScalingLaws, compute_budget: float) -> HparamSuggestion:
    if compute_budget <= 0:
        raise ValueError("target compute must be positive.")

    target_params = round(laws.params.predict(compute_budget))
    d_model, depth, actual_params = closest_architecture(
        model_kind=laws.model_kind,
        target_params=target_params,
        target_d_model=laws.d_model.predict(compute_budget),
        target_depth=laws.depth.predict(compute_budget),
    )
    num_heads = None
    if laws.num_heads is not None:
        num_heads = closest_divisor(d_model, laws.num_heads.predict(compute_budget))
    return HparamSuggestion(
        model_kind=laws.model_kind,
        compute_budget=compute_budget,
        d_model=d_model,
        depth=depth,
        num_heads=num_heads,
        batch_size=round_to_batch_ladder(laws.batch_size.predict(compute_budget)),
        lr=round_to_lr_ladder(laws.lr.predict(compute_budget)),
        target_params=target_params,
        actual_params=actual_params,
        samples_seen=round_to_int(laws.samples.predict(compute_budget)),
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


def fit_linear_law(points: Iterable[tuple[float, float]]) -> LinearLaw:
    values = [(math.log10(x), y) for x, y in points]
    mean_x = sum(x for x, _ in values) / len(values)
    mean_y = sum(y for _, y in values) / len(values)
    variance_x = sum((x - mean_x) ** 2 for x, _ in values)
    if variance_x == 0:
        raise ValueError("Cannot fit a linear law with identical x values.")
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in values)
    slope = covariance / variance_x
    intercept = mean_y - slope * mean_x
    return LinearLaw(intercept=intercept, slope=slope)


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
    model_kind: str,
    target_params: int,
    target_d_model: float | None = None,
    target_depth: float | None = None,
) -> tuple[int, int, int]:
    candidates: list[tuple[float, int, int, int]] = []
    for d_model in range(64, 2049, 64):
        for depth in range(2, 25):
            params = parameter_count(model_kind=model_kind, d_model=d_model, depth=depth)
            distance = abs(math.log(params / target_params))
            if target_d_model is not None:
                distance += 0.5 * abs(math.log(d_model / target_d_model))
            if target_depth is not None:
                distance += 0.5 * abs(math.log(depth / target_depth))
            candidates.append((distance, d_model, depth, params))
    _, d_model, depth, params = min(candidates)
    return d_model, depth, params


def closest_divisor(value: int, target: float) -> int:
    divisors = [candidate for candidate in range(1, value + 1) if value % candidate == 0]
    return min(divisors, key=lambda candidate: abs(math.log(candidate / target)))


def parameter_count(
    *,
    model_kind: str = "mlp",
    d_model: int,
    depth: int,
    mlp_ratio: float = 4.0,
) -> int:
    if model_kind == "transformer64":
        return transformer64_parameter_count(d_model=d_model, depth=depth, mlp_ratio=mlp_ratio)
    if model_kind == "mlp_moe":
        return mlp_moe_parameter_count(
            d_model=d_model,
            depth=depth,
            num_experts=16,
            expert_mlp_ratio=2.0,
        )
    if model_kind != "mlp":
        raise ValueError(f"unknown model kind: {model_kind}")
    return mlp_parameter_count(d_model=d_model, depth=depth, mlp_ratio=mlp_ratio)


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


def report_output_dir(output_root: Path, compute_budget: float) -> Path:
    return output_root / f"{compute_budget:.0e}".replace("+", "")


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
    min_compute = min(result.compute for result in best_results)
    max_compute = max(suggestion.compute_budget, max(result.compute for result in best_results))
    curve_compute = logspace(min_compute, max_compute, 96)

    write_svg_chart(
        output_dir / "loss.svg",
        title="Loss fit",
        x_label="Training compute",
        y_label="Loss",
        x_log=True,
        y_log=True,
        plain_y_ticks=True,
        series=[
            Series(
                "observed best",
                [(result.compute, result.loss) for result in best_results],
                "#1f77b4",
                line=False,
            ),
            Series(
                "fit",
                [(compute, laws.loss.predict(compute)) for compute in curve_compute],
                "#d62728",
                markers=False,
                dashed=True,
            ),
            Series(
                "target",
                [(suggestion.compute_budget, laws.loss.predict(suggestion.compute_budget))],
                "#2ca02c",
            ),
        ],
    )

    write_svg_chart(
        output_dir / "policy_top1.svg",
        title="Policy top-1",
        x_label="Training compute",
        y_label="Top-1 accuracy",
        x_log=True,
        y_log=False,
        series=[
            Series(
                "observed best",
                [(result.compute, result.policy_top1) for result in best_results],
                "#1f77b4",
                line=False,
            ),
            Series(
                "fit",
                [(compute, laws.policy_top1.predict(compute)) for compute in curve_compute],
                "#1f77b4",
                markers=False,
                dashed=True,
            ),
        ],
    )

    write_svg_chart(
        output_dir / "model_size.svg",
        title="Model size fit",
        x_label="Training compute",
        y_label="Total parameters",
        x_log=True,
        y_log=True,
        series=[
            Series(
                "params observed",
                [(result.compute, result.params) for result in best_results],
                "#9467bd",
                line=False,
            ),
            Series(
                "params fit",
                [(compute, laws.params.predict(compute)) for compute in curve_compute],
                "#9467bd",
                markers=False,
                dashed=True,
            ),
        ],
    )

    write_svg_chart(
        output_dir / "datapoints_per_parameter.svg",
        title="Datapoints per parameter",
        x_label="Training compute",
        y_label="Datapoints per parameter",
        x_log=True,
        y_log=False,
        series=[
            Series(
                "observed best",
                [
                    (result.compute, result.samples_seen / result.params)
                    for result in best_results
                ],
                "#8c564b",
                line=False,
            ),
            Series(
                "fit",
                [
                    (compute, laws.datapoints_per_parameter.predict(compute))
                    for compute in curve_compute
                ],
                "#8c564b",
                markers=False,
                dashed=True,
            ),
        ],
    )

    write_svg_chart(
        output_dir / "data_samples.svg",
        title="Data samples fit",
        x_label="Training compute",
        y_label="Samples seen",
        x_log=True,
        y_log=True,
        series=[
            Series(
                "D observed",
                [(result.compute, result.samples_seen) for result in best_results],
                "#ff7f0e",
                line=False,
            ),
            Series(
                "D fit",
                [(compute, laws.samples.predict(compute)) for compute in curve_compute],
                "#ff7f0e",
                markers=False,
                dashed=True,
            ),
        ],
    )

    write_svg_chart(
        output_dir / "batch_size.svg",
        title="Batch size fit",
        x_label="Training compute",
        y_label="Batch size",
        x_log=True,
        y_log=True,
        series=[
            Series(
                "batch observed",
                [(result.compute, result.batch_size) for result in best_results],
                "#1f77b4",
                line=False,
            ),
            Series(
                "batch fit",
                [(compute, laws.batch_size.predict(compute)) for compute in curve_compute],
                "#1f77b4",
                markers=False,
                dashed=True,
            ),
            Series("target", [(suggestion.compute_budget, suggestion.batch_size)], "#2ca02c"),
        ],
    )

    write_svg_chart(
        output_dir / "learning_rate.svg",
        title="Learning rate fit",
        x_label="Training compute",
        y_label="Learning rate",
        x_log=True,
        y_log=True,
        series=[
            Series(
                "lr observed",
                [(result.compute, result.lr) for result in best_results],
                "#d62728",
                line=False,
            ),
            Series(
                "lr fit",
                [(compute, laws.lr.predict(compute)) for compute in curve_compute],
                "#d62728",
                markers=False,
                dashed=True,
            ),
            Series("target", [(suggestion.compute_budget, suggestion.lr)], "#2ca02c"),
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
    plain_y_ticks: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9.2, 5.6), layout="constrained")
    for item in series:
        x_values = [x for x, _ in item.points]
        y_values = [y for _, y in item.points]
        if item.line:
            axis.plot(
                x_values,
                y_values,
                color=item.color,
                linewidth=1.4,
                linestyle="--" if item.dashed else "-",
                label=item.label,
            )
        if item.markers:
            axis.scatter(
                x_values,
                y_values,
                color=item.color,
                s=52,
                zorder=3,
                label=item.label if not item.line else None,
            )

    if x_log:
        axis.set_xscale("log")
    if y_log:
        axis.set_yscale("log")
    if plain_y_ticks:
        import matplotlib.ticker as ticker

        y_values = [y for item in series for _, y in item.points]
        y_min = min(y_values)
        y_max = max(y_values)
        tick_values = ticker.MaxNLocator(nbins=6).tick_values(y_min, y_max)
        visible_ticks = [value for value in tick_values if y_min <= value <= y_max]
        axis.yaxis.set_major_locator(ticker.FixedLocator(visible_ticks))
        axis.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        axis.yaxis.set_minor_formatter(ticker.NullFormatter())
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, which="major", color="#e4e7eb", linewidth=0.8)
    axis.grid(True, which="minor", color="#f1f3f5", linewidth=0.5, alpha=0.7)
    axis.legend(loc="best", frameon=False)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.savefig(path, format="svg")
    plt.close(figure)


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
        "the true scaling law is identified. With only a small number of compute budgets, "
        "the fitted curves are useful for choosing the next run and fragile as forecasts.",
        "",
        "## Best Observed Points",
        "",
        "| Budget | Compute | Model | Batch | LR | Samples | Loss | Policy Top-1 | W&B |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in best_results:
        lines.append(
            f"| `{result.budget}` | {result.compute:.1e} | {format_model_label(result)} | "
            f"{result.batch_size} | {result.lr:g} | {result.samples_seen:,} | "
            f"{result.loss:.4f} | {result.policy_top1:.4f} | "
            f"{result.wandb_url} |"
        )

    num_heads_arg = f" --num-heads {suggestion.num_heads}" if suggestion.num_heads else ""
    command = (
        f"uv run train-modal --config {config} --compute-budget {suggestion.compute_budget:.0e} "
        f"--d-model {suggestion.d_model} --depth {suggestion.depth}{num_heads_arg} "
        f"--batch-size {suggestion.batch_size} --lr {suggestion.lr:g}"
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
            f"- RMSE on the observed loss points: `{laws.loss.rmse:.4f}`",
            f"- Predicted loss at `{suggestion.compute_budget:.0e}` compute: "
            f"`{laws.loss.predict(suggestion.compute_budget):.4f}`",
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
            f"```text\n{laws.params.format('params')}\n```",
            "",
            f"```text\n{laws.samples.format('D_samples')}\n```",
            "",
            f"- Width exponent: `{laws.d_model.slope:.4f}`",
            f"- Depth exponent: `{laws.depth.slope:.4f}`",
            f"- Model-size exponent: `{laws.params.slope:.4f}`",
            f"- Data-size exponent: `{laws.samples.slope:.4f}`",
            f"- Predicted sample/parameter ratio at target: "
            f"`{suggestion.samples_seen / suggestion.actual_params:.4f}`",
            "",
            "## Policy And Allocation Fits",
            "",
            f"```text\n{laws.policy_top1.format('policy_top1')}\n```",
            "",
            f"```text\n{laws.datapoints_per_parameter.format('datapoints_per_parameter')}\n```",
            "",
            "## Training Hyperparameter Fits",
            "",
            f"```text\n{laws.batch_size.format('batch_size')}\n```",
            "",
            f"```text\n{laws.lr.format('lr')}\n```",
            "",
            "## Extrapolated Target",
            "",
            f"- Compute budget: `{suggestion.compute_budget:.0e}`",
            f"- Model: `{format_suggestion_model_label(suggestion)}`",
            f"- GPU type: `{gpu}`",
            f"- Batch size: `{suggestion.batch_size}`",
            f"- LR: `{suggestion.lr:g}`",
            f"- Target params: `{suggestion.target_params:,}`",
            f"- Actual params: `{suggestion.actual_params:,}`",
            f"- Estimated samples: `{suggestion.samples_seen:,}`",
            "",
            "## Launch Command",
            "",
            f"```bash\n{command}\n```",
        ]
    )
    return "\n".join(lines)


def format_config(suggestion: HparamSuggestion, *, gpu: str = "l4") -> str:
    name = f"{suggestion.compute_budget:.0e}".replace("+", "")
    model_lines = [
        "[model]",
        f'kind = "{suggestion.model_kind}"',
        f"d_model = {suggestion.d_model}",
        f"depth = {suggestion.depth}",
    ]
    if suggestion.num_heads is not None:
        model_lines.append(f"num_heads = {suggestion.num_heads}")
    if suggestion.model_kind == "mlp_moe":
        model_lines.extend(
            [
                "expert_mlp_ratio = 2.0",
                "num_experts = 16",
                "num_experts_per_token = 2",
                "rms_norm_eps = 1e-6",
            ]
        )
    else:
        model_lines.extend(
            [
                "mlp_ratio = 4.0",
                "rms_norm_eps = 1e-6",
            ]
        )
    if suggestion.model_kind == "transformer64":
        model_lines.extend(
            [
                "learned_square_embeddings = true",
                "",
                "[model.policy]",
                'kind = "attention"',
            ]
        )
    return "\n".join(
        [
            "[run]",
            f'name = "{name}"',
            "seed = 1",
            f"compute_budget = {suggestion.compute_budget:.0e}",
            "",
            "[infra]",
            f'gpu_type = "{gpu}"',
            "",
            "[data]",
            f"batch_size = {suggestion.batch_size}",
            "",
            *model_lines,
            "",
            "[optimizer]",
            f"lr = {suggestion.lr:g}",
            "weight_decay = 0.01",
            "",
            "[loss]",
            "policy = 1.0",
            "value = 1.0",
            "moves_left = 1.0",
            *(["router_aux = 0.01"] if suggestion.model_kind == "mlp_moe" else []),
            "",
        ]
    )


def format_model_label(result: SweepResult) -> str:
    if result.num_heads is None:
        return f"d{result.d_model}x{result.depth}"
    return f"d{result.d_model}x{result.depth}h{result.num_heads}"


def format_suggestion_model_label(suggestion: HparamSuggestion) -> str:
    if suggestion.num_heads is None:
        return f"d{suggestion.d_model}x{suggestion.depth}"
    return f"d{suggestion.d_model}x{suggestion.depth}h{suggestion.num_heads}"
