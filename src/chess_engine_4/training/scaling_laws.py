"""Scaling-law fitting and hyperparameter extrapolation."""

from __future__ import annotations

import argparse
import math
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from chess_engine_4.model import dense_parameter_count

DEFAULT_BEST_RUNS = Path("experiments/best-runs-dense.toml")


@dataclass(frozen=True, slots=True)
class SweepResult:
    budget: str
    model_kind: str
    compute: float
    run_name: str
    batch_size: int
    lr: float
    d_model: int
    depth: int
    params: int
    samples_seen: int
    loss: float
    loss_std: float
    loss_upper_1sd: float
    policy_top1: float
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

    def compute_for_loss(self, loss: float) -> float:
        if loss <= self.floor:
            return math.inf
        return (self.coefficient / (loss - self.floor)) ** (1.0 / self.exponent)

    def format(self) -> str:
        return f"L(C) = {self.floor:.4f} + {self.coefficient:.4g} * C^-{self.exponent:.4f}"


@dataclass(frozen=True, slots=True)
class ScalingLaws:
    model_kind: str
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
    modified_compute: float
    d_model: int
    depth: int
    batch_size: int
    lr: float
    target_params: int
    actual_params: int
    samples_seen: int
    steps: int


def scaling_laws() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-modified-compute", type=float, default=1e24)
    parser.add_argument("--best-runs", type=Path, default=DEFAULT_BEST_RUNS)
    parser.add_argument("--config", default="configs/dense/d128.toml")
    parser.add_argument("--write-config", type=Path, default=None)
    args = parser.parse_args()

    best_results = read_best_runs(args.best_runs)
    laws = fit_scaling_laws(best_results)
    suggestion = extrapolate(laws, args.target_modified_compute)
    gpu = "b200"
    report = format_report(
        best_results=best_results,
        laws=laws,
        suggestion=suggestion,
        config=args.config,
        gpu=gpu,
    )
    print(report)

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
            model_kind=str(row["model_kind"]),
            compute=float(row["compute"]),
            run_name=str(row["run_name"]),
            batch_size=int(row["batch_size"]),
            lr=float(row["lr"]),
            d_model=int(row["d_model"]),
            depth=int(row["depth"]),
            params=int(row["params"]),
            samples_seen=int(row["samples_seen"]),
            loss=float(row["loss"]),
            loss_std=float(row["loss_std"]),
            loss_upper_1sd=float(row["loss_upper_1sd"]),
            policy_top1=float(row["policy_top1"]),
            wandb_url=str(row["wandb_url"]),
        )
        for budget, row in raw_runs.items()
    ]
    if not results:
        raise ValueError(f"No rows found in {path}.")
    return sorted(results, key=lambda result: result.compute)


def fit_scaling_laws(best_results: list[SweepResult]) -> ScalingLaws:
    if len(best_results) < 2:
        raise ValueError("At least two best-run points are required for extrapolation.")
    model_kinds = {result.model_kind for result in best_results}
    if len(model_kinds) != 1:
        raise ValueError("Best-run points must belong to exactly one model family.")
    model_kind = model_kinds.pop()
    parameter_count(model_kind=model_kind, d_model=64, depth=2)
    return ScalingLaws(
        model_kind=model_kind,
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


def extrapolate(laws: ScalingLaws, modified_compute: float) -> HparamSuggestion:
    if modified_compute <= 0:
        raise ValueError("target compute must be positive.")

    target_params = round(laws.params.predict(modified_compute))
    d_model, depth, actual_params = closest_architecture(
        model_kind=laws.model_kind,
        target_params=target_params,
        target_d_model=laws.d_model.predict(modified_compute),
        target_depth=laws.depth.predict(modified_compute),
    )
    return HparamSuggestion(
        model_kind=laws.model_kind,
        modified_compute=modified_compute,
        d_model=d_model,
        depth=depth,
        batch_size=round_to_batch_ladder(laws.batch_size.predict(modified_compute)),
        lr=round_to_lr_ladder(laws.lr.predict(modified_compute)),
        target_params=target_params,
        actual_params=actual_params,
        samples_seen=round_to_int(laws.samples.predict(modified_compute)),
        steps=round_to_int(
            laws.samples.predict(modified_compute)
            / round_to_batch_ladder(laws.batch_size.predict(modified_compute))
        ),
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


def parameter_count(
    *,
    model_kind: str = "dense",
    d_model: int,
    depth: int,
    expansion_ratio: float = 4.0,
) -> int:
    if model_kind == "dense":
        return dense_parameter_count(
            d_model=d_model,
            depth=depth,
            expansion_ratio=expansion_ratio,
        )
    raise ValueError(f"unknown model kind: {model_kind}")


def round_to_batch_ladder(value: float) -> int:
    ladder = sorted(
        {round(multiplier * 2**power) for power in range(5, 21) for multiplier in (1, 1.5)}
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


def format_report(
    *,
    best_results: list[SweepResult],
    laws: ScalingLaws,
    suggestion: HparamSuggestion,
    config: str,
    gpu: str,
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

    command = (
        f"uv run train-modal --config {config} --steps {suggestion.steps} "
        f"--d-model {suggestion.d_model} --depth {suggestion.depth} "
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
            f"- Predicted loss at `{suggestion.modified_compute:.0e}` modified compute: "
            f"`{laws.loss.predict(suggestion.modified_compute):.4f}`",
            "",
        ]
    )
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
            f"- Modified compute: `{suggestion.modified_compute:.0e}`",
            f"- Model: `{format_suggestion_model_label(suggestion)}`",
            f"- GPU type: `{gpu}`",
            f"- Batch size: `{suggestion.batch_size}`",
            f"- LR: `{suggestion.lr:g}`",
            f"- Target params: `{suggestion.target_params:,}`",
            f"- Actual params: `{suggestion.actual_params:,}`",
            f"- Estimated samples: `{suggestion.samples_seen:,}`",
            f"- Steps: `{suggestion.steps:,}`",
            "",
            "## Launch Command",
            "",
            f"```bash\n{command}\n```",
        ]
    )
    return "\n".join(lines)


def format_config(suggestion: HparamSuggestion) -> str:
    name = f"d{suggestion.d_model}"
    model_lines = [
        "[model]",
        f'kind = "{suggestion.model_kind}"',
        f"d_model = {suggestion.d_model}",
        f"depth = {suggestion.depth}",
        "expansion_ratio = 4.0",
        "rms_norm_eps = 1e-6",
    ]
    return "\n".join(
        [
            "[run]",
            f'name = "{name}"',
            "seed = 1",
            f"steps = {suggestion.steps}",
            f"batch_size = {suggestion.batch_size}",
            "",
            "[infra]",
            "dataloader_threads = 4",
            "dataloader_prefetch_per_thread = 2",
            "",
            "[precision]",
            'recipe = "mxfp8"',
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
            "",
        ]
    )


def format_model_label(result: SweepResult) -> str:
    return f"d{result.d_model}"


def format_suggestion_model_label(suggestion: HparamSuggestion) -> str:
    return f"d{suggestion.d_model}"
