"""Scaling-law fitting and hyperparameter extrapolation."""

from __future__ import annotations

import argparse
import math
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit

from chess_engine_4.model import dense_parameter_count
from chess_engine_4.training.config import TrainingConfig, load_training_config

DEFAULT_BEST_RUNS = Path("experiments/best-runs-dense.toml")
DEFAULT_CONFIG = Path("configs/dense.py")


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
    stale: bool


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
class SigmoidLaw:
    ceiling: float
    slope: float
    midpoint: float
    rmse: float

    def predict(self, x: float) -> float:
        exponent = -self.slope * (math.log10(x) - self.midpoint)
        return self.ceiling / (1.0 + math.exp(exponent))

    def format(self, variable: str, input_variable: str = "log10(C)") -> str:
        return (
            f"{variable} = {self.ceiling:.4g} / "
            f"(1 + exp(-{self.slope:.4g} * ({input_variable} - {self.midpoint:.4g})))"
        )


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
    policy_top1: SigmoidLaw
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    best_results = read_best_runs(args.best_runs)
    laws = fit_scaling_laws(best_results)
    suggestion = extrapolate(laws, args.target_modified_compute, config=args.config)
    gpu = "b200"
    report = format_report(
        best_results=best_results,
        laws=laws,
        suggestion=suggestion,
        config=str(args.config),
        gpu=gpu,
    )
    print(report)


def read_best_runs(path: Path, *, include_stale: bool = False) -> list[SweepResult]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    raw_runs = data.get("runs", {})
    results = []
    for budget, row in raw_runs.items():
        stale = row.get("stale", False)
        if not isinstance(stale, bool):
            raise ValueError(f"{path}: runs.{budget}.stale must be a boolean.")
        if stale and not include_stale:
            continue
        results.append(
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
                stale=stale,
            )
        )
    if not results:
        raise ValueError(f"No current best-run rows found in {path}.")
    return sorted(results, key=lambda result: result.compute)


def fit_scaling_laws(best_results: list[SweepResult]) -> ScalingLaws:
    if len(best_results) < 3:
        raise ValueError("At least three current best-run points are required for extrapolation.")
    model_kinds = {result.model_kind for result in best_results}
    if len(model_kinds) != 1:
        raise ValueError("Best-run points must belong to exactly one model family.")
    model_kind = model_kinds.pop()
    parameter_count(model_kind=model_kind, d_model=64, depth=2)
    return ScalingLaws(
        model_kind=model_kind,
        loss=fit_loss_power_law((r.compute, r.loss) for r in best_results),
        policy_top1=fit_sigmoid_law((r.compute, r.policy_top1) for r in best_results),
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


def extrapolate(
    laws: ScalingLaws,
    modified_compute: float,
    *,
    config: Path = DEFAULT_CONFIG,
) -> HparamSuggestion:
    if modified_compute <= 0:
        raise ValueError("target compute must be positive.")

    target_params = round(laws.params.predict(modified_compute))
    family, actual_params = closest_family_config(
        config=config,
        model_kind=laws.model_kind,
        target_params=target_params,
    )
    return HparamSuggestion(
        model_kind=laws.model_kind,
        modified_compute=modified_compute,
        d_model=family.model.d_model,
        depth=family.model.depth,
        batch_size=family.run.batch_size,
        lr=family.optimizer.lr,
        target_params=target_params,
        actual_params=actual_params,
        samples_seen=family.run.batch_size * family.run.steps,
        steps=family.run.steps,
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


def fit_sigmoid_law(points: Iterable[tuple[float, float]]) -> SigmoidLaw:
    values = [(math.log10(x), y) for x, y in points]
    if len(values) < 3:
        raise ValueError("At least three points are required to fit a sigmoid law.")
    if any(not 0 < y < 1 for _, y in values):
        raise ValueError("Sigmoid-law values must be between zero and one.")

    x_values = np.array([x for x, _ in values])
    y_values = np.array([y for _, y in values])

    def sigmoid(x: np.ndarray, ceiling: float, slope: float, midpoint: float) -> np.ndarray:
        return ceiling * expit(slope * (x - midpoint))

    parameters, _ = curve_fit(
        sigmoid,
        x_values,
        y_values,
        p0=(0.7, 0.4, 15.0),
        bounds=((max(y_values) + 1e-6, 0.0, -np.inf), (1.0, np.inf, np.inf)),
        maxfev=10_000,
    )
    ceiling, slope, midpoint = (float(value) for value in parameters)
    rmse = float(np.sqrt(np.mean(np.square(sigmoid(x_values, *parameters) - y_values))))
    return SigmoidLaw(ceiling=ceiling, slope=slope, midpoint=midpoint, rmse=rmse)


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


def parameter_count(
    *,
    model_kind: str = "dense",
    d_model: int,
    depth: int,
    expansion_ratio: float = 4.0,
    activation: str = "swiglu",
) -> int:
    if model_kind == "dense":
        return dense_parameter_count(
            d_model=d_model,
            depth=depth,
            expansion_ratio=expansion_ratio,
            activation=activation,
        )
    raise ValueError(f"unknown model kind: {model_kind}")


def closest_family_config(
    *,
    config: Path,
    model_kind: str,
    target_params: int,
) -> tuple[TrainingConfig, int]:
    candidates = []
    d_model = 32
    while True:
        family = load_training_config(config, d_model=d_model)
        if family.model.kind != model_kind:
            raise ValueError(
                f"{config} generated model kind {family.model.kind!r}, expected {model_kind!r}."
            )
        params = parameter_count(
            model_kind=model_kind,
            d_model=family.model.d_model,
            depth=family.model.depth,
            expansion_ratio=family.model.expansion_ratio,
            activation=family.model.activation,
        )
        candidates.append((abs(math.log(params / target_params)), family, params))
        if params >= target_params:
            break
        d_model += 32

    _, family, params = min(candidates, key=lambda candidate: candidate[0])
    return family, params


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

    command = f"uv run train-modal --config {config} --d-model {suggestion.d_model}"

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


def format_model_label(result: SweepResult) -> str:
    return f"d{result.d_model}"


def format_suggestion_model_label(suggestion: HparamSuggestion) -> str:
    return f"d{suggestion.d_model}"
