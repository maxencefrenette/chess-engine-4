"""Scaling-law fitting primitives and canonical run loading."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit

DEFAULT_BEST_RUNS = Path("experiments/best-runs-dense.toml")


@dataclass(frozen=True, slots=True)
class SweepResult:
    budget: str
    model_kind: str
    flops: float
    run_name: str
    training_ratio: float
    batch_size: int
    lr: float
    d_model: int
    depth: int
    params: int
    samples_seen: int
    loss: float
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

    def flops_for_loss(self, loss: float) -> float:
        if loss <= self.floor:
            return math.inf
        return (self.coefficient / (loss - self.floor)) ** (1.0 / self.exponent)

    def format(self) -> str:
        return f"L(C) = {self.floor:.4f} + {self.coefficient:.4g} * C^-{self.exponent:.4f}"


@dataclass(frozen=True, slots=True)
class UndertrainingLossLaw:
    baseline: LossPowerLaw
    penalty_coefficient: float
    compute_exponent: float
    ratio_exponent: float
    rmse: float

    def predict(self, one_x_flops: float, training_ratio: float) -> float:
        if one_x_flops <= 0:
            raise ValueError("1x training FLOPs must be positive.")
        if training_ratio <= 0:
            raise ValueError("training ratio must be positive.")
        penalty = (
            self.penalty_coefficient
            * (one_x_flops / 1e15) ** (-self.compute_exponent)
            * (training_ratio ** (-self.ratio_exponent) - 1.0)
        )
        return self.baseline.predict(one_x_flops) + penalty


def read_best_runs(
    path: Path,
    *,
    include_stale: bool = False,
) -> list[SweepResult]:
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
                flops=float(row["flops"]),
                run_name=str(row["run_name"]),
                training_ratio=float(row.get("training_ratio", 1.0)),
                batch_size=int(row["batch_size"]),
                lr=float(row["lr"]),
                d_model=int(row["d_model"]),
                depth=int(row["depth"]),
                params=int(row["params"]),
                samples_seen=int(row["samples_seen"]),
                loss=float(row["loss"]),
                policy_top1=float(row["policy_top1"]),
                wandb_url=str(row["wandb_url"]),
                stale=stale,
            )
        )
    if not results:
        raise ValueError(f"No current best-run rows found in {path}.")
    return sorted(results, key=lambda result: result.flops)


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


def fit_undertraining_loss_law(
    baseline: LossPowerLaw,
    points: Iterable[tuple[float, float, float]],
) -> UndertrainingLossLaw:
    values = list(points)
    if len(values) < 3:
        raise ValueError("At least three undertrained points are required.")
    if any(one_x_flops <= 0 or ratio <= 0 for one_x_flops, ratio, _ in values):
        raise ValueError("Undertraining-law inputs must be positive.")

    one_x_flops = np.array([flops / 1e15 for flops, _, _ in values])
    training_ratios = np.array([ratio for _, ratio, _ in values])
    penalties = np.array([loss - baseline.predict(flops) for flops, _, loss in values])

    def penalty(
        inputs: tuple[np.ndarray, np.ndarray],
        coefficient: float,
        compute_exponent: float,
        ratio_exponent: float,
    ) -> np.ndarray:
        flops, ratios = inputs
        return coefficient * flops ** (-compute_exponent) * (ratios ** (-ratio_exponent) - 1.0)

    parameters, _ = curve_fit(
        penalty,
        (one_x_flops, training_ratios),
        penalties,
        p0=(0.08, 0.17, 1.4),
        bounds=((0.0, -2.0, 1e-3), (20.0, 2.0, 5.0)),
        maxfev=100_000,
    )
    coefficient, compute_exponent, ratio_exponent = (float(value) for value in parameters)
    rmse = float(
        np.sqrt(
            np.mean(np.square(penalty((one_x_flops, training_ratios), *parameters) - penalties))
        )
    )
    return UndertrainingLossLaw(
        baseline=baseline,
        penalty_coefficient=coefficient,
        compute_exponent=compute_exponent,
        ratio_exponent=ratio_exponent,
        rmse=rmse,
    )
