"""Scaling-law fitting primitives and canonical run loading."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.special import expit
from scipy.stats import qmc

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
class SkalingLaw:
    model_coefficient: float
    data_coefficient: float
    model_exponent: float
    data_exponent: float
    coupling: float
    floor: float
    rmse: float

    def predict(self, params: float, samples: float) -> float:
        if params <= 0 or samples <= 0:
            raise ValueError("Skaling-law inputs must be positive.")
        model_size = params / 1e6
        data = samples / 1e8
        inner = (
            self.model_coefficient * model_size ** (-self.model_exponent)
            + self.data_coefficient * data ** (-self.data_exponent)
        )
        return inner**self.coupling + self.floor

    def format(self) -> str:
        return (
            "L(N,D) = ("
            f"{self.model_coefficient:.4g} * N^-{self.model_exponent:.4f} + "
            f"{self.data_coefficient:.4g} * D^-{self.data_exponent:.4f}"
            f")^{self.coupling:.4f} + {self.floor:.4f}"
        )

    def samples_for_loss(self, params: float, loss: float) -> float:
        """Return samples needed to reach loss at fixed model size."""
        if params <= 0:
            raise ValueError("Skaling-law model size must be positive.")
        if loss <= self.floor:
            return math.inf
        model_size = params / 1e6
        reducible = (loss - self.floor) ** (1.0 / self.coupling)
        data_term = reducible - self.model_coefficient * model_size ** (-self.model_exponent)
        if data_term <= 0:
            return math.inf
        return 1e8 * (self.data_coefficient / data_term) ** (1.0 / self.data_exponent)


def read_scaling_points(
    path: Path = DEFAULT_BEST_RUNS,
) -> list[tuple[int, int, float]]:
    """Read curated Skaling observations from a canonical best-runs file."""
    with path.open("rb") as handle:
        rows = tomllib.load(handle).get("scaling_runs", {})
    if len(rows) < 7:
        raise ValueError(f"{path}: at least seven scaling_runs are required")
    points = []
    families = {str(row.get("model_kind")) for row in rows.values()}
    if len(families) != 1:
        raise ValueError(f"{path}: scaling_runs must contain exactly one model family")
    for name, row in rows.items():
        params = int(row["params"])
        samples = int(row["samples_seen"])
        loss = float(row["loss"])
        if params <= 0 or samples <= 0 or loss <= 0:
            raise ValueError(f"{path}: scaling_runs.{name} has non-positive fit data")
        points.append((params, samples, loss))
    return points


def read_dense_scaling_points(path: Path = DEFAULT_BEST_RUNS) -> list[tuple[int, int, float]]:
    """Backward-compatible dense scaling-point reader."""
    with path.open("rb") as handle:
        rows = tomllib.load(handle).get("scaling_runs", {})
    if rows and {str(row.get("model_kind")) for row in rows.values()} != {"dense"}:
        raise ValueError(f"{path}: scaling_runs are not dense")
    return read_scaling_points(path)


def fit_dense_skaling_law(
    path: Path = DEFAULT_BEST_RUNS,
    *,
    initial: SkalingLaw | None = None,
    restarts: int = 64,
) -> SkalingLaw:
    """Fit the dense Skaling law from canonical best-runs observations."""
    return fit_skaling_law(read_dense_scaling_points(path), initial=initial, restarts=restarts)


def fit_family_skaling_law(
    path: Path,
    *,
    dense_path: Path = DEFAULT_BEST_RUNS,
    initial: SkalingLaw | None = None,
    restarts: int = 64,
) -> SkalingLaw:
    """Fit one canonical family, sharing dense E with sparse families."""
    with path.open("rb") as handle:
        rows = tomllib.load(handle).get("scaling_runs", {})
    families = {str(row.get("model_kind")) for row in rows.values()}
    fixed_floor = None
    if families != {"dense"}:
        fixed_floor = fit_dense_skaling_law(dense_path, restarts=restarts).floor
    return fit_skaling_law(
        read_scaling_points(path),
        initial=initial,
        restarts=restarts,
        fixed_floor=fixed_floor,
    )


def fit_skaling_law(
    points: Iterable[tuple[int, int, float]],
    *,
    initial: SkalingLaw | None = None,
    restarts: int = 64,
    fixed_floor: float | None = None,
) -> SkalingLaw:
    values = list(points)
    if len(values) < 7:
        raise ValueError("At least seven points are required to fit a Skaling law.")
    if restarts <= 0:
        raise ValueError("restarts must be positive")
    if any(params <= 0 or samples <= 0 or loss <= 0 for params, samples, loss in values):
        raise ValueError("Skaling-law fit data must be positive.")

    model_size = np.array([params / 1e6 for params, _, _ in values])
    data = np.array([samples / 1e8 for _, samples, _ in values])
    losses = np.array([loss for _, _, loss in values])
    lower = np.array([-13.8, -13.8, 0.01, 0.01, 0.01])
    upper = np.array([16.2, 16.2, 2.0, 2.0, 2.0])
    if fixed_floor is None:
        lower = np.append(lower, 0.0)
        upper = np.append(upper, 3.0)

    def predictions(parameters: np.ndarray) -> np.ndarray:
        log_a, log_b, alpha, beta, coupling = parameters[:5]
        floor = float(parameters[5]) if fixed_floor is None else fixed_floor
        inner = np.exp(log_a) * model_size**-alpha + np.exp(log_b) * data**-beta
        return inner**coupling + floor

    def residuals(parameters: np.ndarray) -> np.ndarray:
        return np.log(predictions(parameters)) - np.log(losses)

    sampler = qmc.Sobol(d=len(lower), scramble=True, seed=20260810)
    power = math.ceil(math.log2(restarts))
    starts = list(qmc.scale(sampler.random_base2(power), lower, upper)[:restarts])
    if initial is not None:
        values = [
            math.log(initial.model_coefficient),
            math.log(initial.data_coefficient),
            initial.model_exponent,
            initial.data_exponent,
            initial.coupling,
        ]
        if fixed_floor is None:
            values.append(initial.floor)
        starts.insert(0, np.array(values))

    best_parameters = None
    best_objective = math.inf
    for start in starts:
        candidate = least_squares(
            residuals,
            start,
            bounds=(lower, upper),
            loss="huber",
            f_scale=0.05,
            max_nfev=20_000,
        )
        objective = float(np.sum(candidate.fun**2))
        if objective < best_objective:
            best_parameters = candidate.x
            best_objective = objective
    assert best_parameters is not None
    log_a, log_b, alpha, beta, coupling = best_parameters[:5]
    floor = float(best_parameters[5]) if fixed_floor is None else fixed_floor
    rmse = float(np.sqrt(np.mean(np.square(predictions(best_parameters) - losses))))
    return SkalingLaw(
        model_coefficient=math.exp(float(log_a)),
        data_coefficient=math.exp(float(log_b)),
        model_exponent=float(alpha),
        data_exponent=float(beta),
        coupling=float(coupling),
        floor=float(floor),
        rmse=rmse,
    )


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
