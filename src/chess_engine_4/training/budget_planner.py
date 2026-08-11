"""Plan the lowest-loss trainable model for dollar budgets."""

from __future__ import annotations

import argparse
import math
import tomllib
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

import numpy as np

from chess_engine_4.hardware import hardware_dollars_per_second
from chess_engine_4.training.families import FAMILY_SPECS, FamilySpec
from chess_engine_4.training.scaling_laws import (
    SkalingLaw,
    UndertrainingLossLaw,
    fit_dense_skaling_law,
    fit_loss_power_law,
    fit_skaling_law,
    fit_undertraining_loss_law,
    read_best_runs,
    read_dense_scaling_points,
)

DEFAULT_DATASET = Path("experiments/training-data.toml")
DEFAULT_RATIO_EXTRAPOLATION_LIMIT = 2.0
DEFAULT_BOOTSTRAP_SAMPLES = 200
DEFAULT_FOCUS_BUDGET = 10.0
MIN_SUGGESTION_STEPS = 1_000
MAX_SUGGESTION_LOSS_GAP = 0.35
UNCERTAINTY_QUANTILES = (0.1, 0.9)
SUGGESTION_RATIOS = (
    0.005,
    0.0075,
    0.01,
    0.015,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.5,
    0.75,
    1.0,
)


@dataclass(frozen=True, slots=True)
class FamilyFit:
    spec: FamilySpec
    law: UndertrainingLossLaw | SkalingLaw
    observed_ratio_min: float
    observed_ratio_max: float
    observed_width_min: int
    observed_width_max: int


@dataclass(frozen=True, slots=True)
class FamilyEvidence:
    spec: FamilySpec
    anchor_points: tuple[tuple[float, float], ...]
    allocation_points: tuple[tuple[float, float, float], ...]
    scaling_points: tuple[tuple[int, int, float], ...]
    observed_coordinates: frozenset[tuple[int, float]]


@dataclass(frozen=True, slots=True)
class BudgetCandidate:
    budget: float
    family: str
    d_model: int
    gpu: str
    batch_size: int
    steps: int
    samples: int
    training_ratio: float
    predicted_loss: float
    estimated_cost: float
    sample_limited: bool
    extrapolated: bool
    config: Path

    @property
    def command(self) -> str:
        return (
            f"uv run train-modal --config {self.config} --d-model {self.d_model} "
            f"--training-ratio {self.training_ratio:.8g} --steps {self.steps}"
        )


@dataclass(frozen=True, slots=True)
class BudgetEstimate:
    candidate: BudgetCandidate
    loss_lower: float
    loss_upper: float
    selection_probability: float


@dataclass(frozen=True, slots=True)
class RunSuggestion:
    family: str
    d_model: int
    gpu: str
    training_ratio: float
    steps: int
    samples: int
    estimated_cost: float
    predicted_loss: float
    loss_lower: float
    loss_upper: float
    direct_expected_loss: float
    pilot_policy_expected_loss: float
    expected_loss_improvement: float
    probability_improves: float
    config: Path

    @property
    def command(self) -> str:
        return (
            f"uv run train-modal --config {self.config} --d-model {self.d_model} "
            f"--training-ratio {self.training_ratio:.8g} --steps {self.steps}"
        )


def budget_planner() -> None:
    parser = argparse.ArgumentParser(
        description="Select the predicted lowest-validation-loss trainable model by budget."
    )
    parser.add_argument("budgets", type=float, nargs="+", help="Dollar budgets to plan.")
    parser.add_argument(
        "--assume-samples",
        type=int,
        default=None,
        help=(
            "Available unique training samples. Defaults to the current corpus count in "
            f"{DEFAULT_DATASET}."
        ),
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--ratio-extrapolation-limit",
        type=float,
        default=DEFAULT_RATIO_EXTRAPOLATION_LIMIT,
        help="Maximum multiplicative extrapolation beyond observed training ratios.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Bootstrap fits used for loss intervals.",
    )
    parser.add_argument(
        "--suggest-runs",
        type=int,
        default=0,
        help="Compare this many preliminary runs by expected final-loss value.",
    )
    parser.add_argument(
        "--focus-budget",
        type=float,
        default=DEFAULT_FOCUS_BUDGET,
        help="Budget breakpoint whose uncertainty suggestions should reduce.",
    )
    parser.add_argument(
        "--max-suggestion-cost",
        type=float,
        default=None,
        help="Maximum steady-state cost per suggested run; defaults to 10%% of focus budget.",
    )
    args = parser.parse_args()

    if any(budget <= 0 for budget in args.budgets):
        parser.error("budgets must be positive")
    assume_samples = (
        read_dataset_samples(args.dataset) if args.assume_samples is None else args.assume_samples
    )
    if assume_samples <= 0:
        parser.error("--assume-samples must be positive")
    if args.ratio_extrapolation_limit < 1:
        parser.error("--ratio-extrapolation-limit must be at least 1")
    if args.bootstrap_samples < 20:
        parser.error("--bootstrap-samples must be at least 20")
    if args.suggest_runs < 0:
        parser.error("--suggest-runs must be non-negative")
    if args.focus_budget <= 0:
        parser.error("--focus-budget must be positive")
    max_suggestion_cost = (
        args.focus_budget * 0.1
        if args.max_suggestion_cost is None
        else args.max_suggestion_cost
    )
    if max_suggestion_cost <= 0:
        parser.error("--max-suggestion-cost must be positive")

    evidence = read_family_evidence(FAMILY_SPECS)
    fits = fit_families(evidence)
    bootstrap_fits = bootstrap_family_fits(
        evidence,
        fits,
        samples=args.bootstrap_samples,
        seed=2026,
    )
    print(f"assumed_samples: {assume_samples:,}")
    print("cost_basis: measured steady-state GPU and CPU time; startup excluded")
    print(
        f"uncertainty_basis: {len(bootstrap_fits)} parametric bootstrap fits; "
        "80% interval"
    )
    print("")
    for budget in args.budgets:
        estimate = estimate_budget(
            budget,
            assume_samples=assume_samples,
            fits=fits,
            bootstrap_fits=bootstrap_fits,
            ratio_extrapolation_limit=args.ratio_extrapolation_limit,
        )
        if estimate is None:
            print(f"${budget:g}: no trainable configuration")
            continue
        candidate = estimate.candidate
        status = []
        if candidate.sample_limited:
            status.append("sample-limited")
        if candidate.extrapolated:
            status.append("ratio-extrapolation")
        suffix = f" ({', '.join(status)})" if status else ""
        print(
            f"${budget:g}: {candidate.family} d{candidate.d_model} on {candidate.gpu}, "
            f"loss={candidate.predicted_loss:.4f} "
            f"[{estimate.loss_lower:.4f}, {estimate.loss_upper:.4f}], "
            f"selection={estimate.selection_probability:.0%}, "
            f"ratio={candidate.training_ratio:.4g}x, "
            f"steps={candidate.steps:,}, samples={candidate.samples:,}, "
            f"cost=${candidate.estimated_cost:.2f}{suffix}"
        )
        print(f"  {candidate.command}")

    if args.suggest_runs:
        suggestions = suggest_runs(
            focus_budget=args.focus_budget,
            count=args.suggest_runs,
            max_cost=max_suggestion_cost,
            assume_samples=assume_samples,
            evidence=evidence,
            fits=fits,
            bootstrap_fits=bootstrap_fits,
            ratio_extrapolation_limit=args.ratio_extrapolation_limit,
        )
        print("")
        print(
            f"value-of-information comparisons for ${args.focus_budget:g} total budget "
            f"(max ${max_suggestion_cost:g} each):"
        )
        if not suggestions:
            print("  no eligible unmeasured run")
        for index, suggestion in enumerate(suggestions, start=1):
            print(
                f"  {index}. {suggestion.family} d{suggestion.d_model}, "
                f"ratio={suggestion.training_ratio:.4g}x, steps={suggestion.steps:,}, "
                f"cost=${suggestion.estimated_cost:.3f}, "
                f"loss={suggestion.predicted_loss:.4f} "
                f"[{suggestion.loss_lower:.4f}, {suggestion.loss_upper:.4f}], "
                f"net_loss_improvement={suggestion.expected_loss_improvement:+.5f}, "
                f"P(improves)={suggestion.probability_improves:.0%}"
            )
            print(
                f"     direct_loss={suggestion.direct_expected_loss:.5f}, "
                f"pilot_then_final_loss={suggestion.pilot_policy_expected_loss:.5f}"
            )
            print(f"     {suggestion.command}")
        if suggestions and suggestions[0].expected_loss_improvement > 0:
            print("  decision: run suggestion 1 before the final training run")
        elif suggestions:
            print("  decision: skip preliminary runs and spend the full budget on final training")


def read_dataset_samples(path: Path) -> int:
    with path.open("rb") as handle:
        value = tomllib.load(handle)["dataset"]["samples"]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path}: dataset.samples must be a positive integer")
    return value


def read_family_evidence(specs: tuple[FamilySpec, ...]) -> list[FamilyEvidence]:
    evidence = []
    for spec in specs:
        try:
            evidence.append(read_one_family_evidence(spec))
        except InsufficientFamilyEvidence:
            continue
    return evidence


class InsufficientFamilyEvidence(ValueError):
    """A family has too few current observations for budget planning."""


def read_one_family_evidence(spec: FamilySpec) -> FamilyEvidence:
    if spec.family == "dense":
        scaling_points = tuple(read_dense_scaling_points(spec.best_runs))
        with spec.best_runs.open("rb") as handle:
            rows = tomllib.load(handle)["scaling_runs"].values()
        return FamilyEvidence(
            spec=spec,
            anchor_points=(),
            allocation_points=(),
            scaling_points=scaling_points,
            observed_coordinates=frozenset(
                (int(row["d_model"]), float(row["training_ratio"])) for row in rows
            ),
        )
    anchor_runs = read_best_runs(spec.best_runs)
    if any(not math.isclose(run.training_ratio, spec.anchor_ratio) for run in anchor_runs):
        raise ValueError(
            f"{spec.best_runs}: active runs must use the {spec.anchor_ratio:g}x anchor ratio"
        )
    with spec.best_runs.open("rb") as handle:
        allocation_runs = list(tomllib.load(handle).get("allocation_runs", {}).values())
    if len(allocation_runs) < 3:
        raise InsufficientFamilyEvidence(
            f"{spec.best_runs}: at least three allocation_runs are required"
        )
    if any(row["model_kind"] != spec.family for row in allocation_runs):
        raise ValueError(f"{spec.best_runs}: allocation_runs contain another model family")
    observed_coordinates = {
        (run.d_model, run.training_ratio) for run in anchor_runs
    } | {
        (int(row["d_model"]), float(row["training_ratio"])) for row in allocation_runs
    }
    return FamilyEvidence(
        spec=spec,
        anchor_points=tuple((run.flops, run.loss) for run in anchor_runs),
        allocation_points=tuple(
            (float(row["flops"]), float(row["training_ratio"]), float(row["loss"]))
            for row in allocation_runs
        ),
        scaling_points=(),
        observed_coordinates=frozenset(observed_coordinates),
    )


def fit_families(
    evidence: list[FamilyEvidence],
    *,
    initial_fits: list[FamilyFit] | None = None,
    skaling_restarts: int = 64,
) -> list[FamilyFit]:
    initial_by_family = (
        {} if initial_fits is None else {fit.spec.family: fit for fit in initial_fits}
    )
    return [
        fit_family(
            item,
            initial=initial_by_family.get(item.spec.family),
            skaling_restarts=skaling_restarts,
        )
        for item in evidence
    ]


def fit_family(
    evidence: FamilyEvidence,
    *,
    initial: FamilyFit | None = None,
    skaling_restarts: int = 64,
) -> FamilyFit:
    spec = evidence.spec
    if spec.family == "dense":
        initial_law = initial.law if initial is not None else None
        if initial_law is not None and not isinstance(initial_law, SkalingLaw):
            raise TypeError("dense initial fit must use SkalingLaw")
        law = (
            _fit_canonical_dense_law(spec.best_runs, skaling_restarts)
            if initial_law is None
            else fit_skaling_law(
                evidence.scaling_points,
                initial=initial_law,
                restarts=skaling_restarts,
            )
        )
        ratios = [ratio for _, ratio in evidence.observed_coordinates]
        return FamilyFit(
            spec=spec,
            law=law,
            observed_ratio_min=min(ratios),
            observed_ratio_max=max(ratios),
            observed_width_min=min(width for width, _ in evidence.observed_coordinates),
            observed_width_max=max(width for width, _ in evidence.observed_coordinates),
        )
    baseline = fit_loss_power_law(evidence.anchor_points)
    law = fit_undertraining_loss_law(
        baseline,
        (
            (
                flops * spec.anchor_ratio / training_ratio,
                training_ratio / spec.anchor_ratio,
                loss,
            )
            for flops, training_ratio, loss in evidence.allocation_points
        ),
    )
    ratios = [row[1] for row in evidence.allocation_points]
    ratios.append(spec.anchor_ratio)
    return FamilyFit(
        spec=spec,
        law=law,
        observed_ratio_min=min(ratios),
        observed_ratio_max=max(ratios),
        observed_width_min=min(width for width, _ in evidence.observed_coordinates),
        observed_width_max=max(width for width, _ in evidence.observed_coordinates),
    )


def bootstrap_family_fits(
    evidence: list[FamilyEvidence],
    central_fits: list[FamilyFit],
    *,
    samples: int,
    seed: int,
) -> list[list[FamilyFit]]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    generator = np.random.default_rng(seed)
    ensembles = []
    attempts = 0
    while len(ensembles) < samples and attempts < samples * 10:
        attempts += 1
        simulated_evidence = [
            _simulate_family_evidence(item, fit, generator)
            for item, fit in zip(evidence, central_fits, strict=True)
        ]
        try:
            ensembles.append(
                fit_families(
                    simulated_evidence,
                    initial_fits=central_fits,
                    skaling_restarts=4,
                )
            )
        except (RuntimeError, ValueError):
            continue
    if len(ensembles) < samples:
        raise RuntimeError(f"only {len(ensembles)} of {samples} bootstrap fits converged")
    return ensembles


def _simulate_family_evidence(
    evidence: FamilyEvidence,
    fit: FamilyFit,
    generator: np.random.Generator,
) -> FamilyEvidence:
    if isinstance(fit.law, SkalingLaw):
        sigma = max(fit.law.rmse, 0.005)
        return FamilyEvidence(
            spec=evidence.spec,
            anchor_points=(),
            allocation_points=(),
            scaling_points=tuple(
                (
                    params,
                    samples,
                    fit.law.predict(params, samples)
                    + float(generator.normal(0.0, sigma)),
                )
                for params, samples, _ in evidence.scaling_points
            ),
            observed_coordinates=evidence.observed_coordinates,
        )
    baseline_sigma = max(fit.law.baseline.rmse, 0.005)
    allocation_sigma = max(fit.law.rmse, 0.005)
    anchor_points = tuple(
        (
            flops,
            fit.law.baseline.predict(flops) + float(generator.normal(0.0, baseline_sigma)),
        )
        for flops, _ in evidence.anchor_points
    )
    allocation_points = []
    for flops, training_ratio, _ in evidence.allocation_points:
        anchor_flops = flops * evidence.spec.anchor_ratio / training_ratio
        relative_ratio = training_ratio / evidence.spec.anchor_ratio
        prediction = fit.law.predict(anchor_flops, relative_ratio)
        allocation_points.append(
            (
                flops,
                training_ratio,
                prediction + float(generator.normal(0.0, allocation_sigma)),
            )
        )
    return FamilyEvidence(
        spec=evidence.spec,
        anchor_points=anchor_points,
        allocation_points=tuple(allocation_points),
        scaling_points=(),
        observed_coordinates=evidence.observed_coordinates,
    )


def estimate_budget(
    budget: float,
    *,
    assume_samples: int,
    fits: list[FamilyFit],
    bootstrap_fits: list[list[FamilyFit]],
    ratio_extrapolation_limit: float = DEFAULT_RATIO_EXTRAPOLATION_LIMIT,
) -> BudgetEstimate | None:
    candidate = plan_budget(
        budget,
        assume_samples=assume_samples,
        fits=fits,
        ratio_extrapolation_limit=ratio_extrapolation_limit,
    )
    if candidate is None:
        return None
    predictions = []
    selected = 0
    for ensemble in bootstrap_fits:
        ensemble_candidates = _all_budget_candidates(
            budget,
            assume_samples=assume_samples,
            fits=ensemble,
            ratio_extrapolation_limit=ratio_extrapolation_limit,
        )
        winner = min(
            ensemble_candidates,
            key=lambda item: item.predicted_loss,
            default=None,
        )
        matching = next(
            (
                item
                for item in ensemble_candidates
                if (item.family, item.d_model) == (candidate.family, candidate.d_model)
            ),
            None,
        )
        if matching is not None:
            predictions.append(matching.predicted_loss)
        if winner is not None and (winner.family, winner.d_model) == (
            candidate.family,
            candidate.d_model,
        ):
            selected += 1
    if not predictions:
        raise RuntimeError("bootstrap fits produced no matching budget candidate")
    lower, upper = np.quantile(predictions, UNCERTAINTY_QUANTILES)
    return BudgetEstimate(
        candidate=candidate,
        loss_lower=float(lower),
        loss_upper=float(upper),
        selection_probability=selected / len(bootstrap_fits),
    )


def plan_budget(
    budget: float,
    *,
    assume_samples: int,
    fits: list[FamilyFit],
    ratio_extrapolation_limit: float = DEFAULT_RATIO_EXTRAPOLATION_LIMIT,
) -> BudgetCandidate | None:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if assume_samples <= 0:
        raise ValueError("assume_samples must be positive")
    if ratio_extrapolation_limit < 1:
        raise ValueError("ratio_extrapolation_limit must be at least 1")
    candidates = _all_budget_candidates(
        budget,
        assume_samples=assume_samples,
        fits=fits,
        ratio_extrapolation_limit=ratio_extrapolation_limit,
    )
    return min(candidates, key=lambda candidate: candidate.predicted_loss, default=None)


def _all_budget_candidates(
    budget: float,
    *,
    assume_samples: int,
    fits: list[FamilyFit],
    ratio_extrapolation_limit: float,
) -> list[BudgetCandidate]:
    return [
        candidate
        for fit in fits
        for candidate in candidates_for_family(
            budget,
            assume_samples=assume_samples,
            fit=fit,
            ratio_extrapolation_limit=ratio_extrapolation_limit,
        )
    ]


def candidates_for_family(
    budget: float,
    *,
    assume_samples: int,
    fit: FamilyFit,
    ratio_extrapolation_limit: float = DEFAULT_RATIO_EXTRAPOLATION_LIMIT,
) -> list[BudgetCandidate]:
    with fit.spec.throughput.open("rb") as handle:
        models = tomllib.load(handle)["models"]
    candidates = []
    for row in models.values():
        if isinstance(fit.law, SkalingLaw) and not (
            fit.observed_width_min <= int(row["d_model"]) <= fit.observed_width_max
        ):
            continue
        batch_size = int(row["batch_size"])
        milliseconds_per_step = float(row["measured_wall_ms_per_step"])
        rate = hardware_dollars_per_second(str(row["gpu"]), int(row["cpu_cores"]))
        budget_steps = math.floor(budget / rate * 1000.0 / milliseconds_per_step)
        sample_steps = assume_samples // batch_size
        steps = min(budget_steps, sample_steps)
        if steps <= 0:
            continue
        samples = steps * batch_size
        steps_1x = int(row["steps_1x"])
        training_ratio = steps / steps_1x
        prediction_ratio_min = fit.observed_ratio_min / ratio_extrapolation_limit
        prediction_ratio_max = fit.observed_ratio_max * ratio_extrapolation_limit
        if not prediction_ratio_min <= training_ratio <= prediction_ratio_max:
            continue
        predicted_loss = _predict_dimensions(row, samples, training_ratio, fit)
        estimated_cost = steps * milliseconds_per_step / 1000.0 * rate
        candidates.append(
            BudgetCandidate(
                budget=budget,
                family=fit.spec.family,
                d_model=int(row["d_model"]),
                gpu=str(row["gpu"]),
                batch_size=batch_size,
                steps=steps,
                samples=samples,
                training_ratio=training_ratio,
                predicted_loss=predicted_loss,
                estimated_cost=estimated_cost,
                sample_limited=sample_steps <= budget_steps,
                extrapolated=not (
                    fit.observed_ratio_min <= training_ratio <= fit.observed_ratio_max
                ),
                config=fit.spec.config,
            )
        )
    return candidates


def suggest_runs(
    *,
    focus_budget: float,
    count: int,
    max_cost: float,
    assume_samples: int,
    evidence: list[FamilyEvidence],
    fits: list[FamilyFit],
    bootstrap_fits: list[list[FamilyFit]],
    ratio_extrapolation_limit: float = DEFAULT_RATIO_EXTRAPOLATION_LIMIT,
) -> list[RunSuggestion]:
    if count <= 0:
        return []
    if focus_budget <= 0 or max_cost <= 0:
        raise ValueError("focus budget and suggestion cost must be positive")
    central_target = plan_budget(
        focus_budget,
        assume_samples=assume_samples,
        fits=fits,
        ratio_extrapolation_limit=ratio_extrapolation_limit,
    )
    if central_target is None:
        return []
    candidates = _suggestion_candidates(
        max_cost=max_cost,
        assume_samples=assume_samples,
        evidence=evidence,
        fits=fits,
        bootstrap_fits=bootstrap_fits,
        ratio_extrapolation_limit=ratio_extrapolation_limit,
        max_predicted_loss=central_target.predicted_loss + MAX_SUGGESTION_LOSS_GAP,
    )
    _, direct_predictions = _budget_action_matrix(
        focus_budget,
        assume_samples=assume_samples,
        fits=fits,
        bootstrap_fits=bootstrap_fits,
        ratio_extrapolation_limit=ratio_extrapolation_limit,
    )
    direct_action = int(np.argmin(np.mean(direct_predictions, axis=0)))
    direct_losses = direct_predictions[:, direct_action]
    direct_expected_loss = float(np.mean(direct_losses))
    evaluated = []
    for candidate, predictions, noise_variance in candidates:
        remaining_budget = focus_budget - candidate.estimated_cost
        if remaining_budget <= 0:
            continue
        _, final_predictions = _budget_action_matrix(
            remaining_budget,
            assume_samples=assume_samples,
            fits=fits,
            bootstrap_fits=bootstrap_fits,
            ratio_extrapolation_limit=ratio_extrapolation_limit,
        )
        pilot_policy_loss, probability_improves = _expected_pilot_policy_loss(
            pilot_predictions=predictions,
            observation_noise_variance=noise_variance,
            final_predictions=final_predictions,
            direct_losses=direct_losses,
        )
        lower, upper = np.quantile(predictions, UNCERTAINTY_QUANTILES)
        evaluated.append(
            RunSuggestion(
                family=candidate.family,
                d_model=candidate.d_model,
                gpu=candidate.gpu,
                training_ratio=candidate.training_ratio,
                steps=candidate.steps,
                samples=candidate.samples,
                estimated_cost=candidate.estimated_cost,
                predicted_loss=candidate.predicted_loss,
                loss_lower=float(lower),
                loss_upper=float(upper),
                direct_expected_loss=direct_expected_loss,
                pilot_policy_expected_loss=pilot_policy_loss,
                expected_loss_improvement=direct_expected_loss - pilot_policy_loss,
                probability_improves=probability_improves,
                config=candidate.config,
            )
        )
    return sorted(
        evaluated,
        key=lambda suggestion: suggestion.expected_loss_improvement,
        reverse=True,
    )[:count]


def _budget_action_matrix(
    budget: float,
    *,
    assume_samples: int,
    fits: list[FamilyFit],
    bootstrap_fits: list[list[FamilyFit]],
    ratio_extrapolation_limit: float,
) -> tuple[list[BudgetCandidate], np.ndarray]:
    candidates = _all_budget_candidates(
        budget,
        assume_samples=assume_samples,
        fits=fits,
        ratio_extrapolation_limit=ratio_extrapolation_limit,
    )
    if not candidates:
        raise RuntimeError(f"no final candidates for ${budget:g} budget")
    ensemble_by_family = [
        {fit.spec.family: fit for fit in ensemble} for ensemble in bootstrap_fits
    ]
    predictions = np.asarray(
        [
            [
                _predict_candidate(candidate, family_fits[candidate.family])
                for candidate in candidates
            ]
            for family_fits in ensemble_by_family
        ],
        dtype=np.float64,
    )
    return candidates, predictions


def _expected_pilot_policy_loss(
    *,
    pilot_predictions: np.ndarray,
    observation_noise_variance: float,
    final_predictions: np.ndarray,
    direct_losses: np.ndarray,
) -> tuple[float, float]:
    if observation_noise_variance <= 0:
        raise ValueError("observation noise variance must be positive")
    quadrature_points = (-math.sqrt(3.0), 0.0, math.sqrt(3.0))
    quadrature_weights = (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)
    noise_std = math.sqrt(observation_noise_variance)
    policy_loss = 0.0
    improvement_probability = 0.0
    prior_weight = 1.0 / len(pilot_predictions)
    for true_index, true_prediction in enumerate(pilot_predictions):
        for point, quadrature_weight in zip(
            quadrature_points,
            quadrature_weights,
            strict=True,
        ):
            observation = true_prediction + noise_std * point
            log_weights = -0.5 * np.square(observation - pilot_predictions) / (
                observation_noise_variance
            )
            log_weights -= np.max(log_weights)
            posterior_weights = np.exp(log_weights)
            posterior_weights /= np.sum(posterior_weights)
            expected_losses = posterior_weights @ final_predictions
            final_action = int(np.argmin(expected_losses))
            realized_loss = float(final_predictions[true_index, final_action])
            scenario_weight = prior_weight * quadrature_weight
            policy_loss += scenario_weight * realized_loss
            if realized_loss < direct_losses[true_index]:
                improvement_probability += scenario_weight
    return policy_loss, improvement_probability


def _suggestion_candidates(
    *,
    max_cost: float,
    assume_samples: int,
    evidence: list[FamilyEvidence],
    fits: list[FamilyFit],
    bootstrap_fits: list[list[FamilyFit]],
    ratio_extrapolation_limit: float,
    max_predicted_loss: float,
) -> list[tuple[BudgetCandidate, np.ndarray, float]]:
    evidence_by_family = {item.spec.family: item for item in evidence}
    ensemble_by_family = [
        {fit.spec.family: fit for fit in ensemble} for ensemble in bootstrap_fits
    ]
    candidates = []
    for fit in fits:
        family_evidence = evidence_by_family[fit.spec.family]
        ratio_min = fit.observed_ratio_min / ratio_extrapolation_limit
        ratio_max = fit.observed_ratio_max * ratio_extrapolation_limit
        for ratio in (value for value in SUGGESTION_RATIOS if ratio_min <= value <= ratio_max):
            for row in _throughput_models(fit.spec).values():
                d_model = int(row["d_model"])
                if isinstance(fit.law, SkalingLaw) and not (
                    fit.observed_width_min <= d_model <= fit.observed_width_max
                ):
                    continue
                if _is_observed(family_evidence, d_model, ratio):
                    continue
                candidate = _candidate_at_ratio(
                    ratio,
                    assume_samples=assume_samples,
                    fit=fit,
                    row=row,
                )
                if (
                    candidate is None
                    or candidate.steps < MIN_SUGGESTION_STEPS
                    or candidate.estimated_cost > max_cost
                    or candidate.predicted_loss > max_predicted_loss
                ):
                    continue
                predictions = np.asarray(
                    [
                        _predict_candidate(candidate, family_fits[fit.spec.family])
                        for family_fits in ensemble_by_family
                    ],
                    dtype=np.float64,
                )
                noise = _fit_noise(fit)
                candidates.append((candidate, predictions, noise**2))
    return candidates


def _candidate_at_ratio(
    training_ratio: float,
    *,
    assume_samples: int,
    fit: FamilyFit,
    row: dict[str, object],
) -> BudgetCandidate | None:
    steps = round(training_ratio * int(row["steps_1x"]))
    batch_size = int(row["batch_size"])
    if steps <= 0 or steps * batch_size > assume_samples:
        return None
    milliseconds_per_step = float(row["measured_wall_ms_per_step"])
    rate = hardware_dollars_per_second(str(row["gpu"]), int(row["cpu_cores"]))
    cost = steps * milliseconds_per_step / 1000.0 * rate
    samples = steps * batch_size
    candidate = BudgetCandidate(
        budget=cost,
        family=fit.spec.family,
        d_model=int(row["d_model"]),
        gpu=str(row["gpu"]),
        batch_size=batch_size,
        steps=steps,
        samples=samples,
        training_ratio=training_ratio,
        predicted_loss=0.0,
        estimated_cost=cost,
        sample_limited=False,
        extrapolated=not (
            fit.observed_ratio_min <= training_ratio <= fit.observed_ratio_max
        ),
        config=fit.spec.config,
    )
    return replace(candidate, predicted_loss=_predict_candidate(candidate, fit))


def _predict_candidate(candidate: BudgetCandidate, fit: FamilyFit) -> float:
    row = _throughput_models(fit.spec)[f"d{candidate.d_model}"]
    return _predict_dimensions(row, candidate.samples, candidate.training_ratio, fit)


def _predict_dimensions(
    row: dict[str, object],
    samples: int,
    training_ratio: float,
    fit: FamilyFit,
) -> float:
    if isinstance(fit.law, SkalingLaw):
        return fit.law.predict(int(row["params"]), samples)
    anchor_flops = (
        float(row["flops_per_sample"]) * int(row["samples_1x"]) * fit.spec.anchor_ratio
    )
    return fit.law.predict(anchor_flops, training_ratio / fit.spec.anchor_ratio)


def _fit_noise(fit: FamilyFit) -> float:
    if isinstance(fit.law, SkalingLaw):
        return max(fit.law.rmse, 0.005)
    return max(fit.law.baseline.rmse, fit.law.rmse, 0.005)


@cache
def _fit_canonical_dense_law(path: Path, restarts: int) -> SkalingLaw:
    return fit_dense_skaling_law(path, restarts=restarts)


@cache
def _throughput_models(spec: FamilySpec) -> dict[str, dict[str, object]]:
    with spec.throughput.open("rb") as handle:
        return tomllib.load(handle)["models"]


def _is_observed(evidence: FamilyEvidence, d_model: int, ratio: float) -> bool:
    return any(
        width == d_model and math.isclose(observed_ratio, ratio, rel_tol=0.01)
        for width, observed_ratio in evidence.observed_coordinates
    )


if __name__ == "__main__":
    budget_planner()
