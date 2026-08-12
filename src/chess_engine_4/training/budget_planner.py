"""Plan the lowest-loss trainable model for dollar budgets."""

from __future__ import annotations

import argparse
import math
import tomllib
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from chess_engine_4.hardware import hardware_dollars_per_second
from chess_engine_4.model import DenseChessNetConfig, dense_parameter_count
from chess_engine_4.training.config import TrainingConfig, load_training_config
from chess_engine_4.training.families import FAMILIES, FamilySpec
from chess_engine_4.training.flops import measure_training_flops_per_sample
from chess_engine_4.training.scaling_laws import (
    SkalingLaw,
    fit_skaling_law,
    read_scaling_points,
)

DEFAULT_DATASET = Path("experiments/training-data.toml")
DEFAULT_ASSUMED_SAMPLES = 25_000_000_000
DEFAULT_FAMILIES = ("dense",)
DEFAULT_RATIO_EXTRAPOLATION_LIMIT = 2.0
DEFAULT_MAX_D_MODEL = 2560
DEFAULT_BOOTSTRAP_SAMPLES = 200
DEFAULT_FOCUS_BUDGET = 10.0
MIN_SUGGESTION_STEPS = 1_000
MAX_SUGGESTION_LOSS_GAP = 0.35
UNCERTAINTY_QUANTILES = (0.1, 0.9)
DENSE_EXTRAPOLATION_STEP = 256
DENSE_SAMPLES_PER_PARAMETER = 50.0
DENSE_MINIMUM_STEPS_COEFFICIENT = 62.7575303963433
DENSE_MINIMUM_STEPS_WIDTH_EXPONENT = 0.8073049254601639
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
    law: SkalingLaw
    observed_ratio_min: float
    observed_ratio_max: float
    observed_width_min: int
    observed_width_max: int


@dataclass(frozen=True, slots=True)
class FamilyEvidence:
    spec: FamilySpec
    scaling_points: tuple[tuple[int, int, float], ...]
    observed_coordinates: frozenset[tuple[int, float]]


@dataclass(frozen=True, slots=True)
class BudgetCandidate:
    budget: float
    family: str
    d_model: int
    params: int
    gpu: str
    batch_size: int
    steps: int
    samples: int
    training_ratio: float
    predicted_loss: float
    estimated_cost: float
    sample_limited: bool
    extrapolated: bool
    width_extrapolated: bool
    config: Path

    @property
    def command(self) -> str:
        return (
            f"uv run train-modal --config {self.config} --d-model {self.d_model} "
            f"--training-ratio {self.training_ratio:.8g}"
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
    width_extrapolated: bool
    config: Path

    @property
    def command(self) -> str:
        return (
            f"uv run train-modal --config {self.config} --d-model {self.d_model} "
            f"--training-ratio {self.training_ratio:.8g}"
        )


@dataclass(frozen=True, slots=True)
class PlanningRun:
    steps: int
    batch_size: int
    training_ratio: float


@dataclass(frozen=True, slots=True)
class PlanningConfig:
    run: PlanningRun


def budget_planner() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the predicted lowest-validation-loss model by budget."
    )
    parser.add_argument("budgets", type=float, nargs="+", help="Dollar budgets to plan.")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=tuple(FAMILIES),
        default=list(DEFAULT_FAMILIES),
        help="Model families to consider; defaults to dense only.",
    )
    parser.add_argument(
        "--assume-samples",
        type=int,
        default=None,
        help=(
            f"Available unique training samples. Defaults to {DEFAULT_ASSUMED_SAMPLES:,}."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=f"Read the sample budget from a dataset TOML, such as {DEFAULT_DATASET}.",
    )
    parser.add_argument(
        "--ratio-extrapolation-limit",
        type=float,
        default=DEFAULT_RATIO_EXTRAPOLATION_LIMIT,
        help="Maximum multiplicative extrapolation beyond observed training ratios.",
    )
    parser.add_argument(
        "--max-d-model",
        type=int,
        default=DEFAULT_MAX_D_MODEL,
        help="Maximum residual width to consider, including extrapolated dense widths.",
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
    if "moe64a2" in args.families and "dense" not in args.families:
        parser.error("moe64a2 planning requires dense so the families share a loss floor")
    if args.assume_samples is not None:
        assume_samples = args.assume_samples
    elif args.dataset is not None:
        assume_samples = read_dataset_samples(args.dataset)
    else:
        assume_samples = DEFAULT_ASSUMED_SAMPLES
    if assume_samples <= 0:
        parser.error("--assume-samples must be positive")
    if args.ratio_extrapolation_limit < 1:
        parser.error("--ratio-extrapolation-limit must be at least 1")
    if args.max_d_model <= 0:
        parser.error("--max-d-model must be positive")
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

    specs = tuple(FAMILIES[family] for family in args.families)
    evidence = read_family_evidence(specs)
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
        "width_extrapolation_basis: largest measured dense-width MFU; "
        "wall time scales with FLOPs per step"
    )
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
            max_d_model=args.max_d_model,
        )
        if estimate is None:
            print(f"${budget:g}: no eligible configuration")
            continue
        candidate = estimate.candidate
        status = []
        if candidate.sample_limited:
            status.append("sample-limited")
        if candidate.extrapolated:
            status.append("ratio-extrapolation")
        if candidate.width_extrapolated:
            status.append("width-extrapolation")
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
            max_d_model=args.max_d_model,
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
    with spec.best_runs.open("rb") as handle:
        rows = tomllib.load(handle).get("scaling_runs", {})
    if len(rows) < 7:
        raise InsufficientFamilyEvidence(f"{spec.best_runs}: at least seven scaling_runs required")
    if any(row.get("model_kind") != spec.family for row in rows.values()):
        raise ValueError(f"{spec.best_runs}: scaling_runs contain another model family")
    return FamilyEvidence(
        spec=spec,
        scaling_points=tuple(read_scaling_points(spec.best_runs)),
        observed_coordinates=frozenset(
            (int(row["d_model"]), float(row["training_ratio"]))
            for row in rows.values()
        ),
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
    fits = []
    dense_floor = None
    for item in evidence:
        fit = fit_family(
            item,
            initial=initial_by_family.get(item.spec.family),
            skaling_restarts=skaling_restarts,
            fixed_floor=None if item.spec.family == "dense" else dense_floor,
        )
        fits.append(fit)
        if item.spec.family == "dense":
            dense_floor = fit.law.floor
    return fits


def fit_family(
    evidence: FamilyEvidence,
    *,
    initial: FamilyFit | None = None,
    skaling_restarts: int = 64,
    fixed_floor: float | None = None,
) -> FamilyFit:
    spec = evidence.spec
    law = fit_skaling_law(
        evidence.scaling_points,
        initial=initial.law if initial is not None else None,
        restarts=skaling_restarts,
        fixed_floor=fixed_floor,
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
    sigma = max(fit.law.rmse, 0.005)
    return FamilyEvidence(
        spec=evidence.spec,
        scaling_points=tuple(
            (
                params,
                samples,
                fit.law.predict(params, samples) + float(generator.normal(0.0, sigma)),
            )
            for params, samples, _ in evidence.scaling_points
        ),
        observed_coordinates=evidence.observed_coordinates,
    )


def estimate_budget(
    budget: float,
    *,
    assume_samples: int,
    fits: list[FamilyFit],
    bootstrap_fits: list[list[FamilyFit]],
    ratio_extrapolation_limit: float = DEFAULT_RATIO_EXTRAPOLATION_LIMIT,
    max_d_model: int = DEFAULT_MAX_D_MODEL,
) -> BudgetEstimate | None:
    candidate = plan_budget(
        budget,
        assume_samples=assume_samples,
        fits=fits,
        ratio_extrapolation_limit=ratio_extrapolation_limit,
        max_d_model=max_d_model,
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
            max_d_model=max_d_model,
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
                if (item.family, item.d_model, item.batch_size)
                == (candidate.family, candidate.d_model, candidate.batch_size)
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
    max_d_model: int = DEFAULT_MAX_D_MODEL,
) -> BudgetCandidate | None:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if assume_samples <= 0:
        raise ValueError("assume_samples must be positive")
    if ratio_extrapolation_limit < 1:
        raise ValueError("ratio_extrapolation_limit must be at least 1")
    if max_d_model <= 0:
        raise ValueError("max_d_model must be positive")
    candidates = _all_budget_candidates(
        budget,
        assume_samples=assume_samples,
        fits=fits,
        ratio_extrapolation_limit=ratio_extrapolation_limit,
        max_d_model=max_d_model,
    )
    return min(candidates, key=lambda candidate: candidate.predicted_loss) if candidates else None


def _all_budget_candidates(
    budget: float,
    *,
    assume_samples: int,
    fits: list[FamilyFit],
    ratio_extrapolation_limit: float,
    max_d_model: int,
) -> list[BudgetCandidate]:
    return [
        candidate
        for fit in fits
        for candidate in candidates_for_family(
            budget,
            assume_samples=assume_samples,
            fit=fit,
            ratio_extrapolation_limit=ratio_extrapolation_limit,
            max_d_model=max_d_model,
        )
    ]


def candidates_for_family(
    budget: float,
    *,
    assume_samples: int,
    fit: FamilyFit,
    ratio_extrapolation_limit: float = DEFAULT_RATIO_EXTRAPOLATION_LIMIT,
    max_d_model: int = DEFAULT_MAX_D_MODEL,
) -> list[BudgetCandidate]:
    models = _planning_models(fit.spec, max_d_model=max_d_model)
    candidates = []
    for (d_model, _), profile in _planning_profiles(
        fit.spec, max_d_model=max_d_model
    ).items():
        row = models.get(f"d{d_model}")
        if row is None:
            continue
        if d_model < fit.observed_width_min or d_model > max_d_model:
            continue
        prediction_ratio_min = fit.observed_ratio_min / ratio_extrapolation_limit
        prediction_ratio_max = fit.observed_ratio_max * ratio_extrapolation_limit
        candidate = _candidate_for_profile_budget(
            budget,
            assume_samples=assume_samples,
            fit=fit,
            row=row,
            profile=profile,
            ratio_min=prediction_ratio_min,
            ratio_max=prediction_ratio_max,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _candidate_for_profile_budget(
    budget: float,
    *,
    assume_samples: int,
    fit: FamilyFit,
    row: dict[str, Any],
    profile: dict[str, Any],
    ratio_min: float,
    ratio_max: float,
) -> BudgetCandidate | None:
    batch_size = int(profile["batch_size"])
    milliseconds_per_step = float(profile["measured_wall_ms_per_step"])
    rate = hardware_dollars_per_second(str(profile["gpu"]), int(profile["cpu_cores"]))
    steps_1x = int(profile["steps_1x"])
    samples_1x = int(profile["samples_1x"])
    ratio_cap = min(
        ratio_max,
        assume_samples / samples_1x,
        budget / (steps_1x * milliseconds_per_step / 1000.0 * rate),
    )
    if ratio_cap < ratio_min:
        return None

    config = _highest_matching_recipe_config(
        fit.spec,
        d_model=int(row["d_model"]),
        batch_size=batch_size,
        ratio_min=ratio_min,
        ratio_max=ratio_cap,
    )
    if config is None:
        return None
    steps = config.run.steps
    samples = steps * batch_size
    estimated_cost = steps * milliseconds_per_step / 1000.0 * rate
    if estimated_cost > budget * (1.0 + 1e-12) or samples > assume_samples:
        return None
    training_ratio = config.run.training_ratio
    return BudgetCandidate(
        budget=budget,
        family=fit.spec.family,
        d_model=int(row["d_model"]),
        params=int(row["params"]),
        gpu=str(profile["gpu"]),
        batch_size=batch_size,
        steps=steps,
        samples=samples,
        training_ratio=training_ratio,
        predicted_loss=fit.law.predict(int(row["params"]), samples),
        estimated_cost=estimated_cost,
        sample_limited=math.isclose(ratio_cap, assume_samples / samples_1x),
        extrapolated=not (
            fit.observed_ratio_min <= training_ratio <= fit.observed_ratio_max
        ),
        width_extrapolated=int(row["d_model"]) > fit.observed_width_max,
        config=fit.spec.config,
    )


def _highest_matching_recipe_config(
    spec: FamilySpec,
    *,
    d_model: int,
    batch_size: int,
    ratio_min: float,
    ratio_max: float,
):
    # Batch-selection regions are contiguous. A short descending grid locates
    # the highest valid region; bisection then recovers its upper edge.
    previous_ratio = ratio_max
    for ratio in np.geomspace(ratio_max, ratio_min, num=129):
        config = _planning_config(
            spec,
            d_model=d_model,
            training_ratio=float(ratio),
        )
        if config is None:
            previous_ratio = float(ratio)
            continue
        if config.run.batch_size != batch_size:
            previous_ratio = float(ratio)
            continue
        lower = float(ratio)
        upper = previous_ratio
        for _ in range(32):
            midpoint = (lower + upper) / 2.0
            candidate = _planning_config(
                spec,
                d_model=d_model,
                training_ratio=midpoint,
            )
            if candidate is None:
                upper = midpoint
                continue
            if candidate.run.batch_size == batch_size:
                lower = midpoint
                config = candidate
            else:
                upper = midpoint
        return config
    return None


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
    max_d_model: int = DEFAULT_MAX_D_MODEL,
    allowed_coordinates: frozenset[tuple[int, float]] | None = None,
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
        max_d_model=max_d_model,
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
        max_d_model=max_d_model,
        max_predicted_loss=central_target.predicted_loss + MAX_SUGGESTION_LOSS_GAP,
        allowed_coordinates=allowed_coordinates,
    )
    _, direct_predictions = _budget_action_matrix(
        focus_budget,
        assume_samples=assume_samples,
        fits=fits,
        bootstrap_fits=bootstrap_fits,
        ratio_extrapolation_limit=ratio_extrapolation_limit,
        max_d_model=max_d_model,
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
            max_d_model=max_d_model,
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
                width_extrapolated=candidate.width_extrapolated,
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
    max_d_model: int,
) -> tuple[list[BudgetCandidate], np.ndarray]:
    candidates = _all_budget_candidates(
        budget,
        assume_samples=assume_samples,
        fits=fits,
        ratio_extrapolation_limit=ratio_extrapolation_limit,
        max_d_model=max_d_model,
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
    max_d_model: int,
    max_predicted_loss: float,
    allowed_coordinates: frozenset[tuple[int, float]] | None = None,
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
        ratios = (
            SUGGESTION_RATIOS
            if allowed_coordinates is None
            else tuple(sorted({ratio for _, ratio in allowed_coordinates}))
        )
        for ratio in (value for value in ratios if ratio_min <= value <= ratio_max):
            for row in _planning_models(fit.spec, max_d_model=max_d_model).values():
                d_model = int(row["d_model"])
                if allowed_coordinates is not None and not _is_coordinate_allowed(
                    allowed_coordinates, d_model, ratio
                ):
                    continue
                if d_model < fit.observed_width_min or d_model > max_d_model:
                    continue
                if _is_observed(family_evidence, d_model, ratio):
                    continue
                candidate = _candidate_at_ratio(
                    ratio,
                    assume_samples=assume_samples,
                    fit=fit,
                    row=row,
                    max_d_model=max_d_model,
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
    row: dict[str, Any],
    max_d_model: int,
) -> BudgetCandidate | None:
    config = _planning_config(
        fit.spec,
        d_model=int(row["d_model"]),
        training_ratio=training_ratio,
    )
    if config is None:
        return None
    steps = config.run.steps
    batch_size = config.run.batch_size
    if steps <= 0 or steps * batch_size > assume_samples:
        return None
    profile = _planning_profiles(fit.spec, max_d_model=max_d_model).get(
        (int(row["d_model"]), batch_size)
    )
    if profile is None:
        return None
    milliseconds_per_step = float(profile["measured_wall_ms_per_step"])
    rate = hardware_dollars_per_second(str(profile["gpu"]), int(profile["cpu_cores"]))
    cost = steps * milliseconds_per_step / 1000.0 * rate
    samples = steps * batch_size
    candidate = BudgetCandidate(
        budget=cost,
        family=fit.spec.family,
        d_model=int(row["d_model"]),
        params=int(row["params"]),
        gpu=str(profile["gpu"]),
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
        width_extrapolated=int(row["d_model"]) > fit.observed_width_max,
        config=fit.spec.config,
    )
    return replace(candidate, predicted_loss=_predict_candidate(candidate, fit))


def _predict_candidate(candidate: BudgetCandidate, fit: FamilyFit) -> float:
    return fit.law.predict(candidate.params, candidate.samples)


def _fit_noise(fit: FamilyFit) -> float:
    return max(fit.law.rmse, 0.005)


@cache
def _throughput_models(spec: FamilySpec) -> dict[str, dict[str, Any]]:
    with spec.throughput.open("rb") as handle:
        return tomllib.load(handle)["models"]


@cache
def _planning_models(
    spec: FamilySpec, *, max_d_model: int
) -> dict[str, dict[str, Any]]:
    models = {name: dict(row) for name, row in _throughput_models(spec).items()}
    if spec.family != "dense":
        return models
    largest_measured = max(int(row["d_model"]) for row in models.values())
    first_extrapolated = (
        (largest_measured // DENSE_EXTRAPOLATION_STEP) + 1
    ) * DENSE_EXTRAPOLATION_STEP
    for d_model in range(
        first_extrapolated,
        max_d_model + 1,
        DENSE_EXTRAPOLATION_STEP,
    ):
        params = dense_parameter_count(
            d_model=d_model,
            depth=8,
            history_length=8,
            expansion_ratio=4.0,
            activation="swiglu",
        )
        models[f"d{d_model}"] = {
            "d_model": d_model,
            "params": params,
            "flops_per_sample": _dense_flops_per_sample(d_model),
        }
    return models


@cache
def _throughput_profiles(spec: FamilySpec) -> dict[tuple[int, int], dict[str, Any]]:
    profiles = {}
    for path in (spec.throughput, *spec.throughput_variants):
        if not path.exists():
            continue
        with path.open("rb") as handle:
            rows = tomllib.load(handle)["models"].values()
        for row in rows:
            profiles[(int(row["d_model"]), int(row["batch_size"]))] = row
    return profiles


@cache
def _planning_profiles(
    spec: FamilySpec, *, max_d_model: int
) -> dict[tuple[int, int], dict[str, Any]]:
    profiles = {key: dict(row) for key, row in _throughput_profiles(spec).items()}
    if spec.family != "dense":
        return profiles
    models = _planning_models(spec, max_d_model=max_d_model)
    largest_measured_width = max(width for width, _ in profiles)
    target_widths = sorted(
        int(row["d_model"])
        for row in models.values()
        if int(row["d_model"]) >= largest_measured_width
    )
    for batch_factor in (16, 32):
        anchors = [
            row
            for (width, _), row in profiles.items()
            if width == largest_measured_width
        ]
        anchor = min(
            anchors,
            key=lambda row: abs(int(row["batch_size"]) / largest_measured_width - batch_factor),
        )
        anchor_flops_per_step = int(anchor["flops_per_sample"]) * int(
            anchor["batch_size"]
        )
        for d_model in target_widths:
            row = models[f"d{d_model}"]
            batch_size = batch_factor * d_model
            if (d_model, batch_size) in profiles:
                continue
            flops_per_step = int(row["flops_per_sample"]) * batch_size
            steps_1x = round(
                DENSE_SAMPLES_PER_PARAMETER * int(row["params"]) / batch_size
            )
            profiles[(d_model, batch_size)] = {
                "d_model": d_model,
                "batch_size": batch_size,
                "steps_1x": steps_1x,
                "samples_1x": steps_1x * batch_size,
                "flops_per_sample": int(row["flops_per_sample"]),
                "measured_wall_ms_per_step": float(anchor["measured_wall_ms_per_step"])
                * flops_per_step
                / anchor_flops_per_step,
                "gpu": anchor["gpu"],
                "cpu_cores": anchor["cpu_cores"],
            }
    return profiles


@cache
def _dense_flops_per_sample(d_model: int) -> int:
    return measure_training_flops_per_sample(
        DenseChessNetConfig(
            d_model=d_model,
            depth=8,
            history_length=8,
            expansion_ratio=4.0,
            activation="swiglu",
            precision="mxfp8",
            input_pipeline="overlap",
        ),
        batch_size=1,
    )


def _planning_config(
    spec: FamilySpec,
    *,
    d_model: int,
    training_ratio: float,
) -> TrainingConfig | PlanningConfig | None:
    try:
        return load_training_config(
            spec.config,
            d_model=d_model,
            training_ratio=training_ratio,
        )
    except ValueError:
        if spec.family != "dense" or f"d{d_model}" in _throughput_models(spec):
            return None
    params = dense_parameter_count(
        d_model=d_model,
        depth=8,
        history_length=8,
        expansion_ratio=4.0,
        activation="swiglu",
    )
    batch_size = 32 * d_model
    steps = round(training_ratio * DENSE_SAMPLES_PER_PARAMETER * params / batch_size)
    minimum_steps = (
        DENSE_MINIMUM_STEPS_COEFFICIENT
        * d_model**DENSE_MINIMUM_STEPS_WIDTH_EXPONENT
    )
    if steps < minimum_steps:
        batch_size //= 2
        steps *= 2
    if steps < minimum_steps:
        return None
    return PlanningConfig(
        run=PlanningRun(
            steps=steps,
            batch_size=batch_size,
            training_ratio=training_ratio,
        )
    )


def _is_observed(evidence: FamilyEvidence, d_model: int, ratio: float) -> bool:
    return any(
        width == d_model and math.isclose(observed_ratio, ratio, rel_tol=0.01)
        for width, observed_ratio in evidence.observed_coordinates
    )


def _is_coordinate_allowed(
    allowed: frozenset[tuple[int, float]], d_model: int, ratio: float
) -> bool:
    return any(
        width == d_model and math.isclose(allowed_ratio, ratio, rel_tol=0.01)
        for width, allowed_ratio in allowed
    )


if __name__ == "__main__":
    budget_planner()
