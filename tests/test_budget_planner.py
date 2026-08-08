from pathlib import Path

import pytest

from chess_engine_4.training.budget_planner import (
    FAMILY_SPECS,
    bootstrap_family_fits,
    estimate_budget,
    fit_families,
    plan_budget,
    read_dataset_samples,
    read_family_evidence,
    suggest_runs,
)


def planner_inputs():
    evidence = read_family_evidence(FAMILY_SPECS)
    return evidence, fit_families(evidence)


def test_read_current_dataset_samples() -> None:
    assert read_dataset_samples(Path("experiments/training-data.toml")) == 3_949_735_220


def test_assumed_samples_hard_caps_plan() -> None:
    _, fits = planner_inputs()
    candidate = plan_budget(
        100.0,
        assume_samples=1_000_000,
        fits=fits,
        ratio_extrapolation_limit=100,
    )

    assert candidate is not None
    assert candidate.samples <= 1_000_000
    assert candidate.sample_limited


def test_planner_compares_dense_and_moe() -> None:
    _, fits = planner_inputs()
    candidate = plan_budget(5.0, assume_samples=3_949_735_220, fits=fits)

    assert candidate is not None
    assert candidate.family == "moe64a2"
    assert candidate.estimated_cost <= candidate.budget


def test_planner_rejects_unbounded_ratio_extrapolation() -> None:
    _, fits = planner_inputs()
    candidate = plan_budget(100.0, assume_samples=3_949_735_220, fits=fits)

    assert candidate is not None
    assert candidate.training_ratio <= {"dense": 1.0, "moe64a2": 0.1}[candidate.family]


@pytest.mark.parametrize("budget, samples", [(0.0, 100), (1.0, 0)])
def test_planner_rejects_non_positive_inputs(budget: float, samples: int) -> None:
    with pytest.raises(ValueError):
        plan_budget(budget, assume_samples=samples, fits=[])


def test_budget_estimate_reports_bootstrap_uncertainty() -> None:
    evidence, fits = planner_inputs()
    ensembles = bootstrap_family_fits(evidence, fits, samples=20, seed=7)
    estimate = estimate_budget(
        10.0,
        assume_samples=3_949_735_220,
        fits=fits,
        bootstrap_fits=ensembles,
    )

    assert estimate is not None
    assert estimate.loss_lower < estimate.loss_upper
    assert 0 <= estimate.selection_probability <= 1


def test_value_of_information_runs_are_unmeasured_and_cheap() -> None:
    evidence, fits = planner_inputs()
    ensembles = bootstrap_family_fits(evidence, fits, samples=20, seed=11)
    suggestions = suggest_runs(
        focus_budget=10.0,
        count=2,
        max_cost=1.0,
        assume_samples=3_949_735_220,
        evidence=evidence,
        fits=fits,
        bootstrap_fits=ensembles,
    )

    assert len(suggestions) == 2
    assert all(suggestion.estimated_cost <= 1.0 for suggestion in suggestions)
    for suggestion in suggestions:
        family_evidence = next(
            item for item in evidence if item.spec.family == suggestion.family
        )
        assert not any(
            width == suggestion.d_model and ratio == pytest.approx(suggestion.training_ratio)
            for width, ratio in family_evidence.observed_coordinates
        )


def test_twice_dataset_prefers_direct_ten_dollar_training() -> None:
    evidence, fits = planner_inputs()
    ensembles = bootstrap_family_fits(evidence, fits, samples=50, seed=2026)
    comparisons = suggest_runs(
        focus_budget=10.0,
        count=3,
        max_cost=1.0,
        assume_samples=2 * 3_949_735_220,
        evidence=evidence,
        fits=fits,
        bootstrap_fits=ensembles,
    )

    assert comparisons
    assert comparisons[0].expected_loss_improvement < 0
    assert all(0 <= comparison.probability_improves <= 1 for comparison in comparisons)
