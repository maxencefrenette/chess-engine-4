from pathlib import Path

import pytest

from chess_engine_4.training.budget_planner import (
    DEFAULT_ASSUMED_SAMPLES,
    DEFAULT_FAMILIES,
    DEFAULT_MAX_D_MODEL,
    _planning_models,
    _planning_profiles,
    bootstrap_family_fits,
    candidates_for_family,
    estimate_budget,
    fit_families,
    plan_budget,
    read_dataset_samples,
    read_family_evidence,
    suggest_runs,
)
from chess_engine_4.training.config import load_training_config
from chess_engine_4.training.families import FAMILIES, FAMILY_SPECS
from chess_engine_4.training.scaling_laws import SkalingLaw


def planner_inputs():
    evidence = read_family_evidence(FAMILY_SPECS)
    return evidence, fit_families(evidence)


def test_read_current_dataset_samples() -> None:
    assert read_dataset_samples(Path("experiments/training-data.toml")) == 8_020_779_820


def test_planner_defaults_to_dense_with_25b_positions_and_d2560_ceiling() -> None:
    assert DEFAULT_FAMILIES == ("dense",)
    assert DEFAULT_ASSUMED_SAMPLES == 25_000_000_000
    assert DEFAULT_MAX_D_MODEL == 2560


def test_planner_rejects_sample_cap_below_recipe_minimum_steps() -> None:
    _, fits = planner_inputs()
    candidate = plan_budget(
        100.0,
        assume_samples=1_000_000,
        fits=fits,
        ratio_extrapolation_limit=100,
    )

    assert candidate is None


def test_assumed_samples_hard_caps_trainable_plan() -> None:
    _, fits = planner_inputs()
    candidate = plan_budget(
        100.0,
        assume_samples=20_000_000,
        fits=fits,
        ratio_extrapolation_limit=100,
    )

    assert candidate is not None
    assert candidate.samples <= 20_000_000
    assert candidate.sample_limited


def test_planner_uses_adaptive_dense_throughput() -> None:
    _, fits = planner_inputs()
    candidate = plan_budget(5.0, assume_samples=3_949_735_220, fits=fits)

    assert candidate is not None
    assert candidate.family == "dense"
    assert candidate.estimated_cost <= candidate.budget


def test_both_families_use_skaling_and_share_floor() -> None:
    _, fits = planner_inputs()

    assert isinstance(next(fit.law for fit in fits if fit.spec.family == "dense"), SkalingLaw)
    assert all(isinstance(fit.law, SkalingLaw) for fit in fits)
    dense = next(fit for fit in fits if fit.spec.family == "dense")
    moe = next(fit for fit in fits if fit.spec.family == "moe64a2")
    assert moe.law.floor == dense.law.floor
    candidates = candidates_for_family(
        100.0,
        assume_samples=8_020_779_820,
        fit=dense,
    )
    widest = max(candidates, key=lambda candidate: candidate.d_model)
    assert widest.d_model == 2560
    assert widest.width_extrapolated
    assert "--d-model 2560" in widest.command


def test_planner_uses_family_wide_horizon_with_two_x_extrapolation() -> None:
    _, fits = planner_inputs()
    dense = next(fit for fit in fits if fit.spec.family == "dense")
    candidates = candidates_for_family(
        1_000_000.0,
        assume_samples=1_000_000_000_000,
        fit=dense,
        max_d_model=1280,
    )

    assert dense.observed_ratio_max == 2.0
    assert max(candidate.training_ratio for candidate in candidates) == pytest.approx(4.0)
    d1280_ratios = [
        candidate.training_ratio for candidate in candidates if candidate.d_model == 1280
    ]
    assert max(d1280_ratios) == pytest.approx(4.0)


def test_extrapolated_width_cost_assumes_largest_width_mfu() -> None:
    spec = FAMILIES["dense"]
    models = _planning_models(spec, max_d_model=2560)
    profiles = _planning_profiles(spec, max_d_model=2560)
    anchor = profiles[(1280, 49_152)]
    extrapolated = profiles[(2560, 32 * 2560)]
    expected_ratio = (
        int(models["d2560"]["flops_per_sample"]) * int(extrapolated["batch_size"])
    ) / (int(anchor["flops_per_sample"]) * int(anchor["batch_size"]))

    assert float(extrapolated["measured_wall_ms_per_step"]) / float(
        anchor["measured_wall_ms_per_step"]
    ) == pytest.approx(expected_ratio)


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
        config = load_training_config(
            suggestion.config,
            d_model=suggestion.d_model,
            training_ratio=suggestion.training_ratio,
        )
        assert config.run.steps == suggestion.steps
        assert config.run.batch_size * config.run.steps == suggestion.samples
        assert "--steps" not in suggestion.command


def test_spiked_scaling_run_is_still_an_observed_coordinate() -> None:
    evidence = read_family_evidence(FAMILY_SPECS)
    dense = next(item for item in evidence if item.spec.family == "dense")

    assert (256, 0.1) in dense.observed_coordinates
