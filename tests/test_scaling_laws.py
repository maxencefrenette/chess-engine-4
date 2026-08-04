from __future__ import annotations

import math
from pathlib import Path

import pytest

from chess_engine_4.training.scaling_laws import (
    extrapolate,
    fit_loss_power_law,
    fit_scaling_laws,
    fit_sigmoid_law,
    fit_undertraining_loss_law,
    format_report,
    parameter_count,
    read_best_runs,
)


def test_read_best_runs_and_extrapolate() -> None:
    best_runs = read_best_runs(Path("experiments/best-runs-dense.toml"))
    assert [run.budget for run in best_runs] == [
        "d32",
        "d64",
        "d128",
        "d256",
    ]

    laws = fit_scaling_laws(best_runs)
    suggestion = extrapolate(laws, 1e18)

    assert 0.28 < laws.policy_top1.predict(1e14) < 0.30
    assert suggestion.d_model % 32 == 0
    assert suggestion.depth >= 5
    assert suggestion.batch_size > best_runs[-1].batch_size
    assert suggestion.lr < best_runs[-1].lr
    assert suggestion.actual_params > best_runs[-1].params


def test_parameter_count_formulas_match_current_baselines() -> None:
    assert parameter_count(d_model=32, depth=1) == 306176


def test_read_best_runs_tracks_non_frontier_ratios_separately() -> None:
    path = Path("experiments/best-runs-dense.toml")

    frontier = read_best_runs(path)
    tracked = read_best_runs(path, include_non_frontier=True)

    assert all(run.training_ratio == 1.0 for run in frontier)
    assert {run.training_ratio for run in tracked if not run.frontier} == {
        0.125,
        0.25,
        0.5,
        2.0,
    }


def test_undertraining_loss_law_penalizes_shorter_runs() -> None:
    baseline = fit_loss_power_law([(1e13, 4.0), (1e14, 3.6), (1e15, 3.3), (1e16, 3.1)])
    law = fit_undertraining_loss_law(
        baseline,
        [
            (1e14, 0.25, 4.3),
            (1e14, 0.5, 3.9),
            (1e15, 0.25, 3.8),
            (1e15, 0.5, 3.5),
        ],
    )

    assert law.predict(1e15, 0.25) > law.predict(1e15, 0.5)
    assert law.predict(1e15, 0.5) > law.predict(1e15, 1.0)


def test_sigmoid_law_is_bounded_and_recovers_synthetic_curve() -> None:
    law = fit_sigmoid_law(
        (10.0**exponent, 0.7 / (1.0 + math.exp(-0.5 * (exponent - 15))))
        for exponent in range(12, 19)
    )

    assert law.ceiling == pytest.approx(0.7)
    assert law.predict(1e30) <= law.ceiling
    assert law.rmse < 1e-8


def test_format_report() -> None:
    best_runs = read_best_runs(Path("experiments/best-runs-dense.toml"))
    laws = fit_scaling_laws(best_runs)
    suggestion = extrapolate(laws, 1e20)

    report = format_report(
        best_results=best_runs,
        laws=laws,
        suggestion=suggestion,
        config="configs/dense.py",
        gpu="l4",
    )

    assert "L(C) =" in report
    assert "uv run train-modal" in report


def test_read_best_runs_excludes_stale_rows(tmp_path: Path) -> None:
    source = Path("experiments/best-runs-dense.toml").read_text(encoding="utf-8")
    source = source.replace("[runs.d32]", "[runs.d32]\nstale = true", 1)
    path = tmp_path / "best-runs.toml"
    path.write_text(source, encoding="utf-8")

    current = read_best_runs(path)
    historical = read_best_runs(path, include_stale=True)

    assert all(result.budget != "d32" for result in current)
    assert next(result for result in historical if result.budget == "d32").stale
