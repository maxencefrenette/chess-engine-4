from __future__ import annotations

from pathlib import Path

import pytest

from chess_engine_4.training.scaling_laws import (
    extrapolate,
    fit_scaling_laws,
    format_report,
    parameter_count,
    read_best_runs,
    round_to_batch_ladder,
    round_to_lr_ladder,
)


def test_read_best_runs_and_extrapolate() -> None:
    best_runs = read_best_runs(Path("experiments/best-runs-dense.toml"))
    assert [run.budget for run in best_runs] == [
        "1e18",
        "3e18",
        "1e19",
        "3e19",
        "1e20",
        "1e21",
        "1e22",
        "1e23",
    ]

    laws = fit_scaling_laws(best_runs)
    suggestion = extrapolate(laws, 1e24)

    assert 0.29 < laws.policy_top1.predict(1e18) < 0.31
    assert suggestion.d_model % 64 == 0
    assert suggestion.depth >= 5
    assert suggestion.batch_size == 262144
    assert suggestion.lr == pytest.approx(0.0002)
    assert suggestion.actual_params > best_runs[-1].params


def test_parameter_count_formulas_match_current_baselines() -> None:
    assert parameter_count(d_model=32, depth=1) == 306176


def test_rounding_ladders() -> None:
    assert round_to_batch_ladder(1400) == 1536
    assert round_to_batch_ladder(154_454) == 131_072
    assert round_to_lr_ladder(0.00019) == 0.0002


def test_format_report() -> None:
    best_runs = read_best_runs(Path("experiments/best-runs-dense.toml"))
    laws = fit_scaling_laws(best_runs)
    suggestion = extrapolate(laws, 1e20)

    report = format_report(
        best_results=best_runs,
        laws=laws,
        suggestion=suggestion,
        config="configs/dense/1e19.toml",
        gpu="l4",
    )

    assert "L(C) =" in report
    assert "uv run train-modal" in report
