from __future__ import annotations

from pathlib import Path

import pytest

from chess_engine_4.training.scaling_laws import (
    extrapolate,
    fit_scaling_laws,
    parameter_count,
    read_best_runs,
    round_to_batch_ladder,
    round_to_lr_ladder,
    write_report_artifacts,
)


def test_read_best_runs_and_extrapolate() -> None:
    best_runs = read_best_runs(Path("experiments/best-runs-mlp.toml"))
    assert [run.budget for run in best_runs] == ["1e18", "1e19", "1e20", "1e21"]

    laws = fit_scaling_laws(best_runs)
    suggestion = extrapolate(laws, 1e22)

    assert 0.28 < laws.policy_top1.predict(1e18) < 0.29
    assert suggestion.d_model % 64 == 0
    assert suggestion.depth >= 5
    assert suggestion.batch_size == 24576
    assert suggestion.lr == pytest.approx(0.0001)
    assert suggestion.actual_params > best_runs[-1].params


def test_parameter_count_formulas_match_current_baselines() -> None:
    assert parameter_count(d_model=32, depth=1) == 303142
    assert parameter_count(model_kind="transformer64", d_model=96, depth=2) == 340612
    assert parameter_count(model_kind="mlp_moe", d_model=32, depth=1) == 389670


def test_rounding_ladders() -> None:
    assert round_to_batch_ladder(1400) == 1536
    assert round_to_lr_ladder(0.00019) == 0.0002


def test_write_report_artifacts(tmp_path: Path) -> None:
    best_runs = read_best_runs(Path("experiments/best-runs-mlp.toml"))
    laws = fit_scaling_laws(best_runs)
    suggestion = extrapolate(laws, 1e20)

    write_report_artifacts(
        output_dir=tmp_path,
        best_results=best_runs,
        laws=laws,
        suggestion=suggestion,
        config="configs/mlp/1e19.toml",
        gpu="t4",
    )

    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "loss.svg").exists()
    assert (tmp_path / "policy_top1.svg").exists()
    assert (tmp_path / "model_size.svg").exists()
    assert (tmp_path / "datapoints_per_parameter.svg").exists()
    assert (tmp_path / "data_samples.svg").exists()
    assert (tmp_path / "batch_size.svg").exists()
    assert (tmp_path / "learning_rate.svg").exists()
    assert not (tmp_path / "runtime.svg").exists()
    report = (tmp_path / "README.md").read_text()
    assert "![Loss fit](loss.svg)" in report
    assert "![Policy top-1](policy_top1.svg)" in report
    assert "![Datapoints per parameter](datapoints_per_parameter.svg)" in report
    assert "runtime_sec" not in report
    assert "Policy Top-1 Fit" not in report
    assert "Probe Commands" not in report
