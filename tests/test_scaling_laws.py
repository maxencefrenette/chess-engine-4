from __future__ import annotations

from pathlib import Path

from chess_engine_4.training.scaling_laws import (
    extrapolate,
    fit_scaling_laws,
    non_embedding_parameter_count,
    parameter_count,
    read_best_runs,
    round_to_batch_ladder,
    round_to_lr_ladder,
    write_report_artifacts,
)


def test_read_best_runs_and_extrapolate() -> None:
    best_runs = read_best_runs(Path("experiments/best-runs.toml"))
    assert [run.budget for run in best_runs] == ["1e13", "1e14", "1e15"]

    laws = fit_scaling_laws(best_runs)
    suggestion = extrapolate(laws, 1e16)

    assert 0.28 < laws.policy_top1_tail_mean < 0.29
    assert suggestion.d_model % 64 == 0
    assert suggestion.depth >= 2
    assert suggestion.batch_size in {1024, 1536, 2048}
    assert suggestion.lr in {0.00015, 0.0002, 0.0003}
    assert suggestion.actual_non_embedding_params > best_runs[-1].non_embedding_params


def test_parameter_count_formulas_match_current_baselines() -> None:
    assert non_embedding_parameter_count(d_model=48, depth=1) == 27696
    assert parameter_count(d_model=48, depth=1) == 463094


def test_rounding_ladders() -> None:
    assert round_to_batch_ladder(1400) == 1536
    assert round_to_lr_ladder(0.00019) == 0.0002


def test_write_report_artifacts(tmp_path: Path) -> None:
    best_runs = read_best_runs(Path("experiments/best-runs.toml"))
    laws = fit_scaling_laws(best_runs)
    suggestion = extrapolate(laws, 1e16)

    write_report_artifacts(
        output_dir=tmp_path,
        best_results=best_runs,
        laws=laws,
        suggestion=suggestion,
        config="configs/1e15.toml",
        gpu="t4",
    )

    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "loss.svg").exists()
    assert (tmp_path / "policy_top1.svg").exists()
    assert (tmp_path / "model_size.svg").exists()
    assert (tmp_path / "data_samples.svg").exists()
    assert (tmp_path / "batch_size.svg").exists()
    assert (tmp_path / "learning_rate.svg").exists()
    assert (tmp_path / "runtime.svg").exists()
    report = (tmp_path / "README.md").read_text()
    assert "![Loss fit](loss.svg)" in report
    assert "![Policy top-1](policy_top1.svg)" in report
    assert "Policy Top-1 Fit" not in report
    assert "Probe Commands" not in report
