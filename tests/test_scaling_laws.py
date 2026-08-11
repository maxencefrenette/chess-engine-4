from __future__ import annotations

import math
from pathlib import Path

import pytest

from chess_engine_4.training.scaling_laws import (
    fit_loss_power_law,
    fit_sigmoid_law,
    fit_undertraining_loss_law,
    read_best_runs,
    read_dense_scaling_points,
)


def test_read_best_runs() -> None:
    best_runs = read_best_runs(Path("experiments/best-runs-dense.toml"))
    assert [run.budget for run in best_runs] == [
        "d32",
        "d64",
        "d128",
        "d256",
        "d512",
        "d768",
        "d1024",
        "d1280",
    ]
    assert all(run.training_ratio == 0.2 for run in best_runs)


def test_dense_scaling_points_exclude_d32_and_very_short_runs() -> None:
    points = read_dense_scaling_points(Path("experiments/best-runs-dense.toml"))

    assert len(points) == 19
    assert min(params for params, _, _ in points) == 979_488
    assert all(samples / params >= 4.99 for params, samples, _ in points)


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


def test_read_best_runs_excludes_stale_rows(tmp_path: Path) -> None:
    source = Path("experiments/best-runs-dense.toml").read_text(encoding="utf-8")
    source = source.replace("[runs.d32]", "[runs.d32]\nstale = true", 1)
    path = tmp_path / "best-runs.toml"
    path.write_text(source, encoding="utf-8")

    current = read_best_runs(path)
    historical = read_best_runs(path, include_stale=True)

    assert all(result.budget != "d32" for result in current)
    assert next(result for result in historical if result.budget == "d32").stale
