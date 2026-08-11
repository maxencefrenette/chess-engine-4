from __future__ import annotations

import math
from pathlib import Path

import pytest

from chess_engine_4.training.scaling_laws import (
    SkalingLaw,
    fit_sigmoid_law,
    fit_skaling_law,
    read_best_runs,
    read_dense_scaling_points,
    read_scaling_points,
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


def test_skaling_inverse_recovers_samples_at_fixed_model_size() -> None:
    law = SkalingLaw(
        model_coefficient=2.0,
        data_coefficient=3.0,
        model_exponent=0.4,
        data_exponent=0.3,
        coupling=0.7,
        floor=2.0,
        rmse=0.0,
    )
    loss = law.predict(100_000_000, 500_000_000)

    assert law.samples_for_loss(100_000_000, loss) == pytest.approx(500_000_000)


def test_fixed_floor_skaling_fit_reuses_floor() -> None:
    source = SkalingLaw(2.0, 3.0, 0.4, 0.3, 0.7, 2.2, 0.0)
    points = [
        (params, samples, source.predict(params, samples))
        for params, samples in (
            (1_000_000, 10_000_000),
            (1_000_000, 30_000_000),
            (1_000_000, 100_000_000),
            (3_000_000, 10_000_000),
            (10_000_000, 10_000_000),
            (30_000_000, 10_000_000),
            (100_000_000, 10_000_000),
        )
    ]

    fit = fit_skaling_law(points, fixed_floor=source.floor, restarts=8)

    assert fit.floor == source.floor
    assert fit.rmse < 1e-5


def test_read_quantile_moe_scaling_points() -> None:
    points = read_scaling_points(Path("experiments/best-runs-moe64a2.toml"))

    assert len(points) == 11
    assert min(params for params, _, _ in points) == 106_213_792


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
