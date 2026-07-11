from __future__ import annotations

import pytest

from chess_engine_4.model import DenseChessNetConfig
from chess_engine_4.training.flops import (
    measure_training_flops_per_sample,
    step_adjusted_compute,
    steps_for_compute_budget,
)


def test_measures_dense_training_flops_on_meta() -> None:
    config = DenseChessNetConfig(d_model=32, depth=1, expansion_ratio=2.0)

    assert measure_training_flops_per_sample(config, batch_size=4) > 0


def test_steps_for_compute_budget_rounds_up() -> None:
    assert (
        steps_for_compute_budget(
            compute_budget=101,
            flops_per_sample=10,
            batch_size=2,
        )
        == 6
    )


def test_steps_for_compute_budget_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="compute_budget"):
        steps_for_compute_budget(compute_budget=0, flops_per_sample=10, batch_size=2)


def test_steps_for_step_adjusted_compute_target() -> None:
    assert (
        steps_for_compute_budget(
            compute_budget=2_000,
            flops_per_sample=10,
            batch_size=2,
            step_penalty_k=2.0,
        )
        == 10
    )


def test_step_adjusted_compute_matches_flops_at_k_one() -> None:
    assert (
        step_adjusted_compute(
            flops_per_sample=10,
            batch_size=2,
            steps=5,
            step_penalty_k=1.0,
        )
        == 100
    )


def test_step_adjusted_compute_penalizes_steps_above_k_one() -> None:
    assert (
        step_adjusted_compute(
            flops_per_sample=10,
            batch_size=2,
            steps=5,
            step_penalty_k=2.0,
        )
        == 500
    )
