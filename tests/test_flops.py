from __future__ import annotations

from chess_engine_4.model import DenseChessNetConfig
from chess_engine_4.training.flops import (
    measure_training_flops_per_sample,
    modified_compute,
)


def test_measures_dense_training_flops_on_meta() -> None:
    config = DenseChessNetConfig(d_model=32, depth=1, expansion_ratio=2.0)

    assert measure_training_flops_per_sample(config, batch_size=4) > 0


def test_modified_compute_uses_fixed_squared_step_penalty() -> None:
    assert (
        modified_compute(
            flops_per_sample=10,
            batch_size=2,
            steps=5,
        )
        == 500
    )
