from __future__ import annotations

import pytest

from chess_engine_4.model import DenseChessNetConfig, Moe64A2ChessNetConfig
from chess_engine_4.training.flops import measure_training_flops_per_sample


def test_measures_dense_training_flops_on_meta() -> None:
    config = DenseChessNetConfig(d_model=32, depth=1, expansion_ratio=2.0)

    assert measure_training_flops_per_sample(config, batch_size=4) > 0


def test_moe_flops_count_only_active_experts() -> None:
    dense = DenseChessNetConfig(d_model=64, depth=2, expansion_ratio=4.0)
    moe = Moe64A2ChessNetConfig(d_model=64, depth=2, expansion_ratio=2.0)

    dense_flops = measure_training_flops_per_sample(dense, batch_size=4)
    moe_flops = measure_training_flops_per_sample(moe, batch_size=4)

    assert moe_flops / dense_flops == pytest.approx(1.0, rel=0.05)
