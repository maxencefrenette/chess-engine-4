from __future__ import annotations

import pytest
import torch

from chess_engine_4.model import MlpChessNet, MlpChessNetConfig
from chess_engine_4.training.flops import measure_training_flops_per_sample, steps_for_flops_target


def test_measure_training_flops_per_sample_uses_meta_model() -> None:
    with torch.device("meta"):
        model = MlpChessNet(MlpChessNetConfig(d_model=32, depth=1, mlp_ratio=2.0))

    flops_per_sample = measure_training_flops_per_sample(model, batch_size=4)

    assert flops_per_sample > 0


def test_measure_training_flops_per_sample_rejects_real_model() -> None:
    model = MlpChessNet(MlpChessNetConfig(d_model=32, depth=1, mlp_ratio=2.0))

    with pytest.raises(ValueError, match="meta device"):
        measure_training_flops_per_sample(model, batch_size=4)


def test_steps_for_flops_target_rounds_up() -> None:
    assert steps_for_flops_target(
        flops_target=101,
        flops_per_sample=10,
        batch_size=2,
    ) == 6


def test_steps_for_flops_target_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="flops_target"):
        steps_for_flops_target(flops_target=0, flops_per_sample=10, batch_size=2)
