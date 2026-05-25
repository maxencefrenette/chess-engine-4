from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from chess_engine_4.model import (
    MlpChessNet,
    MlpChessNetConfig,
    MlpMoeChessNet,
    MlpMoeChessNetConfig,
)
from chess_engine_4.training.flops import (
    measure_training_flops_per_sample,
    step_adjusted_compute,
    steps_for_compute_budget,
)


def test_measure_training_flops_per_sample_uses_meta_model() -> None:
    with torch.device("meta"):
        model = MlpChessNet(MlpChessNetConfig(d_model=32, depth=1, mlp_ratio=2.0))

    flops_per_sample = measure_training_flops_per_sample(model, batch_size=4)

    assert flops_per_sample > 0


def test_measure_training_flops_per_sample_supports_moe_grouped_mm() -> None:
    with torch.device("meta"):
        model = MlpMoeChessNet(
            MlpMoeChessNetConfig(d_model=32, depth=1, num_experts=16, expert_mlp_ratio=2.0)
        )

    flops_per_sample = measure_training_flops_per_sample(model, batch_size=4)

    assert flops_per_sample > 0


def test_grouped_mm_profiler_flops_sentinel() -> None:
    with torch.device("meta"):
        x = torch.empty(8, 32, dtype=torch.bfloat16)
        weight = torch.empty(1, 32, 64, dtype=torch.bfloat16)
        offsets = torch.empty(1, dtype=torch.int32)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_flops=True,
        acc_events=True,
    ) as profiler:
        F.grouped_mm(x, weight, offs=offsets)

    grouped_mm_flops = sum(
        event.flops or 0 for event in profiler.key_averages() if "_grouped_mm" in event.key
    )
    assert grouped_mm_flops == 0


def test_moe_grouped_mm_extra_flops_match_fixed_formula() -> None:
    with torch.device("meta"):
        ratio_2 = MlpMoeChessNet(
            MlpMoeChessNetConfig(
                d_model=32,
                depth=1,
                num_experts=16,
                num_experts_per_token=2,
                expert_mlp_ratio=2.0,
            )
        )
        ratio_4 = MlpMoeChessNet(
            MlpMoeChessNetConfig(
                d_model=32,
                depth=1,
                num_experts=16,
                num_experts_per_token=2,
                expert_mlp_ratio=4.0,
            )
        )

    expected_delta = 3 * 2 * 6 * 32 * (128 - 64)
    measured_delta = (
        ratio_4.extra_training_flops_per_sample() - ratio_2.extra_training_flops_per_sample()
    )

    assert measured_delta == expected_delta


def test_measure_training_flops_per_sample_rejects_real_model() -> None:
    model = MlpChessNet(MlpChessNetConfig(d_model=32, depth=1, mlp_ratio=2.0))

    with pytest.raises(ValueError, match="meta device"):
        measure_training_flops_per_sample(model, batch_size=4)


def test_steps_for_compute_budget_rounds_up() -> None:
    assert steps_for_compute_budget(
        compute_budget=101,
        flops_per_sample=10,
        batch_size=2,
    ) == 6


def test_steps_for_compute_budget_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="compute_budget"):
        steps_for_compute_budget(compute_budget=0, flops_per_sample=10, batch_size=2)


def test_steps_for_step_adjusted_compute_target() -> None:
    assert steps_for_compute_budget(
        compute_budget=2_000,
        flops_per_sample=10,
        batch_size=2,
        step_penalty_k=2.0,
    ) == 10


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
